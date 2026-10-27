"""Run python -m src.preprocess from the repository root."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from .analytics import *
ROOT=Path(__file__).resolve().parents[1]

def read_apartment(path,unit='kW',monitoring_start=None,monitoring_end=None):
    raw=pd.read_csv(path);ts=pd.to_datetime(raw.Timestamp,format='mixed',dayfirst=True,errors='coerce')
    stats=dict(source_rows=len(raw),invalid_timestamps=int(ts.isna().sum()),duplicate_timestamps=int(ts.duplicated().sum()))
    p=raw.drop(columns=['Timestamp','Apartment'],errors='ignore').apply(pd.to_numeric,errors='coerce');p.index=ts
    p=p.loc[p.index.notna()].sort_index();stats['negative_cells']=int((p<0).sum().sum())
    # Absolute each source reading first, then collapse duplicate timestamps.
    p=p.abs().replace([np.inf,-np.inf],np.nan)
    allowed=pd.Series(True,index=p.index)
    if monitoring_start is not None:allowed &= p.index>=pd.Timestamp(monitoring_start)
    if monitoring_end is not None:allowed &= p.index<pd.Timestamp(monitoring_end)
    stats['outside_monitoring_window_rows']=int((~allowed).sum())
    p=p.loc[allowed.to_numpy()].groupby(level=0).mean()
    if unit=='W':p=p/1000
    elif unit=='interval_kWh':p=p/DT_HOURS
    rounded=p.index.round('5min');ok=np.abs((p.index-rounded).total_seconds())<=30
    stats['off_grid_rows']=int((~ok).sum());p=p.loc[ok];p.index=rounded[ok];p=p.groupby(level=0).mean()
    if p.empty:raise ValueError(f'No valid five-minute timestamps: {path.name}')
    p=p.reindex(pd.date_range(p.index.min(),p.index.max(),freq='5min'));p.index.name='Timestamp'
    return p,stats

def channel_mapping(apartment,columns,workbook):
    book=workbook[workbook['Apartment Name'].eq(apartment)];rows=[]
    explicit_phases={str(c).split(' - ',1)[-1].strip().lower() for c in columns}
    for c in columns:
        hits=[];phase_aliases=[]
        for _,r in book.iterrows():
            for k in range(1,7):
                if str(r.get(f'Channel {k}','')).strip()==c:
                    hits.append(str(r.Phases).strip())
                    if k==1:phase_aliases.append(str(r.Phases).strip())
        category=classify_channel(c);alias=False
        # Only use a workbook phase alias where that named R/B/Y meter is absent.
        # Every remaining electrical column is a submetered appliance/circuit.
        if category!='Phase' and c.split(' - ',1)[-1].strip().lower()=='main incomer' and len(set(phase_aliases))==1:
            alias_phase=phase_aliases[0]
            if alias_phase.lower() not in explicit_phases:
                category='Phase';alias=True
        rows.append(dict(apartment=apartment,channel=c,category=category,
            measurement_role='Apartment phase meter' if category=='Phase' else 'Submetered appliance/circuit',
            phase_alias=alias,phase=','.join(sorted(set(hits))),
            mapping_status='Missing in workbook' if not hits else ('Conflicting phases' if len(set(hits))>1 else 'Mapped'),
            source='Lodha-Apartments-Sensors-mapping_modified(1).xlsx'))
    return pd.DataFrame(rows)

def prepare_weather(root):
    w=pd.read_csv(root/'data/raw/Palava weather data (1).csv');w['Timestamp']=pd.to_datetime(w.date,dayfirst=True,errors='coerce')
    w=w.drop(columns='date').dropna(subset=['Timestamp']).groupby('Timestamp').mean(numeric_only=True).sort_index()
    w['temperature']=w.temperature.where(w.temperature.between(-10,60));w['humidity']=w.humidity.where(w.humidity.between(0,100))
    for c in ['irradiance_w_m2','wind_speed_m_s','wind_direction_deg','rainfall_mm','pressure_hpa']:w[c]=np.nan
    ext=root/'data/raw/weather_extension.csv'
    if ext.exists():
        e=pd.read_csv(ext,parse_dates=['Timestamp']).set_index('Timestamp').groupby(level=0).mean(numeric_only=True);w=w.combine_first(e).sort_index()
    w=w.join(psychrometrics(w.temperature,w.humidity));w.to_parquet(root/'data/processed/weather.parquet');return w

def run(root=ROOT):
    out=root/'data/processed';out.mkdir(parents=True,exist_ok=True);settings=json.loads((root/'config/settings.json').read_text())
    meta=pd.read_csv(root/'config/apartments.csv').fillna('');book=pd.read_excel(root/'data/raw/Lodha-Apartments-Sensors-mapping_modified(1).xlsx')
    maps=[];schedules=[];standby=[];qa=[];thresholds=[];events=[];sources=[]
    op=root/'config/threshold_overrides.csv';ov=pd.read_csv(op) if op.exists() else pd.DataFrame()
    for _,a in meta.iterrows():
        apt=a.apartment;path=root/'data/raw'/a.file;p,stats=read_apartment(path,settings['unit'],settings.get('monitoring_start'),settings.get('monitoring_end'))
        source_power=p.copy()
        # Detect a sustained 15-minute cadence from supplied timestamps, not gaps.
        rt=pd.to_datetime(pd.read_csv(path,usecols=['Timestamp']).Timestamp,format='mixed',dayfirst=True,errors='coerce').dropna().sort_values().drop_duplicates()
        rt=rt[(rt>=pd.Timestamp(settings['monitoring_start']))&(rt<pd.Timestamp(settings['monitoring_end']))]
        dif=rt.diff().dt.total_seconds()/60
        cadence=pd.DataFrame({'time':rt.to_numpy(),'delta':dif.to_numpy()})
        cadence['day']=cadence.time.dt.floor('D')
        dailycad=cadence.groupby('day').agg(median=('delta','median'),n=('delta','count'),share15=('delta',lambda x:(x==15).mean()))
        slowdays=dailycad[(dailycad['median']==15)&(dailycad.n>=30)&(dailycad.share15>=.7)].index
        support=pd.DataFrame(False,index=p.index,columns=p.columns)
        for shift in [1,2]:
            candidate=source_power.shift(shift)
            parent=p.index-pd.Timedelta(minutes=5*shift)
            slow=np.isin(parent.floor('D'),slowdays)
            use=source_power.isna()&candidate.notna()&slow[:,None]
            p=p.where(~use,candidate)
            support|=use
        stats['fifteen_minute_cadence_days']=len(slowdays)
        stats['held_support_cells']=int(support.sum().sum())
        source_power.to_parquet(out/f'{slug(apt)}_source.parquet')
        support.to_parquet(out/f'{slug(apt)}_held.parquet')
        dailycad.to_csv(out/f'{slug(apt)}_cadence.csv')
        m=channel_mapping(apt,p.columns,book)
        flags=pd.DataFrame(False,index=p.index,columns=p.columns)
        for _,c in m.iterrows():flags[c.channel]=p[c.channel]>settings['outlier_caps_kw'][c.category]
        clean=p.mask(flags);states=pd.DataFrame(index=p.index);th=[];ev=[]
        for _,c in m[m.category.isin(APPLIANCES)].iterrows():
            spec=cluster_threshold(clean[c.channel],c.category,settings['state_noise_floor_kw'][c.category]);minimum=settings['minimum_event_minutes']
            if not ov.empty:
                match=ov[(ov.apartment==apt)&(ov.channel==c.channel)]
                if len(match):
                    o=match.iloc[-1]
                    for k in ['on_kw','off_kw','design_kw']:
                        if pd.notna(o.get(k)):spec[k]=float(o[k])
                    if pd.notna(o.get('minimum_event_minutes')):minimum=int(o.minimum_event_minutes)
                    spec['method']='Reviewed override'
            if pd.notna(spec['on_kw']) and spec['off_kw']>=spec['on_kw']:raise ValueError(f'off_kw must be below on_kw: {c.channel}')
            states[c.channel]=hysteresis(clean[c.channel],spec['on_kw'],spec['off_kw'],minimum)
            th.append(dict(apartment=apt,channel=c.channel,category=c.category,minimum_event_minutes=minimum,**spec));ev.append(event_table(clean[c.channel],states[c.channel],apt,c.category,c.channel))
        th=pd.DataFrame(th);ev=pd.concat(ev,ignore_index=True);sch=schedule_from_power(clean,states,m,th,apt);sb=standby_summary(clean,states,m,apt);sid=slug(apt)
        balance=electrical_balance(clean,m)
        for name,frame in [('power',p),('flags',flags),('states',states),('balance',balance)]:frame.to_parquet(out/f'{sid}_{name}.parquet')
        ev.to_parquet(out/f'{sid}_events.parquet',index=False);sch.to_csv(out/f'{sid}_schedule.csv',index=False)
        maps.append(m);schedules.append(sch);standby.append(sb);thresholds.append(th);events.append(ev)
        qa.append(dict(apartment=apt,start=p.index.min(),end=p.index.max(),expected_slots=len(p),channels=len(p.columns),observed_cells=int(source_power.notna().sum().sum()),missing_pct=100*(1-source_power.notna().sum().sum()/source_power.size),flagged_cells=int(flags.sum().sum()),**stats))
        sources.append(dict(file=a.file,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),apartment=apt,unit=settings['unit'],start=str(p.index.min()),end=str(p.index.max()),transformations=['parse timestamps','average duplicates','absolute electrical values','five-minute expected grid; sustained native 15-minute cadence uses explicit held interval support','editable spike review caps','adaptive thresholds and gap-reset hysteresis','apartment total is R+B+Y; all other electrical columns are submetered appliances/circuits','signed residual = total phases minus total appliances on complete shared timestamps']))
        print(f'{apt}: {len(p):,} expected slots; {flags.sum().sum():,} flagged cells',flush=True)
    pd.concat(maps,ignore_index=True).to_csv(root/'config/channels.csv',index=False)
    for name,frames in [('schedules',schedules),('standby',standby),('thresholds',thresholds)]:pd.concat(frames,ignore_index=True).to_csv(out/f'{name}.csv',index=False)
    pd.concat(events,ignore_index=True).to_parquet(out/'events.parquet',index=False);pd.DataFrame(qa).to_csv(out/'source_qa.csv',index=False)
    w=prepare_weather(root)
    for f in ['Palava weather data (1).csv','Lodha-Apartments-Sensors-mapping_modified(1).xlsx']:sources.append(dict(file=f,sha256=hashlib.sha256((root/'data/raw'/f).read_bytes()).hexdigest()))
    indoor=root/'data/raw/indoor.csv'
    if indoor.exists():
        d=pd.read_csv(indoor);d['Timestamp']=pd.to_datetime(d.Timestamp,errors='coerce');d=d.dropna(subset=['Timestamp'])
        for c,lo,hi in [('temperature',-10,60),('humidity',0,100)]:d[c]=pd.to_numeric(d[c],errors='coerce').where(lambda x:x.between(lo,hi))
        d=d.groupby(['Apartment','Room','Timestamp'],as_index=False)[['temperature','humidity']].mean();d.to_parquet(out/'indoor.parquet',index=False)
    elif (out/'indoor.parquet').exists():(out/'indoor.parquet').unlink()
    manifest=dict(build_status='complete',project_version='2.0.0',electrical_definitions={'total_phase_kw':'R phase + B phase + Y phase','total_appliance_kw':'sum of every non-phase electrical channel, including lights/fans, mixed and unnamed submetered circuits','residual_kw':'total_phase_kw - total_appliance_kw on timestamps with all inputs valid','positive_residual':'Possible unmetered or unmapped load; not proven standby or avoidable energy','negative_residual':'Retained signed for reconciliation review; not converted to positive load','A701_phase_alias':'Main Incomer is the R-phase channel in the supplied workbook'},generated_at=pd.Timestamp.now(tz='UTC').isoformat(),timezone=settings['timezone'],settings=settings,sources=sources,duplicate_file_excluded='Lakeside K 502(1).csv is an identical copy; excluded',indoor_available=indoor.exists(),weather_records=len(w),population='12 unique apartments; user-confirmed default cohort is latest 10',limitations=['States are inferred from electrical activity, not measured commands','No occupancy labels; activity score is not occupancy probability','Mixed AC/light circuit excluded from pure AC','Weather is mostly hourly and has gaps','Annualized candidate standby is a scenario, not measured annual energy'])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
if __name__=='__main__':run()
