"""Numerical definitions used by the pipeline, dashboard and evidence exports."""
import re
import numpy as np
import pandas as pd
APPLIANCES=['AC','Washing Machine','Geyser','Oven','Fridge']
LIGHTS='Lights & Fans (combined)'
DT_HOURS=5/60

def slug(s):return re.sub(r'[^a-z0-9]+','_',s.lower()).strip('_')

def classify_channel(name):
    s=name.split(' - ',1)[-1].lower()
    if re.fullmatch(r'[rby]\s+phase',s.strip()):return 'Phase'
    if re.search(r'\bwashing\s+machine\b',s):return 'Washing Machine'
    if re.search(r'\bac\b|\bair\s*condition',s):
        return 'Mixed circuit' if re.search(r'light|fan',s) else 'AC'
    if 'geyser' in s:return 'Geyser'
    if 'oven' in s:return 'Oven'
    if 'fridge' in s or 'refrigerator' in s:return 'Fridge'
    if re.search(r'light|fan',s):return LIGHTS
    return 'Other'

def season(index):
    return np.select([np.isin(index.month,[3,4,5]),np.isin(index.month,[6,7,8,9])],['Summer','Monsoon'],default='Mild')

def time_features(index):
    return pd.DataFrame({'season':season(index),'day_type':np.where(index.dayofweek<5,'Weekday','Weekend'),'slot':index.hour*4+index.minute//15},index=index)

def coverage(power,flags,start,end,scope):
    grid=pd.date_range(pd.Timestamp(start).floor('5min'),pd.Timestamp(end).floor('5min'),freq='5min')
    p=power.reindex(grid);f=flags.reindex(grid).eq(True)
    if not len(p.columns):return pd.DataFrame()
    a=pd.DataFrame({'expected_cells':len(p.columns),'observed_cells':p.notna().sum(axis=1),'flagged_cells':(f&p.notna()).sum(axis=1)},index=grid)
    if scope in ['monthly','daily']:a=a.resample('MS' if scope=='monthly' else 'D').sum().rename_axis('period').reset_index()
    elif scope=='hourly':a=a.groupby(a.index.hour).sum().rename_axis('hour').reset_index()
    else:a=a.sum().to_frame().T
    a['missing_cells']=a.expected_cells-a.observed_cells;a['usable_cells']=a.observed_cells-a.flagged_cells
    for name,c in [('missing_pct','missing_cells'),('flagged_pct','flagged_cells'),('usable_pct','usable_cells')]:a[name]=100*a[c]/a.expected_cells
    return a

def category_power(power,mapping):
    result={}
    for category,g in mapping.groupby('category',sort=False):
        cols=[c for c in g.channel.unique() if c in power]
        if cols:result[category]=power[cols].sum(axis=1,min_count=len(cols))
    return pd.DataFrame(result,index=power.index)

def phase_total(power,mapping):
    cols=[c for c in mapping.loc[mapping.category.eq('Phase'),'channel'].unique() if c in power]
    return power[cols].sum(axis=1,min_count=3).rename('total_kw') if len(cols)==3 else pd.Series(np.nan,index=power.index,name='total_kw')


def appliance_columns(power,mapping):
    """Every electrical column other than the three designated phase meters.
    Include submetered circuits whose appliance name or workbook mapping is unknown.
    """
    phase_cols=set(mapping.loc[mapping.category.eq('Phase'),'channel'])
    return [c for c in power.columns if c not in phase_cols]

def electrical_balance(power,mapping):
    """Signed phase-minus-appliance residual on a common complete-data mask."""
    phase=phase_total(power,mapping)
    cols=appliance_columns(power,mapping)
    appliances=power[cols].sum(axis=1,min_count=len(cols)) if cols else pd.Series(np.nan,index=power.index)
    valid=phase.notna()&appliances.notna()
    d=pd.DataFrame({'total_phase_kw':phase,'total_appliance_kw':appliances,
                    'valid_balance':valid,'phase_kw_aligned':phase.where(valid),
                    'appliance_kw_aligned':appliances.where(valid)},index=power.index)
    d['residual_kw']=d.phase_kw_aligned-d.appliance_kw_aligned
    # Absolute values apply to input consumption only, never the derived residual.
    d['positive_residual_kw']=d.residual_kw.clip(lower=0)
    d['negative_residual_kw']=d.residual_kw.clip(upper=0)
    d['negative_residual_flag']=d.residual_kw.lt(-1e-9)
    return d

def balance_summary(balance,scope='monthly'):
    """Paired energies share timestamps; sums reconcile without coverage bias."""
    d=pd.DataFrame({'phase_kwh_aligned':balance.phase_kw_aligned*DT_HOURS,
        'appliance_kwh_aligned':balance.appliance_kw_aligned*DT_HOURS,
        'residual_kwh':balance.residual_kw*DT_HOURS,
        'positive_residual_kwh':balance.positive_residual_kw*DT_HOURS,
        'negative_residual_kwh':balance.negative_residual_kw*DT_HOURS,
        'valid_balance_slots':balance.valid_balance.astype(int),
        'expected_slots':1,'negative_residual_slots':balance.negative_residual_flag.astype(int)},index=balance.index)
    energies=['phase_kwh_aligned','appliance_kwh_aligned','residual_kwh','positive_residual_kwh','negative_residual_kwh']
    if scope=='hourly':
        grouped=d.groupby(d.index.hour)
        out=grouped[energies].sum(min_count=1).join(grouped[['valid_balance_slots','expected_slots','negative_residual_slots']].sum()).rename_axis('hour').reset_index()
    else:
        rule={'monthly':'MS','daily':'D'}[scope]
        grouped=d.resample(rule)
        out=grouped[energies].sum(min_count=1).join(grouped[['valid_balance_slots','expected_slots','negative_residual_slots']].sum()).rename_axis('period').reset_index()
    out['balance_coverage_pct']=100*out.valid_balance_slots/out.expected_slots
    out['mean_residual_kw']=out.residual_kwh/(out.valid_balance_slots*DT_HOURS).replace(0,np.nan)
    out['residual_share_pct']=100*out.residual_kwh/out.phase_kwh_aligned.replace(0,np.nan)
    out['reconstruction_error_kwh']=out.phase_kwh_aligned-out.appliance_kwh_aligned-out.residual_kwh
    return out

def daily_total(power,mapping,minimum_coverage=.9):
    p=phase_total(power,mapping)
    d=pd.DataFrame({'recorded_kwh':(p*DT_HOURS).resample('D').sum(min_count=1),'valid_slots':p.resample('D').count()})
    d['expected_slots']=288;d['coverage_pct']=100*d.valid_slots/288;d['eligible']=d.coverage_pct>=minimum_coverage*100
    d['eligible_recorded_kwh']=d.recorded_kwh.where(d.eligible)
    return d.rename_axis('date').reset_index()

def cluster_threshold(values,category,noise_floor):
    v=values.dropna().to_numpy(float)
    if not len(v):return dict(on_kw=np.nan,off_kw=np.nan,design_kw=np.nan,method='No valid data',separation=np.nan)
    x=v[::max(1,len(v)//10000)];centers=np.quantile(x,[.15,.9])
    for _ in range(40):
        lab=np.abs(x[:,None]-centers).argmin(axis=1)
        nxt=np.array([x[lab==j].mean() if (lab==j).any() else centers[j] for j in range(2)])
        if np.allclose(nxt,centers,atol=1e-7):break
        centers=nxt
    low,high=np.sort(centers);sep=(high-low)/max(high,.001)
    if high<noise_floor or high-low<noise_floor/2:on=max(noise_floor,np.quantile(v,.99)*1.05);method='Low-load / unresolved active state'
    else:on=max(noise_floor,low+.5*(high-low));method='Measured two-cluster threshold'
    off=max(noise_floor*.5,min(on*.8,low+.3*(high-low)))
    active=v[v>=on]
    design=np.quantile(active,.995) if len(active) else np.quantile(v,.995)
    return dict(on_kw=float(on),off_kw=float(off),design_kw=float(max(design,on)),method=method,separation=float(sep),low_center_kw=float(low),high_center_kw=float(high))

def hysteresis(power,on,off,minimum_minutes=5):
    res=np.full(len(power),np.nan);state=0
    for i,x in enumerate(power.to_numpy(float)):
        if not np.isfinite(x):state=0;continue
        if x>=on:state=1
        elif x<=off:state=0
        res[i]=state
    mins=max(1,int(np.ceil(minimum_minutes/5)))
    if mins>1:
        edges=np.flatnonzero(np.diff(np.r_[False,res==1,False]))
        for a,b in zip(edges[::2],edges[1::2]):
            if b-a<mins:res[a:b]=0
    return pd.Series(res,index=power.index,name=power.name)

def event_table(power,state,apartment,category,channel):
    edges=np.flatnonzero(np.diff(np.r_[False,state.to_numpy()==1,False]));rows=[]
    for j,(s,e) in enumerate(zip(edges[::2],edges[1::2])):
        t=power.index[s]
        rows.append(dict(apartment=apartment,category=category,channel=channel,event_id=f'{slug(apartment)}_{slug(channel)}_{j+1}',start=t,end_exclusive=power.index[e-1]+pd.Timedelta(minutes=5),duration_minutes=(e-s)*5,energy_kwh=power.iloc[s:e].sum()*DT_HOURS,left_censored=s==0 or pd.isna(state.iloc[s-1]),right_censored=e==len(state) or pd.isna(state.iloc[e]),short_event=e-s==1,hour=t.hour,season=season(pd.DatetimeIndex([t]))[0],day_type='Weekday' if t.dayofweek<5 else 'Weekend'))
    return pd.DataFrame(rows,columns=['apartment','category','channel','event_id','start','end_exclusive','duration_minutes','energy_kwh','left_censored','right_censored','short_event','hour','season','day_type'])

def schedule_from_power(power,states,mapping,thresholds,apartment):
    rows=[]
    for category in APPLIANCES:
        cols=[c for c in mapping.loc[mapping.category.eq(category),'channel'].unique() if c in states]
        if not cols:continue
        valid=power[cols].notna().all(axis=1)&states[cols].notna().all(axis=1)
        p=power[cols].sum(axis=1,min_count=len(cols)).where(valid)
        st=states[cols].max(axis=1).where(valid)
        active=power[cols].where(states[cols].eq(1),0).sum(axis=1,min_count=len(cols)).where(valid)
        low=(p-active).where(valid)
        design=thresholds.set_index('channel').loc[cols,'design_kw'].sum(min_count=len(cols))
        counts=p.resample('15min').count()
        q=pd.DataFrame({'mean_kw':p.resample('15min').mean(),'active_kw':active.resample('15min').mean(),'inactive_kw':low.resample('15min').mean(),'active_probability':st.resample('15min').mean()})
        q=q.where(counts.eq(3),np.nan);q['fraction']=(q.active_kw/design).clip(0,1) if design>0 else np.nan
        q['clipping_loss_kw']=(q.active_kw-design*q.fraction).clip(lower=0)
        for allseason in [False,True]:
            z=q.join(time_features(q.index))
            if allseason:z['season']='All seasons'
            s=z.groupby(['season','day_type','slot']).agg(mean_kw=('mean_kw','mean'),active_kw=('active_kw','mean'),inactive_kw=('inactive_kw','mean'),fraction=('fraction','mean'),p25=('fraction',lambda x:x.quantile(.25)),median=('fraction','median'),p75=('fraction',lambda x:x.quantile(.75)),active_probability=('active_probability','mean'),valid_days=('fraction','count'),clipping_loss_kw=('clipping_loss_kw','mean')).reset_index()
            s['apartment']=apartment;s['category']=category;s['design_kw']=design;s['time']=s.slot.map(lambda x:f'{x//4:02}:{(x%4)*15:02}')
            s['reconstructed_kw']=design*s.fraction+s.inactive_kw+s.clipping_loss_kw
            s['reconstruction_error_pct']=np.where(s.mean_kw>0,100*(s.reconstructed_kw-s.mean_kw)/s.mean_kw,0)
            rows.append(s)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

def representative_schedule(schedules,cohort):
    s=schedules[schedules.apartment.isin(cohort)&schedules.valid_days.gt(0)].copy()
    if s.empty:return pd.DataFrame()
    keys=['category','season','day_type','slot','time']
    r=s.groupby(keys).agg(mean=('fraction','mean'),median=('fraction','median'),p25=('fraction',lambda x:x.quantile(.25)),p75=('fraction',lambda x:x.quantile(.75)),mean_kw=('mean_kw','mean'),active_probability=('active_probability','mean'),inactive_kw=('inactive_kw','mean'),active_kw=('active_kw','mean'),contributing_apartments=('apartment','nunique'),minimum_valid_days=('valid_days','min')).reset_index()
    r['selected_apartments']=len(cohort);r['complete_cohort']=r.contributing_apartments.eq(len(cohort))
    r['effective_design_kw']=np.where(r['mean']>0,r.active_kw/r['mean'],np.nan)
    r['energy_preserving_kw']=r.active_kw+r.inactive_kw
    return r

def standby_summary(power,states,mapping,apartment):
    rows=[]
    for _,m in mapping[mapping.category.isin(APPLIANCES)].iterrows():
        c=m.channel
        if c not in states:continue
        p=power[c];st=states[c];v=st.notna();low=v&st.eq(0)&p.gt(.001);eligible=m.category in ['AC','Washing Machine','Oven']
        lowenergy=p.where(low).sum()*DT_HOURS
        rows.append(dict(apartment=apartment,category=m.category,channel=c,active_kwh=p.where(st.eq(1)).sum()*DT_HOURS,inactive_low_kwh=lowenergy,off_noise_kwh=p.where(v&st.eq(0)&~low).sum()*DT_HOURS,candidate_standby_kwh=lowenergy if eligible else 0,inactive_median_w=p.where(low).median()*1000,inactive_hours=int(low.sum())*DT_HOURS,valid_hours=int(v.sum())*DT_HOURS,valid_days_equivalent=v.sum()/288,candidate_kwh_per_valid_day=lowenergy/(v.sum()/288) if eligible and v.sum() else 0,annualized_candidate_kwh=lowenergy/(v.sum()/288)*365 if eligible and v.sum() else 0,status='Candidate; field verification required' if eligible else 'Operational cycling / heating; excluded from vampire estimate'))
    return pd.DataFrame(rows)

def psychrometrics(temp,rh):
    t=temp.astype(float);r=rh.astype(float).where(rh.between(.01,100));gamma=np.log(r/100)+17.625*t/(243.04+t)
    dp=243.04*gamma/(17.625-gamma);vapor=(r/100)*.61094*np.exp(17.625*t/(t+243.04));w=.62198*vapor/(101.325-vapor)
    return pd.DataFrame({'dew_point_c':dp,'humidity_ratio_kg_kg':w,'enthalpy_kj_kg':1.006*t+w*(2501+1.86*t)},index=temp.index)

def energy_table(power,mapping,frequency='D'):
    rows=[]
    for category,g in mapping[~mapping.category.eq('Phase')].groupby('category'):
        cols=g.channel.unique().tolist()
        e=(power[cols]*DT_HOURS).sum(axis=1,min_count=1).resample(frequency).sum(min_count=1)
        cells=power[cols].notna().sum(axis=1).resample(frequency).sum();expected=pd.Series(len(cols),index=power.index).resample(frequency).sum()
        z=pd.DataFrame({'recorded_kwh':e,'valid_cells':cells,'expected_cells':expected});z['coverage_pct']=100*z.valid_cells/z.expected_cells;z['category']=category
        rows.append(z.rename_axis('period').reset_index())
    return pd.concat(rows,ignore_index=True)
