"""Room-specific indoor analysis; no simulated sensor values or gap interpolation."""
import numpy as np
import pandas as pd
from .analytics import psychrometrics
from .insights import paired_weather, regression, r2

REQUIRED=['Timestamp','Apartment','Room','temperature','humidity']

def validate_indoor(data):
    missing=set(REQUIRED)-set(data)
    if missing:raise ValueError('Missing indoor columns: '+', '.join(sorted(missing)))
    d=data[REQUIRED].copy();d['Timestamp']=pd.to_datetime(d.Timestamp,format='mixed',errors='coerce')
    if d.Timestamp.dt.tz is not None:d['Timestamp']=d.Timestamp.dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)
    for c,lo,hi in [('temperature',-10,60),('humidity',0,100)]:
        d[c]=pd.to_numeric(d[c],errors='coerce');d[c]=d[c].where(d[c].between(lo,hi))
    d=d.dropna(subset=['Timestamp','Apartment','Room']);d['Apartment']=d.Apartment.astype(str).str.strip();d['Room']=d.Room.astype(str).str.strip()
    return d.groupby(['Timestamp','Apartment','Room'],as_index=False)[['temperature','humidity']].mean()

def align_indoor(indoor,weather,power,start,end):
    idx=pd.date_range(start,end,freq='15min')
    d=indoor.set_index('Timestamp')[['temperature','humidity']].resample('15min').mean().reindex(idx).rename(columns={'temperature':'tin','humidity':'rhin'})
    d['ac_kw']=power.resample('15min').mean().where(power.resample('15min').count().eq(3)).reindex(idx)
    w=paired_weather(idx,weather).rename(columns={'temperature':'tout','humidity':'rhout','dew_point_c':'dew_out','humidity_ratio_kg_kg':'wout','enthalpy_kj_kg':'enthalpy_out'})
    d=d.join(w)
    ps=psychrometrics(d.tin,d.rhin)
    d['dew_in']=ps.dew_point_c;d['win']=ps.humidity_ratio_kg_kg*1000;d['wout']=d.wout*1000
    d['delta_t']=d.tout-d.tin;d['d_tin']=d.tin.shift(-1)-d.tin
    d['d_win']=d.win.shift(-1)-d.win
    return d

def buffering(d,off_kw=.03):
    off=d.ac_kw.le(off_kw)&d.ac_kw.shift(-1).le(off_kw)
    delta=d[['tin','tout','delta_t']].dropna()
    lagrows=[]
    for slots in range(0,25):
        # Differences attenuate the shared daily cycle. Both endpoints and the
        # entire lag window must be observed AC-off intervals.
        mask=off.rolling(slots+2).sum().eq(slots+2)
        q=pd.DataFrame({'inside':d.tin.diff(),'outside':d.tout.diff().shift(slots)}).where(mask).dropna()
        corr=q.inside.corr(q.outside) if len(q)>=48 and q.inside.std()>1e-6 and q.outside.std()>1e-6 else np.nan
        lagrows.append({'lag_hours':slots*.25,'correlation':corr,'paired_transitions':len(q)})
    rates=[]
    for name,mask,night in [('Daytime rise',d.index.hour.to_series(index=d.index).between(10,16),False),('Overnight change',(d.index.hour>=22)|(d.index.hour<6),True)]:
        q=d.loc[off&mask,['tin']].dropna();labels=(q.index-pd.Timedelta(hours=6)).normalize() if night else q.index.normalize()
        for date,g in q.groupby(labels):
            if len(g)<12 or (g.index.max()-g.index.min()).total_seconds()<3*3600:continue
            # Exclude records separated by a long gap/AC interval.
            if g.index.to_series().diff().max()>pd.Timedelta(minutes=15):continue
            x=(g.index-g.index[0]).total_seconds()/3600
            rates.append({'date':date,'period':name,'rate_c_per_hour':np.polyfit(x,g.tin,1)[0],'points':len(g)})
    return delta,pd.DataFrame(lagrows),pd.DataFrame(rates,columns=['date','period','rate_c_per_hour','points'])

def thermal_model(d):
    q=d[['tin','tout','ac_kw','d_tin']].dropna().copy()
    if len(q)<200 or q.index.normalize().nunique()<7 or q.ac_kw.le(.03).sum()<120:
        return dict(ok=False,reason='Need 200 paired 15-minute transitions across seven days, including 30 AC-off hours.')
    gap=q.tout-q.tin
    if gap.std()<.5 or q.ac_kw.std()<.05:return dict(ok=False,reason='Insufficient outdoor–indoor difference or AC variation to identify a time constant.')
    x=np.c_[np.ones(len(q)),gap,q.ac_kw];split=int(len(q)*.8)
    if np.linalg.matrix_rank(x[:split])<3:return dict(ok=False,reason='Thermal predictors are not independently identifiable.')
    b,ci=regression(x[:split],q.d_tin.to_numpy()[:split],q.index.normalize().to_numpy()[:split])
    pred=x[split:]@b;beta=b[1];low=beta-ci[1];high=beta+ci[1];valid=0<low<high<1 and b[2]+ci[2]<0
    quality=r2(q.d_tin.to_numpy()[split:],pred)
    valid=bool(valid and np.isfinite(quality) and quality>0)
    tau=lambda b:-.25/np.log(1-b)
    q['predicted_d_tin']=x@b;q['set']=np.where(np.arange(len(q))<split,'Training','Later holdout')
    return dict(ok=True,supported=valid,tau_hours=float(tau(beta)) if 0<beta<1 else np.nan,tau_low=float(tau(high)) if 0<high<1 else np.nan,tau_high=float(tau(low)) if 0<low<1 else np.nan,beta=float(beta),ac_coefficient=float(b[2]),holdout_r2=quality,rows=q,coefficients=pd.DataFrame({'term':['Intercept','Tout - Tin','AC kW'],'coefficient':b,'low_95':b-ci,'high_95':b+ci}))

def ac_event_response(d,events,channel):
    rows=[];traces=[]
    ev=events[events.channel.eq(channel)&~events.left_censored&~events.right_censored&events.duration_minutes.between(30,360)]
    for e in ev.itertuples():
        t=e.start.ceil('15min');stop=e.end_exclusive.floor('15min')
        if stop<=t:continue
        before=d.loc[t-pd.Timedelta(minutes=30):t-pd.Timedelta(minutes=15)]
        during=d.loc[t:stop-pd.Timedelta(minutes=15)]
        after=d.loc[stop:stop+pd.Timedelta(minutes=15)]
        expected=int((stop-t).total_seconds()/900)
        if len(before)!=2 or len(after)!=2 or len(during)!=expected:continue
        allq=pd.concat([before,during,after])
        if allq[['tin','tout','rhin','rhout','ac_kw']].isna().any().any():continue
        # Require AC-off context around an isolated detected operation.
        if before.ac_kw.gt(.03).any() or after.ac_kw.gt(.03).any():continue
        energy=during.ac_kw.sum()*.25
        if energy<=0:continue
        first=before.mean(numeric_only=True);last=after.mean(numeric_only=True)
        rows.append(dict(event_id=e.event_id,start=e.start,energy_kwh=energy,duration_minutes=e.duration_minutes,tin_before=first.tin,tin_during=during.tin.mean(),tin_after=last.tin,temperature_drop_c=first.tin-last.tin,raw_drop_c_per_kwh=(first.tin-last.tin)/energy,outdoor_change_c=last.tout-first.tout,initial_delta_t=first.tout-first.tin,outdoor_rh=allq.rhout.mean(),rh_change=last.rhin-first.rhin,dew_change_c=last.dew_in-first.dew_in,humidity_ratio_change_g_kg=last.win-first.win))
        for time,v in allq.iterrows():traces.append({'event_id':e.event_id,'relative_minutes':(time-t).total_seconds()/60,'tin':v.tin,'tout':v.tout,'rh':v.rhin,'dew_point_c':v.dew_in,'humidity_ratio_g_kg':v.win,'ac_kw':v.ac_kw})
    return pd.DataFrame(rows),pd.DataFrame(traces)

def effectiveness_model(events):
    q=events.dropna() if not events.empty else events
    if len(q)<20:return dict(ok=False,reason=f'Need 20 complete isolated AC events; found {len(q)}.')
    x=np.c_[np.ones(len(q)),q.energy_kwh,q.outdoor_change_c,q.initial_delta_t,q.outdoor_rh,q.duration_minutes/60]
    if len(q)<3*x.shape[1] or np.linalg.matrix_rank(x)<x.shape[1]:return dict(ok=False,reason='Insufficient independent event variation for weather-adjusted effectiveness.')
    b,ci=regression(x,q.temperature_drop_c.to_numpy(),q.start.dt.normalize().to_numpy())
    return dict(ok=True,effect_c_per_kwh=float(b[1]),low=float(b[1]-ci[1]),high=float(b[1]+ci[1]),coefficients=pd.DataFrame({'term':['Intercept','AC kWh','Outdoor temperature change','Initial Tout-Tin','Outdoor RH','Duration hours'],'coefficient':b,'low_95':b-ci,'high_95':b+ci}))
