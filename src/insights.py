"""Filter-scoped descriptive analytics and explicitly conditional statistical models."""
import re
import numpy as np
import pandas as pd
from scipy.stats import t as student_t
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from .analytics import phase_total, category_power, electrical_balance, season, psychrometrics

WEATHER_FIELDS=['temperature','humidity','irradiance_w_m2','wind_speed_m_s','wind_direction_deg','rainfall_mm','pressure_hpa']

def hourly_power(s, minimum=.9):
    """An hourly mean is known only if the selected share of 12 input slots exists."""
    return s.resample('h').mean().where(s.resample('h').count()>=np.ceil(12*minimum))

def paired_weather(index,weather):
    """Backward as-of join: never use a future reading or bridge >60 minutes."""
    left=pd.DataFrame({'Timestamp':index})
    right=weather.rename_axis('weather_at').reset_index().sort_values('weather_at')
    out=pd.merge_asof(left,right,left_on='Timestamp',right_on='weather_at',direction='backward',tolerance=pd.Timedelta(minutes=60)).set_index('Timestamp')
    out['weather_age_minutes']=(out.index-out.weather_at).dt.total_seconds()/60
    return out

def hourly_dataset(p,m,weather,minimum=.9):
    cat=category_power(p,m)
    out=pd.DataFrame({'total_kw':hourly_power(phase_total(p,m),minimum)})
    out['ac_kw']=hourly_power(cat.get('AC',pd.Series(np.nan,index=p.index)),minimum)
    out=out.join(paired_weather(out.index,weather))
    out['weekend']=(out.index.dayofweek>=5).astype(int)
    out['hour']=out.index.hour
    out['season']=season(out.index)
    return out

def fingerprint(total,minimum=.9):
    daily=total.resample('D').count()/288
    allowed=daily[daily>=minimum].index
    h=hourly_power(total,minimum).to_frame('kw')
    h=h[h.index.normalize().isin(allowed)].dropna()
    h['hour']=h.index.hour;h['day_type']=np.where(h.index.dayofweek<5,'Weekday','Weekend');h['season']=season(h.index)
    def group(keys):return h.groupby(keys,as_index=False).agg(mean_kw=('kw','mean'),median_kw=('kw','median'),p25_kw=('kw',lambda x:x.quantile(.25)),p75_kw=('kw',lambda x:x.quantile(.75)),valid_hours=('kw','count'))
    return group(['hour']),group(['day_type','hour']),group(['season','hour']),len(allowed)

def load_shape(total,minimum=.9,min_days=14):
    h=hourly_power(total,minimum).to_frame('kw');h['date']=h.index.normalize();h['hour']=h.index.hour
    grid=h.pivot(index='date',columns='hour',values='kw').reindex(columns=range(24))
    # Complete 24-hour shapes avoid assigning a profile to unobserved hours.
    grid=grid.dropna();grid=grid[grid.mean(axis=1)>0]
    if len(grid)<min_days:return None,len(grid)
    return grid.div(grid.mean(axis=1),axis=0).mean().to_numpy(),len(grid)

def appliance_shares(p,m):
    bal=electrical_balance(p,m);valid=bal.valid_balance
    cols=[c for c in p if c not in set(m.loc[m.category.eq('Phase'),'channel'])]
    denominator=bal.appliance_kw_aligned.sum(min_count=1)
    if valid.sum()<288 or not pd.notna(denominator) or denominator<=0:return {}
    cat=category_power(p.loc[valid],m)
    return {c:cat[c].sum()/denominator for c in cat if c!='Phase'}

def cluster_apartments(profiles,shares=None):
    """K-means on mean-one shapes; optional equal-block-weight appliance shares."""
    names=sorted(profiles)
    if len(names)<4:return {'ok':False,'reason':'At least four apartments need 14 complete daily profiles in the selected dates.'}
    x=np.array([profiles[n] for n in names]);features=x/np.sqrt(24)
    if shares is not None:
        cats=sorted({c for n in names for c in shares.get(n,{})})
        names=[n for n in names if shares.get(n)]
        if len(names)<4:return {'ok':False,'reason':'At least four apartments also need complete appliance-balance readings for shares.'}
        x=np.array([profiles[n] for n in names]);s=np.array([[shares[n].get(c,0) for c in cats] for n in names])
        features=np.c_[x/np.sqrt(24),s]
    candidates=[]
    for k in range(2,min(4,len(names)-1)+1):
        model=KMeans(n_clusters=k,random_state=42,n_init=20).fit(features)
        labels=model.labels_
        if len(set(labels))!=k or np.bincount(labels).min()<2:continue
        score=silhouette_score(features,labels)
        candidates.append((score,k,labels,model.cluster_centers_))
    if not candidates:return {'ok':False,'reason':'No non-singleton grouping is supported. More apartments or a longer shared period are needed.'}
    score,k,labels,centers=max(candidates,key=lambda z:z[0])
    # Number clusters by centroid peak hour for deterministic readable labels.
    order=sorted(range(k),key=lambda i:int(np.argmax(x[labels==i].mean(axis=0))))
    renumber={old:new+1 for new,old in enumerate(order)}
    assignment=pd.DataFrame({'apartment':names,'cluster':[renumber[v] for v in labels]})
    rows=[]
    for i,n in enumerate(names):
        for hour,val in enumerate(x[i]):rows.append({'apartment':n,'cluster':renumber[labels[i]],'hour':hour,'normalized_kw':val})
    return {'ok':True,'assignments':assignment,'profiles':pd.DataFrame(rows),'k':k,'silhouette':float(score)}

def calendar_design(index):
    hour=index.hour.to_numpy()
    return np.c_[np.ones(len(index)),np.sin(2*np.pi*hour/24),np.cos(2*np.pi*hour/24),np.sin(4*np.pi*hour/24),np.cos(4*np.pi*hour/24),(index.dayofweek>=5).astype(int)]

def regression(x,y,groups=None):
    b=np.linalg.lstsq(x,y,rcond=None)[0];res=y-x@b
    cov=np.linalg.pinv(x.T@x)
    if groups is not None:
        unique=np.unique(groups);meat=np.zeros((x.shape[1],x.shape[1]))
        for g in unique:
            z=x[groups==g].T@res[groups==g];meat+=np.outer(z,z)
        correction=len(unique)/max(len(unique)-1,1)*(len(y)-1)/max(len(y)-x.shape[1],1)
        cov=cov@meat@cov*correction;critical=student_t.ppf(.975,max(len(unique)-1,1))
    else:cov=cov*(res@res/max(len(y)-x.shape[1],1));critical=1.96
    ci=critical*np.sqrt(np.maximum(np.diag(cov),0))
    return b,ci

def r2(y,p):
    den=np.sum((y-np.mean(y))**2)
    return 1-np.sum((y-p)**2)/den if den>1e-12 else np.nan

def weather_model(data,target='total_kw',moisture='humidity',min_hours=240,min_days=30):
    """Cooling hinge + moisture + calendar; nested chronological selection/holdout.
    Diagnostic model only; conditional coefficient intervals cluster residuals by day.
    """
    d=data.dropna(subset=[target,'temperature',moisture]).copy().sort_index()
    if len(d)<min_hours or d.index.normalize().nunique()<min_days:
        return {'ok':False,'reason':f'Need {min_days} paired days and {min_hours} valid hours; found {d.index.normalize().nunique()} days / {len(d)} hours.'}
    if d[moisture].std()<1e-6:
        return {'ok':False,'reason':'The selected moisture variable has no variation; its response cannot be separated in this weather model.'}
    if d[target].std()<1e-5 or d.temperature.max()-d.temperature.min()<4:
        return {'ok':False,'reason':'Insufficient consumption or outdoor-temperature variation to identify a weather response.'}
    dates=sorted(d.index.normalize().unique());cut=dates[int(len(dates)*.8)];train=d[d.index<cut];test=d[d.index>=cut]
    if len(test)<48:return {'ok':False,'reason':'Fewer than 48 later observations remain for validation.'}
    # Optional measured weather fields; never fill missing future station fields.
    extra=[c for c in ['irradiance_w_m2','wind_speed_m_s','rainfall_mm','pressure_hpa'] if c in d and d[c].notna().all() and d[c].std()>1e-6]
    ref={c:float(train[c].median()) for c in ['temperature',moisture]+extra}
    scale={c:float(train[c].std()) for c in [moisture]+extra}
    def design(frame,base,reference=False):
        v=frame.copy()
        if reference:
            for c in ref:v[c]=ref[c]
        return np.c_[calendar_design(v.index),np.maximum(v.temperature.to_numpy()-base,0),*[(v[c].to_numpy()-ref[c])/scale[c] for c in [moisture]+extra]]
    tr_dates=sorted(train.index.normalize().unique());inner=tr_dates[int(len(tr_dates)*.75)]
    best=None;curve=[]
    for base in np.arange(18,33,.5):
        x=design(train,base);y=train[target].to_numpy();mask=train.index<inner
        above=(train.temperature>base).sum();below=(train.temperature<=base).sum()
        if min(above,below)<24 or np.linalg.matrix_rank(x[mask])<x.shape[1]:continue
        b=np.linalg.lstsq(x[mask],y[mask],rcond=None)[0]
        if b[6]<0:continue
        error=float(np.mean((y[~mask]-x[~mask]@b)**2));curve.append({'base_c':base,'validation_mse':error})
        if best is None or error<best[0]:best=(error,base)
    if best is None:return {'ok':False,'reason':'No supported cooling change point has observations on both sides and a non-negative cooling response.'}
    base=best[1];x=design(train,base);y=train[target].to_numpy();b,ci=regression(x,y,train.index.normalize().to_numpy())
    pred_test=design(test,base)@b
    baseline=np.linalg.lstsq(calendar_design(train.index),y,rcond=None)[0]
    baseline_test=calendar_design(test.index)@baseline
    mae=float(np.mean(np.abs(test[target]-pred_test)));baseline_mae=float(np.mean(np.abs(test[target]-baseline_test)))
    out=d.copy();out['predicted_kw']=design(d,base)@b;out['reference_weather_kw']=design(d,base,True)@b
    out['weather_adjusted_kw']=out[target]-out.predicted_kw+out.reference_weather_kw
    out['validation_set']=np.where(out.index<cut,'Training','Later holdout')
    quality=r2(test[target].to_numpy(),pred_test)
    supported=bool(np.isfinite(quality) and quality>0 and mae<baseline_mae and b[6]-ci[6]>0 and (pred_test<0).mean()<.05)
    return dict(ok=True,rows=out,base_c=base,slope=float(b[6]),slope_low=float(b[6]-ci[6]),slope_high=float(b[6]+ci[6]),holdout_r2=quality,mae_kw=mae,calendar_mae_kw=baseline_mae,supported=supported,train_hours=len(train),test_hours=len(test),paired_days=len(dates),features=['Cooling degree term',moisture]+extra,change_point_at_search_boundary=bool(base in [18,32.5]),reference_weather=ref,change_point_curve=pd.DataFrame(curve),coefficients=pd.DataFrame({'term':['Intercept','Hour sin','Hour cos','Hour 2sin','Hour 2cos','Weekend','Cooling slope']+[moisture]+extra,'coefficient':b,'low_95':b-ci,'high_95':b+ci}),cutoff=str(cut))

def daily_cooling(rows,base):
    q=rows[['ac_kw','temperature','humidity','enthalpy_kj_kg']].dropna().copy()
    q['cdh']=np.maximum(q.temperature-base,0)
    d=q.resample('D').agg(ac_kwh=('ac_kw',lambda s:s.sum(min_count=1)),cdh=('cdh',lambda s:s.sum(min_count=1)),temperature=('temperature','mean'),humidity=('humidity','mean'),enthalpy=('enthalpy_kj_kg','mean'),paired_hours=('ac_kw','count'))
    # Use full 24-hour days for degree-hour comparisons; no missing-hour inflation.
    return d[d.paired_hours.eq(24)].rename_axis('date').reset_index()

def bhk(value):
    hit=re.search(r'(\d+)\s*BHK',str(value),re.I)
    return int(hit[1]) if hit else None

def matched_benchmark(selected,metadata,hourly,area_tolerance=.15,floor_tolerance=4,min_hours=168):
    a=metadata[metadata.apartment.eq(selected)].iloc[0];records=[];names=[]
    for _,b in metadata[~metadata.apartment.eq(selected)].iterrows():
        reasons=[]
        if pd.isna(a.area_m2) or pd.isna(b.area_m2):reasons.append('Area unavailable')
        elif abs(b.area_m2/a.area_m2-1)>area_tolerance:reasons.append('Outside area tolerance')
        if bhk(a.flat_type) is None or bhk(b.flat_type) is None or bhk(a.flat_type)!=bhk(b.flat_type):reasons.append('BHK mismatch/unavailable')
        if pd.isna(a.orientation) or pd.isna(b.orientation) or a.orientation!=b.orientation:reasons.append('Orientation mismatch/unavailable')
        if pd.isna(a.floor) or pd.isna(b.floor) or abs(a.floor-b.floor)>floor_tolerance:reasons.append('Floor mismatch/unavailable')
        records.append({'apartment':b.apartment,'area_m2':b.area_m2,'BHK':bhk(b.flat_type),'orientation':b.orientation,'floor':b.floor,'match':not reasons,'reason':'; '.join(reasons) or 'Matched'})
        if not reasons:names.append(b.apartment)
    audit=pd.DataFrame(records)
    if len(names)<2:return dict(ok=False,reason=f'Only {len(names)} metadata-matched peer(s); two are required.',audit=audit)
    target=hourly[selected];peers=[]
    for n in names:
        area=metadata.loc[metadata.apartment.eq(n),'area_m2'].iloc[0]
        peers.append(hourly[n].total_kw.rename(n)*a.area_m2/area)
    panel=pd.concat(peers,axis=1);out=target[['total_kw','temperature','humidity']].copy();out['peer_count']=panel.notna().sum(axis=1)
    out['expected_kw']=panel.median(axis=1).where(out.peer_count>=2)
    out=out.dropna(subset=['total_kw','expected_kw','temperature','humidity'])
    if len(out)<min_hours:return dict(ok=False,reason=f'Only {len(out)} contemporaneous matched hours; {min_hours} are required.',audit=audit)
    out['actual_expected']=out.total_kw/out.expected_kw.replace(0,np.nan)
    expected=out.expected_kw.sum()
    return dict(ok=expected>0,reason='Zero expected demand' if expected<=0 else '',audit=audit,rows=out,ratio=float(out.total_kw.sum()/expected) if expected>0 else np.nan,peers=names,hours=len(out))

def flexibility_events(events,peak=(18,22),target=(22,24),max_delay=6,include_ac=False):
    d=events[events.category.isin(['Washing Machine','Geyser']+(['AC'] if include_ac else []))].copy()
    d=d[~d.left_censored&~d.right_censored&d.energy_kwh.gt(0)].copy()
    if d.empty:return d
    d['in_peak']=d.start.dt.hour.ge(peak[0])&d.start.dt.hour.lt(peak[1])
    proposals=[];eligible=[];delays=[]
    for row in d.itertuples():
        window=row.start.normalize()+pd.Timedelta(hours=target[0]);stop=row.start.normalize()+pd.Timedelta(hours=target[1])
        if row.start>=stop:window+=pd.Timedelta(days=1);stop+=pd.Timedelta(days=1)
        proposed=max(row.start,window);delay=(proposed-row.start).total_seconds()/3600
        ok=row.in_peak and 0<delay<=max_delay and proposed+pd.Timedelta(minutes=row.duration_minutes)<=stop
        # AC requires separate comfort authorization; screen at most a 15-minute deferral.
        if row.category=='AC':ok=ok and delay<=.25
        proposals.append(proposed if ok else pd.NaT);eligible.append(ok);delays.append(delay if ok else np.nan)
    d['candidate_start']=proposals;d['candidate']=eligible;d['delay_hours']=delays
    return d
