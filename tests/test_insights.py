import numpy as np
import pandas as pd
from src.insights import *
from src.indoor_insights import *

def test_backward_weather_never_uses_future_or_bridges_gaps():
    idx=pd.to_datetime(['2025-06-01 12:00','2025-06-01 13:00','2025-06-01 14:01'])
    w=pd.DataFrame({'temperature':[30.]},index=pd.to_datetime(['2025-06-01 12:30']))
    d=paired_weather(idx,w)
    assert pd.isna(d.iloc[0].temperature) and d.iloc[1].temperature==30 and pd.isna(d.iloc[2].temperature)

def test_hourly_missing_not_treated_as_zero():
    s=pd.Series([1.]*11+[np.nan],index=pd.date_range('2025-06-01',periods=12,freq='5min'))
    assert hourly_power(s,.9).iloc[0]==1
    assert pd.isna(hourly_power(s,1).iloc[0])

def test_shape_is_invariant_to_size():
    idx=pd.date_range('2025-06-01',periods=14*288,freq='5min');s=pd.Series(2+np.sin(idx.hour/24*2*np.pi),index=idx)
    a,n=load_shape(s);b,_=load_shape(s*3)
    assert n==14 and np.allclose(a,b)

def test_clustering_no_singletons_and_repeatable():
    x=np.arange(24);profiles={f'A{i}':1+(.4 if i<3 else -.4)*np.cos(x/24*2*np.pi)+i*.0001 for i in range(6)}
    a=cluster_apartments(profiles);b=cluster_apartments(profiles)
    assert a['ok'] and a['assignments'].equals(b['assignments'])
    assert a['assignments'].groupby('cluster').size().min()>=2

def test_weather_hinge_recovers_known_positive_slope():
    rng=np.random.default_rng(4);idx=pd.date_range('2025-05-01',periods=90*24,freq='h')
    temp=26+6*np.sin(np.arange(len(idx))/60)+rng.normal(0,.3,len(idx));rh=60+rng.normal(0,5,len(idx))
    y=.3+.25*np.maximum(temp-25,0)+.002*(rh-60)+.03*np.sin(idx.hour/24*2*np.pi)+rng.normal(0,.01,len(idx))
    d=pd.DataFrame({'temperature':temp,'humidity':rh,'total_kw':y},index=idx)
    m=weather_model(d)
    assert m['ok'] and abs(m['base_c']-25)<=.5 and abs(m['slope']-.25)<.02 and m['supported']
    assert m['rows'].loc[m['rows'].validation_set.eq('Later holdout')].index.min()>=pd.Timestamp(m['cutoff'])

def test_weather_short_scope_has_no_model():
    idx=pd.date_range('2025-05-01',periods=24,freq='h');d=pd.DataFrame({'temperature':30,'humidity':70,'total_kw':1},index=idx)
    assert not weather_model(d)['ok']

def test_benchmark_rejects_missing_metadata_and_matches_common_hours():
    meta=pd.DataFrame({'apartment':['A','B','C','D'],'area_m2':[70]*4,'floor':[5,6,7,np.nan],'flat_type':['2 BHK']*4,'orientation':['SOUTH']*4})
    idx=pd.date_range('2025-05-01',periods=200,freq='h')
    hourly={a:pd.DataFrame({'total_kw':v,'temperature':30,'humidity':60},index=idx) for a,v in zip(meta.apartment,[2,1,1,10])}
    r=matched_benchmark('A',meta,hourly)
    assert r['ok'] and r['ratio']==2 and r['peers']==['B','C'] and r['hours']==200
    assert not r['audit'].loc[r['audit'].apartment.eq('D'),'match'].iloc[0]

def test_flexible_events_respect_window_and_energy_denominator():
    e=pd.DataFrame({'category':['Washing Machine','Geyser'],'start':pd.to_datetime(['2025-06-01 19:00','2025-06-01 04:00']),'duration_minutes':[60,10],'energy_kwh':[2,1],'left_censored':False,'right_censored':False})
    q=flexibility_events(e)
    assert q.candidate.tolist()==[True,False]
    assert q.candidate_start.iloc[0]==pd.Timestamp('2025-06-01 22:00') and q.delay_hours.iloc[0]==3

def test_indoor_import_preserves_negative_temperature_and_invalid_rh():
    d=pd.DataFrame({'Timestamp':['2025-06-01'],'Apartment':['A'],'Room':['Bed'],'temperature':[-2],'humidity':[150]})
    q=validate_indoor(d);assert q.temperature.iloc[0]==-2 and pd.isna(q.humidity.iloc[0])

def test_thermal_response_recovers_synthetic_time_constant():
    rng=np.random.default_rng(1);n=1600;idx=pd.date_range('2025-06-01',periods=n,freq='15min');outside=30+4*np.sin(np.arange(n)/96*2*np.pi);ac=rng.integers(0,2,n).astype(float);tin=np.zeros(n);tin[0]=28
    beta=1-np.exp(-.25/5)
    for i in range(n-1):tin[i+1]=tin[i]+beta*(outside[i]-tin[i])-.2*ac[i]+rng.normal(0,.002)
    d=pd.DataFrame({'tin':tin,'tout':outside,'ac_kw':ac,'d_tin':pd.Series(tin).shift(-1).to_numpy()-tin},index=idx)
    m=thermal_model(d);assert m['ok'] and m['supported'] and abs(m['tau_hours']-5)<.2

def test_indoor_alignment_does_not_fill_missing_or_cross_rooms():
    idx=pd.date_range('2025-06-01',periods=12,freq='5min');p=pd.Series(1.,index=idx)
    w=pd.DataFrame({'temperature':[32.],'humidity':[65.]},index=pd.to_datetime(['2025-06-01']))
    w=w.join(psychrometrics(w.temperature,w.humidity))
    indoor=pd.DataFrame({'Timestamp':pd.to_datetime(['2025-06-01 00:00','2025-06-01 00:30']),'temperature':[28.,27.],'humidity':[70.,69.]})
    d=align_indoor(indoor,w,p,idx[0],idx[-1]);assert pd.isna(d.iloc[1].tin) and pd.isna(d.iloc[0].d_tin)

def test_cdh_keeps_only_complete_days():
    idx=pd.date_range('2025-06-01',periods=47,freq='h');d=pd.DataFrame({'ac_kw':1,'temperature':30,'humidity':60,'enthalpy_kj_kg':75},index=idx)
    q=daily_cooling(d,24);assert len(q)==1 and q.ac_kwh.iloc[0]==24 and q.cdh.iloc[0]==144
