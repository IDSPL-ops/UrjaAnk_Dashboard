import numpy as np
import pandas as pd
from src.analytics import *
from src.preprocess import read_apartment,channel_mapping

def test_classification_has_word_boundaries_and_mixed_exclusion():
    assert classify_channel('X - Washing Machine')=='Washing Machine'
    assert classify_channel('X - Master Bedroom AC')=='AC'
    assert classify_channel('X - Bed1 AC and Washroom Lights')=='Mixed circuit'
    assert classify_channel('X - Kitchen Lights and Fans')==LIGHTS

def test_absolute_before_duplicate_integration(tmp_path):
    f=tmp_path/'x.csv'
    pd.DataFrame({'Timestamp':['2025-05-03 00:00','2025-05-03 00:00','2025-05-03 00:10'],'Apartment':['X']*3,'X - Oven':[-2,2,-1]}).to_csv(f,index=False)
    p,q=read_apartment(f)
    assert p.iloc[0,0]==2
    assert np.isclose(p.sum().iloc[0]*DT_HOURS,.25)
    assert pd.isna(p.iloc[1,0]) and q['duplicate_timestamps']==1

def test_coverage_expected_cells_and_zero_is_observed():
    idx=pd.date_range('2025-05-03',periods=3,freq='5min')
    p=pd.DataFrame({'a':[0,np.nan,1],'b':[2,0,0]},index=idx)
    f=pd.DataFrame(False,index=idx,columns=p.columns);f.iloc[2,0]=True
    q=coverage(p,f,idx[0],idx[-1],'all').iloc[0]
    assert q.expected_cells==6 and q.observed_cells==5 and q.flagged_cells==1
    assert np.isclose(q.missing_pct,100/6)
    assert np.isclose(q.usable_pct,400/6)

def test_missing_resets_hysteresis_and_censors_events():
    idx=pd.date_range('2025-05-03',periods=6,freq='5min');p=pd.Series([0,2,1,np.nan,1,2],index=idx,name='AC')
    s=hysteresis(p,1.5,.5)
    assert s.iloc[:3].tolist()==[0,1,1]
    assert pd.isna(s.iloc[3]) and s.iloc[4]==0 and s.iloc[5]==1
    e=event_table(p,s,'X','AC','X - AC')
    assert len(e)==2 and e.duration_minutes.tolist()==[10,5]
    assert bool(e.iloc[0].right_censored)

def test_phase_requires_all_three_readings_and_no_double_count():
    p=pd.DataFrame({'R':[1,np.nan],'Y':[2,2],'B':[3,3],'AC':[3,3]})
    m=pd.DataFrame({'channel':['R','Y','B','AC'],'category':['Phase']*3+['AC']})
    t=phase_total(p,m)
    assert t.iloc[0]==6 and pd.isna(t.iloc[1])

def test_individual_schedule_reconstructs_and_keeps_missing():
    idx=pd.date_range('2025-05-03',periods=288,freq='5min');p=pd.DataFrame({'X - Oven':1.0},index=idx);s=pd.DataFrame({'X - Oven':1.0},index=idx)
    m=pd.DataFrame({'channel':['X - Oven'],'category':['Oven']});th=pd.DataFrame({'channel':['X - Oven'],'design_kw':[2.]})
    q=schedule_from_power(p,s,m,th,'X');r=q[(q.season=='All seasons')&(q.day_type=='Weekend')]
    assert len(r)==96 and np.allclose(r.fraction,.5)
    assert np.allclose(r.mean_kw,r.reconstructed_kw)
    p.iloc[:3]=np.nan;s.iloc[:3]=np.nan
    q=schedule_from_power(p,s,m,th,'X');r=q[(q.season=='All seasons')&(q.day_type=='Weekend')]
    assert pd.isna(r.loc[r.slot==0,'fraction']).all()

def test_representative_equal_household_weights_and_honest_denominator():
    d=pd.DataFrame({'apartment':['A','B'],'category':['AC']*2,'season':['Summer']*2,'day_type':['Weekday']*2,'slot':[0]*2,'time':['00:00']*2,'valid_days':[100,1],'fraction':[.2,.8],'mean_kw':[.4,2.4],'active_kw':[.4,2.4],'inactive_kw':[0,0],'active_probability':[.2,.8]})
    r=representative_schedule(d,['A','B','C']).iloc[0]
    assert np.isclose(r['mean'],.5) and r.contributing_apartments==2 and not r.complete_cohort
    assert np.isclose(r.energy_preserving_kw,1.4)

def test_fridge_geyser_not_in_vampire_estimate():
    idx=pd.date_range('2025-05-03',periods=3,freq='5min');p=pd.DataFrame({'Fridge':[.1]*3,'Geyser':[.1]*3,'Oven':[.1]*3},index=idx);s=p*0
    m=pd.DataFrame({'channel':p.columns,'category':p.columns})
    q=standby_summary(p,s,m,'X').set_index('category')
    assert q.loc['Fridge','candidate_standby_kwh']==0 and q.loc['Geyser','candidate_standby_kwh']==0
    assert q.loc['Oven','candidate_standby_kwh']>0

def test_epoch_timestamp_quarantined(tmp_path):
    f=tmp_path/'x.csv';pd.DataFrame({'Timestamp':['1970-01-01','2025-05-03'],'Apartment':['X']*2,'power':[1,2]}).to_csv(f,index=False)
    p,q=read_apartment(f,monitoring_start='2025-05-01',monitoring_end='2026-05-28')
    assert len(p)==1 and q['outside_monitoring_window_rows']==1

def test_residual_includes_all_nonphase_channels_even_if_unmapped():
    idx=pd.date_range('2025-05-03',periods=2,freq='5min')
    p=pd.DataFrame({'R':[2.,2.],'B':[2.,2.],'Y':[2.,2.],'AC':[2.,2.],'Lights':[1.,1.],'Unmapped':[.5,.5]},index=idx)
    m=pd.DataFrame({'channel':['R','B','Y','AC','Lights'],'category':['Phase']*3+['AC',LIGHTS]})
    b=electrical_balance(p,m)
    assert (b.total_phase_kw==6).all() and (b.total_appliance_kw==3.5).all()
    assert (b.residual_kw==2.5).all()

def test_negative_residual_is_preserved_and_not_standby():
    idx=pd.date_range('2025-05-03',periods=1,freq='5min')
    p=pd.DataFrame({'R':[1.],'B':[1.],'Y':[1.],'AC':[4.]},index=idx)
    m=pd.DataFrame({'channel':p.columns,'category':['Phase']*3+['AC']})
    b=electrical_balance(p,m)
    assert b.residual_kw.iloc[0]==-1 and b.negative_residual_flag.iloc[0]
    assert b.positive_residual_kw.iloc[0]==0

def test_missing_appliance_does_not_become_residual():
    idx=pd.date_range('2025-05-03',periods=2,freq='5min')
    p=pd.DataFrame({'R':[1.,1.],'B':[1.,1.],'Y':[1.,1.],'AC':[np.nan,0.]},index=idx)
    m=pd.DataFrame({'channel':p.columns,'category':['Phase']*3+['AC']})
    b=electrical_balance(p,m)
    assert pd.isna(b.residual_kw.iloc[0]) and not b.valid_balance.iloc[0]
    assert b.residual_kw.iloc[1]==3 and b.valid_balance.iloc[1]

def test_paired_energy_residual_reconciles_and_has_shared_coverage():
    idx=pd.date_range('2025-05-03',periods=3,freq='5min')
    p=pd.DataFrame({'R':[1.,np.nan,1.],'B':[1.,1.,1.],'Y':[1.,1.,1.],'AC':[2.,1.,4.]},index=idx)
    m=pd.DataFrame({'channel':p.columns,'category':['Phase']*3+['AC']})
    q=balance_summary(electrical_balance(p,m),'daily').iloc[0]
    assert q.valid_balance_slots==2 and np.isclose(q.balance_coverage_pct,200/3)
    assert np.isclose(q.phase_kwh_aligned,.5) and np.isclose(q.appliance_kwh_aligned,.5)
    assert np.isclose(q.residual_kwh,0) and q.negative_residual_slots==1
    assert np.isclose(q.positive_residual_kwh,1/12) and np.isclose(q.negative_residual_kwh,-1/12)
    assert np.isclose(q.reconstruction_error_kwh,0)

def test_no_balance_readings_exports_nan_energy():
    idx=pd.date_range('2025-05-03',periods=3,freq='5min')
    p=pd.DataFrame({'R':[1.]*3,'B':[1.]*3,'Y':[1.]*3,'AC':[np.nan]*3},index=idx)
    m=pd.DataFrame({'channel':p.columns,'category':['Phase']*3+['AC']})
    for scope in ['monthly','daily','hourly']:
        q=balance_summary(electrical_balance(p,m),scope)
        assert q.residual_kwh.isna().all() and q.balance_coverage_pct.eq(0).all()

def test_main_incomer_alias_does_not_create_fourth_phase():
    cols=['X - R Phase','X - B Phase','X - Y Phase','X - Main Incomer']
    book=pd.DataFrame({'Apartment Name':['X']*3,'Phases':['R Phase','B Phase','Y Phase'],'Channel 1':['X - Main Incomer','X - B Phase','X - Y Phase']})
    m=channel_mapping('X',cols,book)
    assert m.category.eq('Phase').sum()==3
    assert m.set_index('channel').loc['X - Main Incomer','category']=='Other'
    alias=channel_mapping('X',cols[1:],book)
    assert alias.category.eq('Phase').sum()==3
    assert bool(alias.set_index('channel').loc['X - Main Incomer','phase_alias'])

def test_nonphase_appliance_is_not_phase_just_because_workbook_slot_one():
    cols=['X - B Phase','X - Y Phase','X - Fridge']
    book=pd.DataFrame({'Apartment Name':['X'],'Phases':['R Phase'],'Channel 1':['X - Fridge']})
    m=channel_mapping('X',cols,book)
    assert m.set_index('channel').loc['X - Fridge','category']=='Fridge'
