"""Management dashboard. Start with: python -m streamlit run streamlit_app.py"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.analytics import *
from src.ui import chart,download,evidence_zip,empty,fmt,PALETTE,EVIDENCE
ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data/processed'
st.set_page_config(page_title='UrjaAnk | Palava energy',page_icon='⚡',layout='wide')
st.markdown('''<style>.block-container{padding-top:1.8rem;max-width:1500px}h1{font-size:2rem!important}h2{font-size:1.3rem!important}h3{font-size:1.05rem!important}div[data-testid="stMetric"]{background:white;border:1px solid #e4eaf0;border-radius:10px;padding:15px}div[data-testid="stTabs"] button{font-size:14px}div[data-testid="stCaptionContainer"]{font-size:13px}</style>''',unsafe_allow_html=True)
st.title('Palava residential energy management')
if not (DATA/'manifest.json').exists():
    st.error('Prepared data not found. From the project folder, run: python -m src.preprocess');st.stop()

@st.cache_data(show_spinner=False)
def read_csv(name):return pd.read_csv(ROOT/name)
@st.cache_data(show_spinner=False,max_entries=36)
def apartment_data(apt):
    sid=slug(apt)
    return tuple(pd.read_parquet(DATA/f'{sid}_{x}.parquet') for x in ['power','flags','states'])
@st.cache_data(show_spinner=False)
def weather_data():return pd.read_parquet(DATA/'weather.parquet')
@st.cache_data(show_spinner=False,max_entries=24)
def scoped_schedule(apt,start,end):
    raw,flags,states=apartment_data(apt);p=raw.mask(flags).loc[start:end];s=states.loc[start:end]
    m=read_csv('config/channels.csv');m=m[m.apartment==apt];th=read_csv('data/processed/thresholds.csv');th=th[th.apartment==apt]
    return schedule_from_power(p,s,m,th,apt) if len(p) else read_csv('data/processed/schedules.csv').iloc[:0]
@st.cache_data(show_spinner=False,max_entries=24)
def cohort_schedules(cohort,start,end):
    frames=[scoped_schedule(a,start,end) for a in cohort];frames=[x for x in frames if not x.empty]
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
@st.cache_data(show_spinner=False,max_entries=24)
def benchmark_data(start,end,cleaned,minimum):
    out=[];meta=read_csv('config/apartments.csv');mapping=read_csv('config/channels.csv')
    for _,a in meta.iterrows():
        p,f,_=apartment_data(a.apartment);p=p.mask(f) if cleaned else p
        grid=pd.date_range(start,end,freq='5min');p=p.reindex(grid)
        d=daily_total(p,mapping[mapping.apartment==a.apartment],minimum)
        good=d[d.eligible];bal=electrical_balance(p,mapping[mapping.apartment==a.apartment])
        paired_phase=bal.phase_kw_aligned.sum(min_count=1);paired_residual=bal.residual_kw.sum(min_count=1)
        out.append(dict(apartment=a.apartment,mean_eligible_day_kwh=good.recorded_kwh.mean(),eligible_days=len(good),area_m2=a.area_m2,kwh_m2_day=good.recorded_kwh.mean()/a.area_m2,flat_type=a.flat_type,orientation=a.orientation,floor=a.floor,mean_coverage_pct=d.coverage_pct.mean(),residual_mean_kw=bal.residual_kw.mean(),balance_coverage_pct=100*bal.valid_balance.sum()/len(bal),residual_share_pct=100*paired_residual/paired_phase if pd.notna(paired_phase) and paired_phase!=0 else np.nan,negative_residual_slots=int(bal.negative_residual_flag.sum())))
    return pd.DataFrame(out)

@st.cache_data(show_spinner=False,max_entries=12)
def cohort_coverage(cohort,start,end):
    rows=[]
    for a in cohort:
        source=pd.read_parquet(DATA/f'{slug(a)}_source.parquet')
        flag=pd.read_parquet(DATA/f'{slug(a)}_flags.parquet')
        for scope in ['monthly','daily','hourly']:
            d=coverage(source,flag,start,end,scope);d['apartment']=a;d['scope']=scope;rows.append(d)
    return pd.concat(rows,ignore_index=True) if rows else pd.DataFrame()

meta=read_csv('config/apartments.csv');qa=read_csv('data/processed/source_qa.csv');mapping=read_csv('config/channels.csv');thresholds=read_csv('data/processed/thresholds.csv')
manifest=json.loads((DATA/'manifest.json').read_text())
with st.sidebar:
    st.header('Analysis scope')
    apt=st.selectbox('Apartment',meta.apartment.tolist(),index=meta.apartment.tolist().index('Lakeside K 502'))
    minimum=st.slider('Minimum daily coverage (%)',50,100,90,5)/100
    cleaned=st.toggle('Exclude readings flagged for review',value=True)
    study_start=pd.to_datetime(qa.start).min().date();study_end=pd.to_datetime(qa.end).max().date()
    # Final export contains only midnight on 27 May; default excludes that incomplete day.
    date_range=st.date_input('Reporting dates',value=(study_start,study_end-pd.Timedelta(days=1)),min_value=study_start,max_value=study_end)
    if len(date_range)!=2:st.info('Select a start and end date.');st.stop()
    start=pd.Timestamp(date_range[0]);end=pd.Timestamp(date_range[1])+pd.Timedelta(days=1)-pd.Timedelta(minutes=5)
    st.caption('Coverage includes every expected interval in these dates, including periods before monitoring began. Energy totals include recorded intervals only.')
    default=meta.loc[meta.default_cohort.astype(str).str.lower().eq('true'),'apartment'].tolist()
    with st.expander('Representative schedule cohort'):
        cohort=st.multiselect('Apartments to average',meta.apartment.tolist(),default=default)
        st.caption(f'{len(cohort)} selected. Ten required for the ten-apartment label. Missing appliances retain their own contributor count.')
    with st.expander('Method and source status'):
        st.write('Active power: kW, confirmed by user. Electrical signs converted to absolute values. Timestamps: Asia/Kolkata.')
        st.caption('Review caps and inferred thresholds are editable. Active state means electrical activity, not a measured appliance command.')
        st.json(manifest['settings'],expanded=False)
    st.divider();st.subheader('Operation schedule downloads')
    selected_schedule=scoped_schedule(apt,str(start),str(end))
    if not selected_schedule.empty:download(selected_schedule,slug(apt)+'_operation_schedule','side_schedule','Selected apartment schedule')
    st.download_button('Indoor import template',(ROOT/'data/templates/indoor_template.csv').read_bytes(),file_name='indoor.csv',mime='text/csv')

raw,flags,states=apartment_data(apt);grid=pd.date_range(start,end,freq='5min');raw=raw.reindex(grid);flags=flags.reindex(grid).eq(True);states=states.reindex(grid)
p=raw.mask(flags) if cleaned else raw
source_power=pd.read_parquet(DATA/f'{slug(apt)}_source.parquet').reindex(grid)
held=pd.read_parquet(DATA/f'{slug(apt)}_held.parquet').reindex(grid).eq(True)
clean_power=raw.mask(flags);m=mapping[mapping.apartment==apt];th=thresholds[thresholds.apartment==apt]
metadata=meta[meta.apartment==apt].iloc[0]
context=dict(apartment=apt,start=start.isoformat(),end=end.isoformat(),timezone='Asia/Kolkata',unit='kW / kWh',electrical_rule='absolute input values before integration; E = |P| × 5/60',residual_definition='Signed (R+B+Y) minus every non-phase appliance/circuit channel, on timestamps with all inputs valid',exclude_review_flags=cleaned,native_cadence_note='Sustained 15-minute cadence is represented with flagged held values; source coverage counts measured readings only',minimum_daily_coverage=minimum,source_file=metadata.file,generated_at=manifest['generated_at'])
st.caption(f'{apt} · {date_range[0]:%d %b %Y} to {date_range[1]:%d %b %Y} · {"Review-filtered" if cleaned else "All recorded"} energy · 12 uploaded apartments / {len(cohort)} schedule cohort')
EVIDENCE.clear()
tabs=st.tabs(['1. Data coverage','2. Energy profile','3. Appliance operation','4. Weather & indoor','5. Apartment comparison','6. Operation schedules','7. Standby loads','8. Whitepaper & prediction'])

with tabs[0]:
    nativeqa=qa[qa.apartment==apt].iloc[0]
    if nativeqa.fifteen_minute_cadence_days:
        st.info(f'{int(nativeqa.fifteen_minute_cadence_days)} days have a sustained 15-minute sampling cadence. The 5-minute coverage percentages below count only supplied readings. Analysis represents each native 15-minute reading for its interval using explicitly marked held values; these are not additional measured readings.')
    st.caption('Missing = blank channel cells + absent timestamp cells. Usable excludes flagged readings. Lights/fans remain a combined circuit category.')
    c=coverage(source_power,flags,start,end,'all').iloc[0]
    cols=st.columns(4)
    for col,label,value in zip(cols,['Expected readings','Missing readings','Usable readings','Flagged readings'],[f'{c.expected_cells:,.0f}',f'{c.missing_pct:.1f}%',f'{c.usable_pct:.1f}%',f'{c.flagged_pct:.2f}%']):col.metric(label,value)
    for scope,title,x in [('monthly','Monthly data coverage','period'),('daily','Daily missing data','period'),('hourly','Missing data by hour of day','hour')]:
        d=coverage(source_power,flags,start,end,scope)
        if scope=='monthly':
            z=d.melt(id_vars=['period'],value_vars=['missing_pct','flagged_pct','usable_pct'],var_name='status',value_name='percent')
            fig=px.bar(z,x='period',y='percent',color='status',color_discrete_sequence=['#D66A6A','#E99841','#087F8C'],labels={'percent':'Expected readings (%)','period':'Month'})
        else:fig=px.line(d,x=x,y='missing_pct',labels={'missing_pct':'Missing readings (%)',x:'Date' if x=='period' else 'Hour'})
        worst=d.loc[d.missing_pct.idxmax()]
        line=f'Missing readings range from {d.missing_pct.min():.1f}% to {d.missing_pct.max():.1f}% across this view. High-gap periods provide a weaker basis for comparing consumption; missing readings are not zero energy.'
        chart(fig,d,title,line,'coverage_'+scope,context)
    channelqa=pd.DataFrame({'channel':raw.columns,'expected_slots':len(grid),'observed_slots':source_power.notna().sum().values,'flagged_slots':flags.sum().values}).merge(m[['channel','category']],on='channel')
    channelqa['missing_pct']=100*(1-channelqa.observed_slots/channelqa.expected_slots)
    chart(px.bar(channelqa,x='missing_pct',y='channel',color='category',orientation='h',color_discrete_sequence=PALETTE),channelqa,'Coverage by sensor channel','Channel coverage can differ within the same apartment. Appliance comparisons use channel-specific denominators rather than assuming all sensors have the same coverage.','coverage_channel',context)
    if cohort:
        with st.expander('Management coverage across the selected cohort'):
            cohortqa=cohort_coverage(tuple(cohort),str(start),str(end))
            monthqa=cohortqa[cohortqa.scope=='monthly'].copy()
            monthqa['period']=pd.to_datetime(monthqa.period)
            grouped=monthqa.groupby('period',as_index=False)[['expected_cells','observed_cells','flagged_cells']].sum()
            grouped['missing_pct']=100*(1-grouped.observed_cells/grouped.expected_cells)
            grouped['usable_pct']=100*(grouped.observed_cells-grouped.flagged_cells)/grouped.expected_cells
            chart(px.line(grouped,x='period',y=['missing_pct','usable_pct'],labels={'value':'Expected readings (%)'}),grouped,'Cohort-wide monthly data coverage',f'This view combines the {len(cohort)} selected apartments using summed expected and observed readings. Apartment installation dates and reduced native cadence remain part of the common-period coverage denominator.','cohort_coverage',dict(context,apartment='Selected cohort',cohort=cohort))
            download(cohortqa,'cohort_monthly_daily_hourly_coverage','cohort_qa_download','Download cohort monthly, daily and hourly missing-data audit')
    with st.expander('All apartment source audit and mapping'):
        st.dataframe(qa,width='stretch',hide_index=True);download(qa,'all_apartment_source_audit','qa_download')
        st.dataframe(m,width='stretch',hide_index=True);download(m,'selected_channel_mapping','map_download')
        conflicts=mapping[mapping.mapping_status.ne('Mapped')];st.write('Mapping exceptions');st.dataframe(conflicts,hide_index=True,width='stretch')

with tabs[1]:
    total=phase_total(p,m);days=daily_total(p,m,minimum);cats=category_power(p,m);balance=electrical_balance(p,m)
    valid=total.notna().sum();energy=(total*DT_HOURS).sum(min_count=1)
    cols=st.columns(4)
    for col,label,value in zip(cols,['Recorded apartment energy','Complete phase coverage','Mean eligible day','Peak sampled demand'],[fmt(energy)+' kWh',f'{valid/len(grid)*100:.1f}%',fmt(days.eligible_recorded_kwh.mean())+' kWh/day',fmt(total.max())+' kW']):col.metric(label,value)
    st.caption('Apartment total kW = R + B + Y phase kW. Every other electrical column is a submetered appliance/circuit, including lights/fans, mixed and unnamed loads. A701 uses the workbook-confirmed Main Incomer alias for R phase. Phase power and appliance power are never added together.')
    monthly=energy_table(p,m,'MS')
    chart(px.bar(monthly,x='period',y='recorded_kwh',color='category',color_discrete_sequence=PALETTE,labels={'period':'Month','recorded_kwh':'Recorded energy (kWh)'}),monthly,'Monthly end-use energy','Monthly columns show observed channel energy. Differences also reflect coverage and partial months; use the coverage tab before describing them as seasonal changes.','energy_monthly',context)
    enduses=monthly.groupby('category',as_index=False).agg(recorded_kwh=('recorded_kwh',lambda x:x.sum(min_count=1)),valid_cells=('valid_cells','sum'),expected_cells=('expected_cells','sum'))
    enduses['coverage_pct']=100*enduses.valid_cells/enduses.expected_cells
    available=enduses[enduses.recorded_kwh.notna()]
    if not available.empty and available.recorded_kwh.sum()>0:
        lead=available.loc[available.recorded_kwh.idxmax()]
        chart(px.pie(available,names='category',values='recorded_kwh',color_discrete_sequence=PALETTE,hole=.55),enduses,'Submetered appliance contribution',f'{lead.category} has the largest observed channel contribution ({lead.recorded_kwh/available.recorded_kwh.sum()*100:.1f}%). This share is within observed submetered appliance energy. The apartment phase total additionally captures possible unmetered/unmapped loads; use the paired residual analysis below.','energy_share',context)
    chart(px.line(days,x='date',y='recorded_kwh',labels={'recorded_kwh':'Recorded energy (kWh)','date':'Date'}),days,'Daily apartment energy',f'{int(days.eligible.sum())} of {len(days)} days meet the selected coverage threshold. The mean eligible-day energy is {fmt(days.eligible_recorded_kwh.mean())} kWh; incomplete days remain visible.','energy_daily',context)
    hp=p.mean(axis=0).to_frame('mean_kw') # not used for the load shape; hourly sample weighting below
    hourly=total.groupby(total.index.hour).agg(['mean','count','max']).reset_index().rename(columns={'index':'hour','mean':'mean_kw','count':'valid_slots','max':'peak_kw'})
    hourly.columns=['hour','mean_kw','valid_slots','peak_kw']
    peak=hourly.loc[hourly.mean_kw.idxmax(),'hour'] if hourly.mean_kw.notna().any() else None
    chart(px.line(hourly,x='hour',y='mean_kw',labels={'mean_kw':'Mean observed demand (kW)','hour':'Hour'}),hourly,'Typical hourly load profile',f'The highest mean observed apartment demand occurs at {int(peak):02}:00. This identifies a period for examining concurrent appliance use.' if peak is not None else 'No complete phase readings are available in this period.','energy_hourly',context)
    d=days[days.eligible].copy();d['day_type']=np.where(d.date.dt.dayofweek<5,'Weekday','Weekend');d['season']=season(pd.DatetimeIndex(d.date))
    comparison=d.groupby(['season','day_type'],as_index=False).agg(mean_kwh=('recorded_kwh','mean'),valid_days=('date','count'))
    chart(px.bar(comparison,x='season',y='mean_kwh',color='day_type',barmode='group',color_discrete_sequence=PALETTE,labels={'mean_kwh':'Mean eligible-day kWh'}),comparison,'Season and weekday/weekend consumption','This comparison excludes low-coverage days. Seasonal groups are defined by month, and differences describe measured use rather than isolate weather or occupant effects.','energy_daytype',context)
    st.subheader('Residual loads outside the submetered appliance total')
    st.caption('Residual kW = total phase kW − total appliance kW. Every appliance/circuit column is included, not only AC, washing machine, geyser, oven and fridge. Residuals require concurrent valid readings from all phase and appliance channels.')
    paired_phase=balance.phase_kw_aligned.sum(min_count=1)*DT_HOURS
    paired_appliance=balance.appliance_kw_aligned.sum(min_count=1)*DT_HOURS
    residual_energy=balance.residual_kw.sum(min_count=1)*DT_HOURS
    balance_slots=int(balance.valid_balance.sum());negative_slots=int(balance.negative_residual_flag.sum())
    residcols=st.columns(4)
    for col,label,value in zip(residcols,['Paired phase energy','Paired appliance energy','Signed residual energy','Complete balance coverage'],[fmt(paired_phase)+' kWh',fmt(paired_appliance)+' kWh',fmt(residual_energy)+' kWh',f'{100*balance_slots/len(grid):.1f}%']):col.metric(label,value)
    inference=f'{balance_slots:,} valid common intervals support this balance. Positive residuals may represent unmetered or unmapped loads. {negative_slots:,} intervals have a negative residual and remain signed for measurement/mapping reconciliation.'
    if balance_slots==0:inference='Residual load cannot be calculated in this period because no timestamp has all required phase and appliance readings. Missing appliance values are not treated as zero.'
    paired_month=balance_summary(balance,'monthly')
    paired_long=paired_month.melt(id_vars=['period','valid_balance_slots','balance_coverage_pct'],value_vars=['phase_kwh_aligned','appliance_kwh_aligned','residual_kwh'],var_name='component',value_name='energy_kwh')
    chart(px.bar(paired_long,x='period',y='energy_kwh',color='component',barmode='group',color_discrete_sequence=PALETTE,labels={'period':'Month','energy_kwh':'Paired energy (kWh)'}),paired_month,'Monthly phase, appliance and residual energy',inference,'phase_appliance_residual',context)
    residual_daily=balance_summary(balance,'daily')
    chart(px.line(residual_daily,x='period',y='mean_residual_kw',labels={'period':'Date','mean_residual_kw':'Mean signed residual (kW)'}),residual_daily,'Daily residual load',inference,'residual_daily',context)
    residual_hourly=balance_summary(balance,'hourly')
    chart(px.line(residual_hourly,x='hour',y='mean_residual_kw',labels={'hour':'Hour','mean_residual_kw':'Mean signed residual (kW)'}),residual_hourly,'Hourly residual-load pattern','Recurring positive residuals indicate when loads outside the submetered total may operate. Their timing alone cannot identify the appliances or establish standby consumption. The export retains valid-interval counts and negative values.','residual_hourly',context)
    with st.expander('Residual interval data and balance definitions'):
        download(balance.rename_axis('Timestamp').reset_index(),slug(apt)+'_electrical_balance_5min','residual_5min_download','Download phase, appliance and residual kW (5-minute grid)')
        st.write('Positive residual: possible unmetered/unmapped consumption. Negative residual: appliance subtotal exceeds the phase total and requires reconciliation. Absolute-value conversion applies to input consumption; it is not applied to the residual. The paired totals use the same timestamps, so phase energy = appliance energy + signed residual energy.')

with tabs[2]:
    device=st.selectbox('Appliance category',APPLIANCES,key='device')
    devicecols=m.loc[m.category.eq(device),'channel'].tolist()
    if not devicecols:empty(f'No separate {device} channel is provided for {apt}.')
    else:
        runtime=[]
        for c in devicecols:
            s=states[c];hours=s.sum()*DT_HOURS;observed=s.notna().sum()*DT_HOURS
            runtime.append(dict(channel=c,active_hours=hours,valid_hours=observed,duty_pct=100*hours/observed if observed else np.nan))
        rt=pd.DataFrame(runtime)
        chart(px.bar(rt,x='channel',y='duty_pct',color_discrete_sequence=PALETTE,labels={'duty_pct':'Electrical active share (%)'}),rt,'Inferred appliance runtime',f'{device} states use a separate measured-distribution threshold for each channel, with hysteresis and missing-data resets. For refrigerators, active means compressor-like electrical activity, not that the refrigerator was switched off between cycles.','runtime',context)
        ev=pd.read_parquet(DATA/f'{slug(apt)}_events.parquet');ev=ev[(ev.start>=start)&(ev.end_exclusive<=end+pd.Timedelta(minutes=5))&ev.category.eq(device)].copy()
        reliable=ev[~ev.left_censored&~ev.right_censored]
        dist=reliable.groupby('hour',as_index=False).agg(events=('event_id','count'))
        chart(px.bar(dist,x='hour',y='events',color_discrete_sequence=PALETTE),dist,'Observed event start times',f'{len(reliable):,} complete detected electrical events are available. Five-minute sampling cannot resolve short appliance cycles, and thermostat/inverter cycling can split a user session into several events.','event_starts',context)
        active_profile=states[devicecols].max(axis=1).where(states[devicecols].notna().all(axis=1))
        activityweek=pd.DataFrame({'weekday':active_profile.index.dayofweek,'hour':active_profile.index.hour,'active':active_profile.values}).groupby(['weekday','hour'],as_index=False).agg(active_fraction=('active','mean'),valid_slots=('active','count'))
        heat=activityweek.pivot(index='weekday',columns='hour',values='active_fraction');heat.index=heat.index.map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})
        chart(px.imshow(heat,zmin=0,zmax=1,aspect='auto',color_continuous_scale='Teal'),activityweek,'Day-of-week appliance pattern','These fractions show the timing of inferred electrical activity on each weekday. Unknown states are excluded from the denominator.','appliance_weekly',context)
        chart(px.histogram(reliable,x='duration_minutes',nbins=35,color_discrete_sequence=PALETTE),reliable[['channel','start','duration_minutes','energy_kwh']],'Detected event durations',f'The median complete detected event lasts {fmt(reliable.duration_minutes.median(),0)} minutes. This is an electrical-event duration, not a confirmed user-operation session.','event_durations',context)
        with st.expander('State thresholds and event records'):
            st.dataframe(th[th.category==device],width='stretch',hide_index=True);download(th[th.category==device],'state_thresholds','threshold_download')
            download(ev,'detected_events','events_download');download(states[devicecols].rename_axis('Timestamp').reset_index(),'inferred_states_5min','states_download');st.dataframe(ev.head(200),width='stretch',hide_index=True)
        first=next((c for c in devicecols if states[c].eq(1).any()),devicecols[0])
        t=states.index[states[first].eq(1)][0] if states[first].eq(1).any() else grid[0]
        timeline=pd.DataFrame({'power_kw':clean_power[first],'state':states[first]}).loc[t.floor('D'):t.floor('D')+pd.Timedelta(days=1)]
        timeline=timeline.rename_axis('Timestamp').reset_index()
        fig=go.Figure();fig.add_trace(go.Scatter(x=timeline.Timestamp,y=timeline.power_kw,name='Power (kW)'));fig.add_trace(go.Scatter(x=timeline.Timestamp,y=timeline.state,name='Inferred active',yaxis='y2',line=dict(shape='hv')));fig.update_layout(yaxis2=dict(overlaying='y',side='right',range=[-.05,1.1]))
        chart(fig,timeline,'Power and inferred state: example day',f'This example shows how measured power on {first.split(" - ")[-1]} produces the inferred state. Review near-threshold readings before treating runtime as a precise measure of appliance use.','state_timeline',context)
    validstate=states.notna().all(axis=1);coincidence=states.sum(axis=1,min_count=len(states.columns))
    cc=coincidence.value_counts(dropna=True).sort_index().rename_axis('simultaneously_active_channels').reset_index(name='slots');cc['hours']=cc.slots*DT_HOURS
    chart(px.bar(cc,x='simultaneously_active_channels',y='hours',color_discrete_sequence=PALETTE),cc,'Appliance coincidence','Coincidence counts electrically active monitored AC, washing-machine, geyser, oven and fridge channels. Missing channel states exclude an interval rather than count it as inactive.','coincidence',context)
    peakidx=total.nlargest(10).index
    peakrows=p.loc[peakidx,m.loc[~m.category.eq('Phase'),'channel']].T.rename_axis('channel').reset_index()
    peaklong=peakrows.melt(id_vars='channel',var_name='Timestamp',value_name='power_kw').merge(m[['channel','category']],on='channel')
    peaklong=peaklong.groupby(['Timestamp','category'],as_index=False).power_kw.sum(min_count=1)
    residual_peaks=balance.loc[peakidx,['residual_kw']].dropna().rename_axis('Timestamp').reset_index().rename(columns={'residual_kw':'power_kw'});residual_peaks['category']='Residual (signed; possible unmetered/unmapped)'
    peaklong=pd.concat([peaklong,residual_peaks],ignore_index=True)
    chart(px.bar(peaklong,x='Timestamp',y='power_kw',color='category',color_discrete_sequence=PALETTE),peaklong,'End-use readings at the ten highest sampled peaks','These appliance readings and available signed residuals occur at the highest complete-phase demand intervals. Residuals are shown only when all appliance channels are valid. Examine AC/geyser overlap before considering any load-shifting action; no flexibility or savings are quantified here.','peak_drivers',context)
    # A transparent activity score, never an unvalidated occupancy probability.
    discretionary=[c for c in states if classify_channel(c) in ['Washing Machine','Geyser','Oven']]
    if discretionary:
        act=states[discretionary].max(axis=1).where(states[discretionary].notna().all(axis=1))
        activity=act.groupby(act.index.hour).agg(['mean','count']).reset_index();activity.columns=['hour','discretionary_activity_share','valid_slots']
        chart(px.line(activity,x='hour',y='discretionary_activity_share'),activity,'Household appliance activity indicator','This shows the fraction of valid intervals with an active washing machine, geyser or oven. Automated heating and appliance cycles prevent interpreting it as measured occupancy.','activity',context)

with tabs[3]:
    weather=weather_data();w=weather.loc[start:end]
    st.caption('Weather measurements are predominantly hourly. The charts preserve observations; the 15-minute export holds the latest measurement for at most 60 minutes and records its age. Longer gaps remain missing.')
    # Native hourly QA: count distinct hourly bins rather than the number of irregular rows.
    wh=weather[['temperature','humidity']].resample('h').mean().reindex(pd.date_range(start.floor('h'),end.floor('h'),freq='h'))
    wc=pd.DataFrame({'expected_hours':1,'valid_hours':wh.notna().all(axis=1).astype(int)},index=wh.index).resample('MS').sum().rename_axis('month').reset_index();wc['missing_pct']=100*(1-wc.valid_hours/wc.expected_hours)
    chart(px.bar(wc,x='month',y='missing_pct',color_discrete_sequence=['#D66A6A']),wc,'Outdoor weather coverage','Monthly gaps limit weather-energy comparisons. An hour counts as observed only when both outdoor temperature and RH are available in that hour.','weather_coverage',context)
    if not w.empty:
        ws=w[['temperature','humidity','dew_point_c','enthalpy_kj_kg']].resample('D').mean().rename_axis('date').reset_index()
        chart(px.line(ws,x='date',y=['temperature','dew_point_c'],labels={'value':'Temperature (°C)'}),ws,'Outdoor temperature and dew point','Dew point expresses moisture conditions without the temperature dependence of relative humidity. Both series are daily averages of available weather readings.','weather_conditions',context)
        native=wh.copy();native['cdh']=np.maximum(native.temperature-24,0) # 24°C is diagnostic, not calibrated.
        wd=native.resample('D').agg(temperature=('temperature','mean'),humidity=('humidity','mean'),enthalpy=('temperature','count'),cdh=('cdh',lambda x:x.sum(min_count=1)))
        wd=wd.rename(columns={'enthalpy':'valid_weather_hours'});wd['weather_coverage_pct']=100*wd.valid_weather_hours/24
        ac=cats['AC'] if 'AC' in cats else pd.Series(np.nan,index=grid)
        ad=pd.DataFrame({'ac_kwh':(ac*DT_HOURS).resample('D').sum(min_count=1),'ac_valid_slots':ac.resample('D').count()})
        paired=ad.join(wd);paired['ac_coverage_pct']=100*paired.ac_valid_slots/288
        pairs=paired[(paired.ac_coverage_pct>=minimum*100)&(paired.weather_coverage_pct>=minimum*100)].rename_axis('date').reset_index()
        corr=pairs.ac_kwh.corr(pairs.temperature) if len(pairs)>=10 and pairs.ac_kwh.std()>0 and pairs.temperature.std()>0 else np.nan
        chart(px.scatter(pairs,x='temperature',y='ac_kwh',color='humidity',labels={'temperature':'Mean outdoor temperature (°C)','ac_kwh':'Recorded AC energy (kWh/day)'}),pairs,'AC use and outdoor conditions',f'{len(pairs)} paired days meet the coverage threshold. Pearson correlation is {fmt(corr,2)}; weather, operation and equipment effects remain combined, so this does not establish causation.','ac_weather',context)
        # Integrated master table with full electrical bins and honest weather age.
        pt=total.resample('15min').mean().where(total.resample('15min').count().eq(3))
        fused=pt.rename('total_kw').to_frame().reset_index().rename(columns={'index':'Timestamp'})
        obs=weather.reset_index().rename(columns={'Timestamp':'weather_observed_at'})
        fused=pd.merge_asof(fused.sort_values('Timestamp'),obs.sort_values('weather_observed_at'),left_on='Timestamp',right_on='weather_observed_at',direction='backward',tolerance=pd.Timedelta(minutes=60))
        fused['weather_age_minutes']=(fused.Timestamp-fused.weather_observed_at).dt.total_seconds()/60
        fused['energy_kwh']=fused.total_kw*.25
        for field in ['total_appliance_kw','residual_kw']:
            series=balance[field]
            fused[field]=series.resample('15min').mean().where(series.resample('15min').count().eq(3)).to_numpy()
        fused['held_support_fraction']=held.mean(axis=1).resample('15min').mean().to_numpy()
        download(fused,'integrated_15min_electrical_weather','fused_download','Download 15-minute integrated dataset')
    placeholders=pd.DataFrame({'field':['irradiance_w_m2','wind_speed_m_s','wind_direction_deg','rainfall_mm','pressure_hpa'],'status':['Available' if weather[c].notna().any() else 'Not provided' for c in ['irradiance_w_m2','wind_speed_m_s','wind_direction_deg','rainfall_mm','pressure_hpa']]})
    st.dataframe(placeholders,width='stretch',hide_index=True)
    st.download_button('Weather extension template',(ROOT/'data/templates/weather_extension_template.csv').read_bytes(),file_name='weather_extension.csv',mime='text/csv')
    ip=DATA/'indoor.parquet'
    if not ip.exists():empty('Indoor temperature/RH readings were not supplied in these apartment CSVs. Import data/raw/indoor.csv using the template, then rerun preprocessing. Indoor-response and cooling-effectiveness results remain unavailable until those data are present.')
    else:
        indoor=pd.read_parquet(ip);indoor=indoor[(indoor.Apartment==apt)&(indoor.Timestamp.between(start,end))]
        if indoor.empty:empty('No indoor sensor readings are available for this apartment and period.')
        else:
            room=st.selectbox('Indoor sensor room',sorted(indoor.Room.unique()));indoor=indoor[indoor.Room==room].set_index('Timestamp').resample('15min')[['temperature','humidity']].mean()
            obs=weather[['temperature','humidity']].reset_index().rename(columns={'Timestamp':'weather_observed_at','temperature':'outdoor_temperature','humidity':'outdoor_humidity'})
            indoors=pd.merge_asof(indoor.reset_index().sort_values('Timestamp'),obs.sort_values('weather_observed_at'),left_on='Timestamp',right_on='weather_observed_at',direction='backward',tolerance=pd.Timedelta(minutes=60))
            indoors['delta_t']=indoors.outdoor_temperature-indoors.temperature
            chart(px.line(indoors,x='Timestamp',y=['temperature','outdoor_temperature']),indoors,'Indoor and outdoor temperature','Indoor/outdoor differences describe this sensor location. Occupancy, windows and AC operation are not separately controlled, so the difference is not an envelope-efficiency rating.','indoor_outdoor',context)
            chart(px.scatter(indoors,x='temperature',y='humidity'),indoors,'Indoor temperature and humidity','The joint indoor readings describe environmental conditions. They do not establish full thermal comfort without air speed, clothing, metabolic rate and occupancy information.','indoor_conditions',context)
            ic=pd.DataFrame({'expected_bins':1,'observed_bins':indoor.notna().all(axis=1).astype(int)},index=indoor.index).resample('D').sum().rename_axis('date').reset_index();ic['missing_pct']=100*(1-ic.observed_bins/ic.expected_bins)
            download(ic,'indoor_daily_coverage','indoor_coverage_download')

with tabs[4]:
    bench=benchmark_data(str(start),str(end),cleaned,minimum)
    bench['eligible_for_comparison']=bench.eligible_days>=10
    plot=bench[bench.eligible_for_comparison]
    chart(px.bar(plot,x='apartment',y='kwh_m2_day',color='flat_type',color_discrete_sequence=PALETTE,labels={'kwh_m2_day':'Recorded kWh/m² per eligible day'}),bench,'Apartment energy intensity',f'{len(plot)} apartments have at least ten eligible days. Area normalization helps describe differences, but each apartment can have different eligible dates, household use and equipment; this is not an efficiency ranking.','benchmark',dict(context,apartment='All 12 uploaded apartments'))
    residual_bench=bench[['apartment','residual_mean_kw','residual_share_pct','balance_coverage_pct','negative_residual_slots']]
    chart(px.bar(residual_bench,x='apartment',y='residual_share_pct',color_discrete_sequence=PALETTE,labels={'residual_share_pct':'Signed residual / paired phase energy (%)'}),residual_bench,'Residual contribution by apartment','Residual shares use phase and appliance totals on identical complete timestamps. Positive shares may reflect unmetered/unmapped loads; negative values require reconciliation. Compare the accompanying coverage before drawing conclusions.','residual_benchmark',dict(context,apartment='All 12 uploaded apartments'))
    st.caption('Matched comparison uses the same valid dates for the selected apartment and each peer, matching flat type and area within 10%.')
    peers=meta[(meta.flat_type==metadata.flat_type)&meta.flat_type.notna()&(abs(meta.area_m2/metadata.area_m2-1)<=.1)&meta.apartment.ne(apt)] if pd.notna(metadata.flat_type) else meta.iloc[:0]
    matched=[]
    base=days[days.eligible].set_index('date')
    for _,peer in peers.iterrows():
        pr,pf,_=apartment_data(peer.apartment);pr=(pr.mask(pf) if cleaned else pr).reindex(grid);pdaily=daily_total(pr,mapping[mapping.apartment==peer.apartment],minimum);pg=pdaily[pdaily.eligible].set_index('date')
        common=base.index.intersection(pg.index)
        if len(common)>=10:
            av=base.loc[common,'recorded_kwh'].mean();pv=pg.loc[common,'recorded_kwh'].mean()
            matched.append(dict(peer=peer.apartment,common_days=len(common),selected_mean_kwh=av,peer_mean_kwh=pv,selected_kwh_m2=av/metadata.area_m2,peer_kwh_m2=pv/peer.area_m2))
    matched=pd.DataFrame(matched)
    if matched.empty:empty('No peer meets the typology, area and common-date requirements in the selected period. Unknown Marvella metadata prevents typology matching.')
    else:
        mm=matched.melt(id_vars=['peer','common_days'],value_vars=['selected_kwh_m2','peer_kwh_m2'],var_name='series',value_name='kwh_m2_day')
        chart(px.bar(mm,x='peer',y='kwh_m2_day',color='series',barmode='group',color_discrete_sequence=PALETTE),matched,'Comparable flats on common observed dates','Each pair uses identical valid dates and a similar flat type/area. Residual differences still include household operation and equipment, so floor/orientation effects require controlled simulation.','matched_flats',dict(context,apartment=apt))
    with st.expander('Apartment metadata'):
        st.dataframe(meta,width='stretch',hide_index=True);download(meta,'apartment_metadata','metadata_download')
    empty('Weather-normalized expected energy, performance ratios and floor/orientation causal effects require a validated model. Prediction is reserved in the final tab; no model scores are fabricated.')

with tabs[5]:
    st.caption('96 fifteen-minute slots per weekday/weekend and season. Electrical states use the native measurements on a 5-minute grid; sustained 15-minute intervals have marked held support. Each 15-minute input bin requires all three valid readings. Normalization uses cleaned active-state P99.5 design power; OEM ratings can override it. P99.5 is calculated on the inferred active state; excess power is exported as clipping_loss_kw.')
    schedseason=st.selectbox('Schedule season',['All seasons','Summer','Monsoon','Mild'],key='schedseason')
    daytype=st.radio('Schedule day type',['Weekday','Weekend'],horizontal=True,key='sched_day')
    s=selected_schedule[(selected_schedule.season==schedseason)&(selected_schedule.day_type==daytype)] if not selected_schedule.empty else pd.DataFrame()
    if not s.empty:
        chart(px.line(s,x='time',y='fraction',color='category',color_discrete_sequence=PALETTE,labels={'fraction':'Normalized active load (0–1)','time':'Time of day'}),s,'Selected-apartment operation schedule','This fractional profile averages observed active demand. A low continuous fraction can represent occasional events across days; it does not mean an appliance operates continuously. AC availability is exported separately from its electrical load fraction.','schedule_single',dict(context,season=schedseason,day_type=daytype))
        heat=s.pivot(index='category',columns='time',values='active_probability')
        chart(px.imshow(heat,zmin=0,zmax=1,aspect='auto',color_continuous_scale='Teal',labels={'color':'Active fraction'}),s,'Appliance electrical activity by time','The heatmap shows the observed fraction of intervals with at least one electrically active device in each category. It is an empirical availability proxy, not a thermostat setting or command schedule.','schedule_heatmap',dict(context,season=schedseason,day_type=daytype))
    schedules=cohort_schedules(tuple(cohort),str(start),str(end)) if cohort else pd.DataFrame()
    representative=representative_schedule(schedules,cohort) if not schedules.empty else pd.DataFrame()
    if len(cohort)!=10:st.warning(f'{len(cohort)} apartments selected. This is a {len(cohort)}-apartment schedule, not a ten-apartment average.')
    if not representative.empty:
        rep=representative[(representative.season==schedseason)&(representative.day_type==daytype)]
        category=st.selectbox('Representative appliance',APPLIANCES,key='rep_device');r=rep[rep.category==category]
        if r.empty:empty('No eligible cohort observations for this appliance, season and day type.')
        else:
            fig=go.Figure();fig.add_trace(go.Scatter(x=r.time,y=r.p75,name='P75',line=dict(width=0)));fig.add_trace(go.Scatter(x=r.time,y=r.p25,name='P25–P75',fill='tonexty',line=dict(width=0),fillcolor='rgba(8,127,140,.16)'));fig.add_trace(go.Scatter(x=r.time,y=r['mean'],name='Mean',line=dict(color='#087F8C',width=3)));fig.add_trace(go.Scatter(x=r.time,y=r['median'],name='Median',line=dict(color='#E99841')))
            fig.update_yaxes(range=[0,1],title='Normalized active load')
            chart(fig,r,f'Representative {category} schedule',f'Each apartment has equal weight after capacity normalization. Contributors range from {r.contributing_apartments.min()} to {r.contributing_apartments.max()} of {len(cohort)} selected apartments. P25–P75 describes variation between apartment profiles, not a forecast confidence interval.','schedule_cohort',dict(context,apartment='Selected cohort',cohort=cohort,season=schedseason,day_type=daytype))
        download(representative,'representative_cohort_schedule','rep_download','Download full cohort schedule')
        download(schedules,'cohort_individual_schedules','cohort_individual_download','Download all cohort individual schedules')
        qa_sched=selected_schedule.groupby(['category','season','day_type'],as_index=False).agg(measured_typical_day_kwh=('mean_kw',lambda x:x.sum(min_count=96)*.25),reconstructed_typical_day_kwh=('reconstructed_kw',lambda x:x.sum(min_count=96)*.25),maximum_bin_error_pct=('reconstruction_error_pct',lambda x:x.abs().max()),minimum_valid_days=('valid_days','min'))
        qa_sched['error_pct']=100*(qa_sched.reconstructed_typical_day_kwh-qa_sched.measured_typical_day_kwh)/qa_sched.measured_typical_day_kwh.replace(0,np.nan)
        st.dataframe(qa_sched,width='stretch',hide_index=True);download(qa_sched,'schedule_energy_validation','schedule_validation_download')
        st.caption('Individual reconstruction = design power × fraction + measured inactive power + over-design power. Over-design power is retained in clipping_loss_kw and must be included when reproducing energy. Cohort mean kW preserves mean recorded power; multiplying mean fraction by one common rated power need not preserve energy across different appliance capacities.')
    else:empty('Select at least one apartment with valid data to generate a representative schedule.')
    # Timestamped 15-min availability and fraction, for schedule-file workflows.
    cols=m.loc[m.category.isin(APPLIANCES),'channel'].tolist();chron=clean_power[cols].resample('15min').mean().where(clean_power[cols].resample('15min').count().eq(3))
    for c in cols:
        design=float(th.loc[th.channel==c,'design_kw'].iloc[0]);sp=states[c]
        active=clean_power[c].where(sp.eq(1),0).where(sp.notna())
        ap=active.resample('15min').mean().where(active.resample('15min').count().eq(3))
        chron[c+' | active_fraction']=(ap/design).clip(0,1)
        chron[c+' | active_probability']=sp.resample('15min').mean().where(sp.resample('15min').count().eq(3))
        chron[c+' | over_design_kw']=(ap-design*chron[c+' | active_fraction']).clip(lower=0)
        chron[c+' | held_sample_fraction']=held[c].resample('15min').mean()
    download(chron.rename_axis('Timestamp').reset_index(),'selected_apartment_timestamped_power_15min','schedule_chron','Download timestamped 15-minute simulation schedules')

with tabs[6]:
    standby=standby_summary(clean_power,states,m,apt)
    candidate=standby.candidate_standby_kwh.sum()
    cols=st.columns(3);cols[0].metric('Candidate standby energy',fmt(candidate)+' kWh');cols[1].metric('Devices with candidate low load',str(int(standby.candidate_standby_kwh.gt(0).sum())));cols[2].metric('Verified avoidable energy','Unavailable')
    st.caption('Candidate low-power residuals can include controllers, meter offsets or another connected load. Refrigerator cycling and geyser heating are excluded from the vampire estimate. No avoidable savings are claimed without field verification.')
    z=standby.melt(id_vars=['channel','category'],value_vars=['active_kwh','inactive_low_kwh','off_noise_kwh'],var_name='electrical_state',value_name='energy_kwh')
    chart(px.bar(z,x='channel',y='energy_kwh',color='electrical_state',color_discrete_sequence=PALETTE),standby,'Active and inactive appliance energy','The inactive component is a measured low-power residual. Only AC, washing-machine and oven residuals contribute to candidate standby; they still require verification.','standby_states',context)
    cand=standby[standby.category.isin(['AC','Washing Machine','Oven'])]
    chart(px.bar(cand,x='channel',y='candidate_kwh_per_valid_day',color='category',color_discrete_sequence=PALETTE),cand,'Candidate standby by device',f'The selected-period candidate residual totals {fmt(candidate)} kWh. Per-day values normalize by valid observation hours; their annualized values are extrapolation scenarios with no guarantee of seasonal representativeness.','standby_devices',context)
    st.dataframe(standby,width='stretch',hide_index=True);download(standby,'candidate_standby_library','standby_download')
    base=pd.DataFrame({'date':total.resample('D').mean().index,'p05_kw':total.resample('D').quantile(.05).values,'valid_slots':total.resample('D').count().values});base['eligible']=base.valid_slots/288>=minimum;base['p05_kw']=base.p05_kw.where(base.eligible)
    chart(px.line(base,x='date',y='p05_kw',labels={'p05_kw':'Daily low-load P5 (kW)'}),base,'Apartment low-load indicator','The daily fifth percentile describes low observed apartment demand. It includes necessary services such as refrigeration and is not an estimate of vampire load.','baseload',context)

with tabs[7]:
    st.subheader('Whitepaper evidence pack')
    st.write('Every chart provides its displayed dataset and a scope-specific inference. The pack below includes those datasets, inference lines, reporting dates, source identity and calculation rules.')
    st.download_button('Download all displayed chart evidence (ZIP)',evidence_zip({'manifest.json':json.dumps(manifest,indent=2),'operation_schedule_selected.csv':selected_schedule.to_csv(index=False),'operation_schedule_cohort.csv':representative.to_csv(index=False) if not representative.empty else '', 'channel_mapping.csv':m.to_csv(index=False),'electrical_balance_5min.csv':balance.rename_axis('Timestamp').reset_index().to_csv(index=False)}),file_name=slug(apt)+'_whitepaper_evidence.zip',mime='application/zip',key='all_evidence')
    st.dataframe(pd.DataFrame([{'chart':v[1]['chart'],'inference':v[1]['inference'],'dataset_rows':len(v[0])} for v in EVIDENCE.values()]),width='stretch',hide_index=True)
    st.subheader('Prediction model: reserved for a later phase')
    status=pd.DataFrame({'output':['Next-day total kWh','Next-day AC activity','Peak-demand interval','P10 / P50 / P90 forecast','Weather-normalized expected energy'],'status':['Not trained']*5})
    st.dataframe(status,width='stretch',hide_index=True)
    st.caption('Planned model: global gradient boosting with apartment metadata, past load, weather and calendar features. Evaluate with rolling time splits, naive persistence baselines and leave-one-apartment-out checks. Future appliance states and measured future weather must not leak into deployable forecasts.')
    st.subheader('Output availability')
    availability=pd.DataFrame({'output':['Missing-data monthly/daily/hourly','Energy profiles and end uses','Phase total, appliance total and signed residual','Electrical state/event and coincidence','Weather and cooling association','Indoor temperature/RH diagnostics','Individual / representative schedules','Standby candidate library','Thermal time constant / full comfort','Validated expected energy / predictions','Simulation validation (Deliverable 3)'],'status':['Implemented','Implemented','Implemented; common timestamp balance','Implemented; inferred states','Implemented; coverage-filtered','Requires indoor readings' if not manifest['indoor_available'] else 'Available for supplied sensors','Implemented; contributor counts retained','Implemented; field verification needed','Requires additional inputs / calibrated model','Placeholder as requested','Separate future deliverable']})
    st.dataframe(availability,width='stretch',hide_index=True);download(availability,'deliverable_output_status','availability_download')
    st.subheader('Add data and refresh')
    st.write('Add optional indoor.csv and weather_extension.csv to data/raw using the supplied templates. Edit thresholds in config/threshold_overrides.csv if reviewed device details are available. Run python -m src.preprocess, then restart the dashboard or clear its data cache.')
    with st.expander('Source manifest and methodological limits'):
        st.json(manifest,expanded=False)
        st.download_button('Download source manifest',json.dumps(manifest,indent=2),file_name='source_manifest.json',mime='application/json')
