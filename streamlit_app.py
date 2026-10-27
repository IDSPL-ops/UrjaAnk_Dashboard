"""Client dashboard: python -m streamlit run streamlit_app.py."""
from pathlib import Path
import io,json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.analytics import *
from src.insights import *
from src.indoor_insights import validate_indoor,align_indoor,buffering,thermal_model,ac_event_response,effectiveness_model
from src.ui import chart,download,evidence_zip,fmt,EVIDENCE
ROOT=Path(__file__).resolve().parent;DATA=ROOT/'data/processed'
st.set_page_config(page_title='Palava | Apartment insights',page_icon='⚡',layout='wide')
st.markdown('''<style>.block-container{max-width:1450px;padding-top:1.7rem}h1{font-size:2rem!important}h3{font-size:1.15rem!important}div[data-testid="stMetric"]{border:1px solid #dce5ec;border-radius:12px;padding:16px;background:#f6fafb}div[data-testid="stCaptionContainer"]{font-size:14px}</style>''',unsafe_allow_html=True)
st.title('Apartment energy insights')
if not (DATA/'manifest.json').exists():st.error('Prepared data is missing. Run python -m src.preprocess.');st.stop()
@st.cache_data(show_spinner=False)
def csv(path):return pd.read_csv(ROOT/path)
@st.cache_data(show_spinner=False,max_entries=20)
def stored(apt):return tuple(pd.read_parquet(DATA/f'{slug(apt)}_{name}.parquet') for name in ['power','flags','states'])
@st.cache_data(show_spinner=False)
def weather():return pd.read_parquet(DATA/'weather.parquet')
@st.cache_data(show_spinner=False,max_entries=40)
def scoped(apt,start,end,cleaned):
    p,f,s=stored(apt);g=pd.date_range(start,end,freq='5min');p=p.reindex(g).abs();f=f.reindex(g).eq(True);s=s.reindex(g)
    return p.mask(f) if cleaned else p,f,s
@st.cache_data(show_spinner=False,max_entries=40)
def hours(apt,start,end,cleaned,minimum):
    p,_,_=scoped(apt,start,end,cleaned);m=csv('config/channels.csv');return hourly_dataset(p,m[m.apartment.eq(apt)],weather(),minimum)
@st.cache_data(show_spinner=False,max_entries=20)
def shapes(start,end,cleaned,minimum,with_shares):
    profiles={};shares={};audit=[];mapping=csv('config/channels.csv')
    for a in csv('config/apartments.csv').apartment:
        p,_,_=scoped(a,start,end,cleaned);m=mapping[mapping.apartment.eq(a)]
        shape,n=load_shape(phase_total(p,m),minimum);audit.append({'apartment':a,'complete_profile_days':n,'eligible':shape is not None})
        if shape is not None:profiles[a]=shape;shares[a]=appliance_shares(p,m)
    return cluster_apartments(profiles,shares if with_shares else None),pd.DataFrame(audit),shares

meta=csv('config/apartments.csv');mapping=csv('config/channels.csv');qa=csv('data/processed/source_qa.csv');manifest=json.loads((DATA/'manifest.json').read_text())
VIEWS=['0 · Data coverage','1 · Apartment energy fingerprint','2 · Energy lifestyle / behavior clustering','3 · Weather-normalized consumption','4 · Cooling sensitivity index','5 · Indoor–outdoor thermal buffering','6 · Apartment thermal time constant','7 · AC cooling effectiveness','8 · Humidity / latent cooling performance','9 · Flexible load / demand response','10 · Matched-apartment benchmark','Operation schedules & evidence']
with st.sidebar:
    st.header('Analysis scope')
    apt=st.selectbox('Apartment',meta.apartment.tolist(),index=meta.apartment.tolist().index('Lakeside K 502'))
    first=pd.to_datetime(qa.start).min().date();last=pd.to_datetime(qa.end).max().date()
    dr=st.date_input('Duration',value=(first,last-pd.Timedelta(days=1)),min_value=first,max_value=last)
    if not isinstance(dr,tuple) or len(dr)!=2:st.info('Select a start and end date.');st.stop()
    start=pd.Timestamp(dr[0]);end=pd.Timestamp(dr[1])+pd.Timedelta(days=1)-pd.Timedelta(minutes=5)
    view=st.radio('Insight tab',VIEWS)
    minimum=st.slider('Minimum hourly / daily coverage (%)',50,100,90,5)/100
    cleaned=st.toggle('Exclude readings flagged for review',value=True)
    st.caption('Coverage counts supplied readings. Held interval support is identified separately. Missing energy stays unknown.')
    with st.expander('Indoor data for thermal insights'):
        uploaded=st.file_uploader('Indoor CSV',type=['csv'])
        st.download_button('Indoor CSV template',(ROOT/'data/templates/indoor_template.csv').read_bytes(),file_name='indoor_template.csv')
        st.caption('Celsius and RH %. Use apartment names shown above. Room-to-AC selection appears when a matching room is available. Timestamps use Asia/Kolkata.')
    with st.expander('Schedule cohort'):
        cohort=st.multiselect('Apartments to average',meta.apartment.tolist(),default=meta.loc[meta.default_cohort,'apartment'].tolist())
    st.caption('Input electrical kW = absolute source values. Apartment = R+B+Y. Appliance = every other electrical channel. Residual = apartment − appliance (signed).')
p,flags,states=scoped(apt,str(start),str(end),cleaned);m=mapping[mapping.apartment.eq(apt)];total=phase_total(p,m);bal=electrical_balance(p,m);cats=category_power(p,m)
h=hours(apt,str(start),str(end),cleaned,minimum)
ctx=dict(apartment=apt,start=str(start),end=str(end),timezone='Asia/Kolkata',view=view,absolute_inputs=True,residual_rule='Signed R+B+Y minus all remaining electrical channels; complete shared timestamps',review_filtered=cleaned,coverage_threshold=minimum,source='Supplied apartment CSVs, mapping workbook and Palava weather station',support_note='Native 15-minute held support is distinct from supplied 5-minute readings',project_version='2.0.0')
EVIDENCE.clear()
st.caption(f'{apt} · {start:%d %b %Y} – {end:%d %b %Y} · {view.split(" · ")[-1]}')

@st.cache_data(show_spinner=False,max_entries=16)
def selected_operation_schedule(a,begin,finish,clean):
    pp,_,ss=scoped(a,begin,finish,clean);mm=mapping[mapping.apartment.eq(a)];tt=csv('data/processed/thresholds.csv');return schedule_from_power(pp,ss,mm,tt[tt.apartment.eq(a)],a)
with st.sidebar:
    with st.expander('Selected-apartment schedule download'):
        download(selected_operation_schedule(apt,str(start),str(end),cleaned),slug(apt)+'_operation_schedule','sidebar_schedule','Download apartment schedule')

def graph(fig,rows,title,line,key,extra=None):chart(fig,rows,title,line,key,{**ctx,**(extra or {})})
def unavailable(reason,need=None):
    st.info('Inference unavailable: '+reason)
    if need:st.caption('Required evidence: '+need)
    EVIDENCE['unavailable_'+slug(view)]=(pd.DataFrame(),dict(ctx,inference='Unavailable: '+reason,required_evidence=need))
def cards(items):
    for c,(label,val) in zip(st.columns(len(items)),items):c.metric(label,val)
def model_details(model):
    if 'coefficients' in model:EVIDENCE['model_coefficients']=(model['coefficients'],dict(ctx,model={k:v for k,v in model.items() if k not in ['rows','coefficients','change_point_curve']}))
    with st.expander('Model evidence and definitions'):
        st.json({k:v for k,v in model.items() if k not in ['rows','coefficients','change_point_curve']},expanded=False)
        if 'coefficients' in model:st.dataframe(model['coefficients'],hide_index=True);download(model['coefficients'],'model_coefficients','coef')
        if 'change_point_curve' in model:download(model['change_point_curve'],'change_point_selection','cp')

def indoor_context():
    if uploaded is not None:
        try:indoor=validate_indoor(pd.read_csv(io.BytesIO(uploaded.getvalue())))
        except Exception as e:unavailable('Indoor import could not be read: '+str(e));return None
    elif (DATA/'indoor.parquet').exists():indoor=validate_indoor(pd.read_parquet(DATA/'indoor.parquet'))
    else:
        unavailable('No indoor temperature/RH time series was supplied in the saved project.','Upload the indoor CSV in the sidebar, or save data/raw/indoor.csv and run preprocessing.');return None
    indoor=indoor[indoor.Apartment.eq(apt)&indoor.Timestamp.between(start,end)]
    if indoor.empty:unavailable('No indoor readings match this apartment and duration.');return None
    room=st.selectbox('Indoor room',sorted(indoor.Room.unique()))
    channels=m.loc[m.category.eq('AC'),'channel'].tolist()
    if not channels:unavailable('No pure AC channel is mapped for this apartment.');return None
    chosen=st.selectbox('AC channel serving this room',channels,index=None,placeholder='Select the matching room AC')
    if chosen is None:unavailable('Select the AC channel that serves the selected indoor room.');return None
    indoor=indoor[indoor.Room.eq(room)];ctx.update(indoor_room=room,room_ac_channel=chosen)
    d=align_indoor(indoor,weather(),p[chosen],start,end)
    ev=pd.read_parquet(DATA/f'{slug(apt)}_events.parquet');ev=ev[ev.start.ge(start)&ev.end_exclusive.le(end+pd.Timedelta(minutes=5))]
    st.caption(f'Room: {room}. AC: {chosen.split(" - ")[-1]}. Sensor readings describe this room; whole-apartment thermal performance is not measured.')
    return d,ev,chosen

if view==VIEWS[0]:
    source=pd.read_parquet(DATA/f'{slug(apt)}_source.parquet').reindex(p.index)
    c=coverage(source,flags,start,end,'all').iloc[0]
    cards([('Missing source readings',f'{c.missing_pct:.1f}%'),('Usable source readings',f'{c.usable_pct:.1f}%'),('Complete phase intervals',f'{total.notna().mean()*100:.1f}%'),('Complete balance intervals',f'{bal.valid_balance.mean()*100:.1f}%')])
    native=qa[qa.apartment.eq(apt)].iloc[0]
    if native.fifteen_minute_cadence_days:st.info(f'{int(native.fifteen_minute_cadence_days)} days use native 15-minute sampling. Analysis holds readings within that interval; source missingness still counts only readings actually supplied.')
    for scope,x in [('monthly','period'),('daily','period'),('hourly','hour')]:
        d=coverage(source,flags,start,end,scope)
        graph(px.bar(d,x=x,y='missing_pct') if scope=='monthly' else px.line(d,x=x,y='missing_pct'),d,scope.title()+' missing data',f'Missing source readings range from {d.missing_pct.min():.1f}% to {d.missing_pct.max():.1f}%. {int((d.usable_pct>=minimum*100).sum())} of {len(d)} groups meet the selected usable-data threshold; consumption conclusions should be limited accordingly.','coverage_'+scope)
    channels=pd.DataFrame({'channel':p.columns,'source_coverage_pct':source.notna().mean().values*100}).merge(m[['channel','category']],on='channel')
    graph(px.bar(channels,x='channel',y='source_coverage_pct',color='category'),channels,'Coverage by electrical channel',f'{int(channels.source_coverage_pct.eq(0).sum())} of {len(channels)} channels have no source readings. A missing appliance channel prevents a complete residual balance, even where phase demand is measured.','channel_coverage')
    st.dataframe(qa,hide_index=True);download(qa,'all_apartment_source_audit','qa')

elif view==VIEWS[1]:
    fp,week,seasons,days=fingerprint(total,minimum)
    cards([('Recorded phase energy',fmt(total.sum(min_count=1)/12)+' kWh'),('Eligible days',str(days)),('Peak sampled demand',fmt(total.max())+' kW'),('Signed residual energy',fmt(bal.residual_kw.sum(min_count=1)/12)+' kWh')])
    if fp.empty:unavailable('No days meet the coverage threshold. Choose a monitored period or inspect coverage.')
    else:
        peak=fp.loc[fp.mean_kw.idxmax()];low=fp.loc[fp.mean_kw.idxmin()];night=fp[fp.hour<6].mean_kw.mean();even=fp[fp.hour.between(18,22)].mean_kw.mean()
        graph(px.line(fp,x='hour',y=['mean_kw','p25_kw','p75_kw'],labels={'value':'Apartment power (kW)'}),fp,'Typical 24-hour consumption',f'The typical peak occurs at {int(peak.hour):02d}:00 ({peak.mean_kw:.2f} kW), compared with {low.mean_kw:.2f} kW at {int(low.hour):02d}:00. Mean 18:00–22:59 demand is {fmt(even/night,2)} times overnight demand. The band is P25–P75 across eligible days.','fingerprint')
        means=week.groupby('day_type').mean_kw.mean();ratio=means.get('Weekend',np.nan)/means.get('Weekday',np.nan) if means.get('Weekday',0)>0 else np.nan
        graph(px.line(week,x='hour',y='mean_kw',color='day_type'),week,'Weekday versus weekend',f'Weekend mean demand is {fmt(ratio,2)} times weekday demand on eligible observations. A change in timing or level is a household-use pattern; it does not by itself demonstrate occupancy.','weekday')
        ss=seasons.groupby('season').mean_kw.mean()
        finding=f'{ss.idxmax()} has the highest observed typical demand ({ss.max():.2f} kW).' if len(ss) else 'No eligible seasonal data.'
        graph(px.line(seasons,x='hour',y='mean_kw',color='season'),seasons,'Seasonal load shapes',finding+' Seasonal readings can represent different household operation and weather; use weather normalization before attributing the difference to heat.','seasonal')
    # Include all non-phase electrical channels in a common-denominator share view.
    shares=appliance_shares(p,m)
    if shares:
        sh=pd.DataFrame({'category':shares.keys(),'share_pct':np.array(list(shares.values()))*100});top=sh.loc[sh.share_pct.idxmax()]
        graph(px.bar(sh,x='category',y='share_pct'),sh,'Appliance energy shares',f'{top.category} accounts for {top.share_pct:.1f}% of submetered energy on complete balance timestamps. Lights/fans and mixed circuits remain combined. Residual is outside this denominator.','shares')
    else:unavailable('Insufficient complete appliance readings for comparable shares.')
    b=balance_summary(bal)
    graph(px.bar(b,x='period',y=['phase_kwh_aligned','appliance_kwh_aligned','residual_kwh'],barmode='group'),b,'Phase, appliance and residual energy',f'The balance uses {bal.valid_balance.mean()*100:.1f}% of selected intervals. Positive residual may indicate unmetered or unmapped loads; {int(bal.negative_residual_flag.sum()):,} negative intervals need reconciliation. Input values are absolute; the derived residual keeps its sign.','residual')
    download(bal.rename_axis('Timestamp').reset_index(),'electrical_balance_5min','balance')

elif view==VIEWS[2]:
    choice=st.selectbox('Clustering features',['Normalized load shape','Normalized shape + appliance shares']);ctx['clustering_features']=choice
    result,audit,shares=shapes(str(start),str(end),cleaned,minimum,choice.endswith('shares'))
    st.caption('Each complete 24-hour daily shape is divided by its daily mean, so apartment size and overall demand level do not determine the shape grouping. Cluster numbers are descriptive groups, not resident identities.')
    if not result['ok']:unavailable(result['reason'])
    else:
        assignments=result['assignments'];selected=assignments[assignments.apartment.eq(apt)]
        own=int(selected.cluster.iloc[0]) if len(selected) else None
        peers=assignments.loc[assignments.cluster.eq(own)&assignments.apartment.ne(apt),'apartment'].tolist() if own else []
        line=(f'{apt} is in cluster {own} with '+', '.join(peers)+'.') if own else f'{apt} lacks enough complete daily shapes for inclusion.'
        line+=f' {len(assignments)} apartments form {result["k"]} groups (silhouette {result["silhouette"]:.2f}). These small-sample groups are exploratory and may change with dates.'
        cards([('Included apartments',str(len(assignments))),('Groups',str(result['k'])),('Shape separation',fmt(result['silhouette'],2))])
        prof=result['profiles'];centers=prof.groupby(['cluster','hour'],as_index=False).normalized_kw.mean();centers['cluster']=centers.cluster.astype(str)
        graph(px.line(centers,x='hour',y='normalized_kw',color='cluster'),centers,'Typical shape by behavior group',line,'clusters')
        prof['cluster']=prof.cluster.astype(str)
        graph(px.line(prof,x='hour',y='normalized_kw',color='cluster',line_group='apartment',hover_name='apartment'),prof,'Apartment shapes within each group','Values above 1 indicate demand above that apartment’s daily mean. Similar curves can occur in different flat sizes and do not establish the reason for that similarity.','members')
        st.dataframe(assignments,hide_index=True);download(assignments,'cluster_assignments','assignments')
        sh=pd.DataFrame([{'apartment':a,'category':c,'share_pct':v*100} for a,vals in shares.items() for c,v in vals.items()])
        if not sh.empty:graph(px.bar(sh,x='apartment',y='share_pct',color='category'),sh,'Appliance mix for shape-eligible apartments','Shares use complete phase/appliance timestamps within each apartment. Where shapes are similar but appliance shares differ, investigate end-use mix before grouping intervention measures.','cluster_shares')
    st.dataframe(audit,hide_index=True);download(audit,'clustering_eligibility','cluster_audit')

elif view==VIEWS[3]:
    moisture=st.selectbox('Moisture variable',['humidity','enthalpy_kj_kg']);ctx['moisture_variable']=moisture
    model=weather_model(h,moisture=moisture)
    if not model['ok']:unavailable(model['reason'])
    else:
        cards([('Cooling change point',fmt(model['base_c'])+' °C'),('Later holdout R²',fmt(model['holdout_r2'],2)),('Weather-model MAE',fmt(model['mae_kw'],3)+' kW'),('Calendar-only MAE',fmt(model['calendar_mae_kw'],3)+' kW')])
        if model.get('change_point_at_search_boundary'):st.caption('The fitted change point reaches the search boundary; the threshold is not well localized. Slope intervals remain conditional on that selected threshold.')
        claim='Weather adds supported predictive information for these dates.' if model['supported'] else 'Weather does not pass all validation checks; do not use the adjustment as an efficiency rating.'
        z=model['rows'].copy();days=z[['total_kw','predicted_kw','reference_weather_kw','weather_adjusted_kw']].resample('D').mean().rename_axis('date').reset_index()
        graph(px.line(days,x='date',y=['total_kw','predicted_kw','weather_adjusted_kw']),days,'Observed and weather-normalized demand',claim+f' The reference temperature is {model["reference_weather"]["temperature"]:.1f} °C. The adjustment keeps observed deviations while replacing modeled weather with the training-period median. It is an association, not proof that weather caused high use.','weather_normalized',{'reference_weather':model['reference_weather']})
        test=z[z.validation_set.eq('Later holdout')].rename_axis('Timestamp').reset_index()
        graph(px.scatter(test,x='predicted_kw',y='total_kw',color='temperature'),test,'Weather model on later observations',f'The model was fitted without these {len(test)} later hours. Its mean absolute error is {model["mae_kw"]:.3f} kW, versus {model["calendar_mae_kw"]:.3f} kW for the calendar baseline. '+claim,'weather_holdout')
        model_details(model)
    fields=pd.DataFrame({'weather_field':WEATHER_FIELDS,'paired_available_hours':[int(h[c].notna().sum()) for c in WEATHER_FIELDS]})
    st.dataframe(fields,hide_index=True);download(fields,'weather_field_availability','weather_availability')
    st.download_button('Weather extension template',(ROOT/'data/templates/weather_extension_template.csv').read_bytes(),file_name='weather_extension.csv')

elif view==VIEWS[4]:
    model=weather_model(h,target='ac_kw')
    if not model['ok']:unavailable(model['reason'])
    else:
        cards([('Apartment AC change point',fmt(model['base_c'])+' °C'),('Cooling sensitivity',fmt(model['slope'],3)+' kW/°C'),('Conditional 95% interval',f'{model["slope_low"]:.3f} – {model["slope_high"]:.3f}'),('Later holdout R²',fmt(model['holdout_r2'],2))])
        if model.get('change_point_at_search_boundary'):st.caption('The fitted change point reaches the search boundary; the threshold is not well localized. Slope intervals remain conditional on that selected threshold.')
        d=daily_cooling(model['rows'],model['base_c'])
        graph(px.scatter(d,x='cdh',y='ac_kwh',color='humidity',hover_data=['date','enthalpy']),d,'AC energy versus cooling-degree-hours',f'{len(d)} fully paired 24-hour days are shown. The fitted hourly response above {model["base_c"]:.1f} °C is {model["slope"]:.3f} kW per °C; this is also kWh per additional degree-hour. '+('The response passes the validation screen.' if model['supported'] else 'The response fails the validation screen and is not a supported cooling ranking.'),'cdh',{'base_c':model['base_c'],'CI':'95% day-cluster interval conditional on selected change point'})
        paired=model['rows'].rename_axis('Timestamp').reset_index()
        graph(px.scatter(paired,x='enthalpy_kj_kg',y='ac_kw',color='humidity'),paired,'AC demand under humid outdoor conditions','Enthalpy combines outdoor temperature and moisture. The spread at similar enthalpy describes combined operation/equipment effects; it cannot isolate latent cooling capacity.','enthalpy')
        model_details(model)
    if st.button('Compare cooling sensitivity across apartments') or st.session_state.get('compare_cooling',False):
        st.session_state.compare_cooling=True;rows=[]
        with st.spinner('Fitting apartment-specific change points…'):
            for a in meta.apartment:
                mod=weather_model(hours(a,str(start),str(end),cleaned,minimum),target='ac_kw')
                rows.append({'apartment':a,'supported':mod.get('supported',False),'slope_kw_c':mod.get('slope',np.nan),'base_c':mod.get('base_c',np.nan),'low_95':mod.get('slope_low',np.nan),'high_95':mod.get('slope_high',np.nan),'holdout_r2':mod.get('holdout_r2',np.nan),'reason':mod.get('reason','')})
        comparison=pd.DataFrame(rows);valid=comparison[comparison.supported].copy();valid['upper']=valid.high_95-valid.slope_kw_c;valid['lower']=valid.slope_kw_c-valid.low_95
        fastest=valid.loc[valid.slope_kw_c.idxmax(),'apartment'] if len(valid) else None
        graph(px.bar(valid,x='apartment',y='slope_kw_c',error_y='upper',error_y_minus='lower'),comparison,'Validated cooling sensitivity comparison',(f'{fastest} has the largest supported point estimate. Overlapping intervals do not establish a difference between apartments.' if fastest else 'No apartment passes all validation checks for this period.'),'cooling_comparison')

elif view in VIEWS[5:9]:
    item=indoor_context()
    if item:
        d,events,channel=item
        if view==VIEWS[5]:
            delta,lag,rates=buffering(d)
            cards([('Paired indoor/outdoor bins',str(len(delta))),('Median Tout − Tin',fmt(delta.delta_t.median())+' °C')])
            line=delta.resample('D').mean().rename_axis('date').reset_index()
            graph(px.line(line,x='date',y=['tin','tout']),line,'Indoor–outdoor temperature buffering',f'Indoor temperature is a median {fmt(delta.delta_t.median())} °C below outdoor temperature on paired observations. AC, windows, gains and sensor placement also affect this difference; it is not an isolated envelope rating.','buffer_delta')
            supported=lag.dropna(subset=['correlation']);best=supported.loc[supported.correlation.idxmax()] if len(supported) else None
            inference=f'The strongest AC-off first-difference correlation occurs at a {best.lag_hours:.2f}-hour lag (r={best.correlation:.2f}). This is a descriptive lag, not a measured material property.' if best is not None and best.correlation>.2 else 'No clear thermal lag is identifiable from observed AC-off transitions.'
            graph(px.line(lag,x='lag_hours',y='correlation'),lag,'AC-off thermal lag',inference,'lag')
            if len(rates):
                avg=rates.groupby('period').rate_c_per_hour.median()
                graph(px.box(rates,x='period',y='rate_c_per_hour',points='all'),rates,'Daytime rise and overnight temperature change','Median rates: '+', '.join(f'{k} {v:+.2f} °C/h' for k,v in avg.items())+'. Negative overnight change indicates cooling. These AC-off slopes require continuous observations and do not isolate ventilation or envelope effects.','buffer_rates')
            else:unavailable('No sufficiently continuous AC-off daytime/overnight windows for rise and decay rates.')
        elif view==VIEWS[6]:
            model=thermal_model(d)
            if not model['ok']:unavailable(model['reason'])
            else:
                model_details(model)
                if model['supported']:cards([('Apparent time constant',fmt(model['tau_hours'],2)+' h'),('Conditional 95% interval',f'{model["tau_low"]:.2f} – {model["tau_high"]:.2f} h'),('Later holdout R²',fmt(model['holdout_r2'],2))])
                else:unavailable('The fitted response fails physical-sign, interval or holdout checks. A thermal time constant is not reported as an accepted result.')
                z=model['rows'].rename_axis('Timestamp').reset_index()
                graph(px.scatter(z,x='predicted_d_tin',y='d_tin',color='set'),z,'Measured and modeled temperature response',f'Later-holdout R² is {fmt(model["holdout_r2"],2)}. The first-order response combines thermal mass, ventilation and unobserved gains; '+('the apparent time constant is conditional on this model.' if model['supported'] else 'it does not support a physical time-constant claim.'),'thermal_response')
        else:
            ev,traces=ac_event_response(d,events,channel)
            if ev.empty:unavailable('No complete isolated 30–360 minute AC events have indoor/outdoor readings and AC-off context before and after.')
            elif view==VIEWS[7]:
                model=effectiveness_model(ev)
                cards([('Complete isolated AC events',str(len(ev))),('Median raw temperature drop',fmt(ev.temperature_drop_c.median(),2)+' °C')])
                avg=traces.groupby('relative_minutes',as_index=False).agg(tin=('tin','mean'),tout=('tout','mean'),events=('event_id','nunique'))
                graph(px.line(avg,x='relative_minutes',y=['tin','tout']),avg,'Temperature before, during and after AC starts',f'{len(ev)} complete isolated events show a median before-to-after temperature change of {-ev.temperature_drop_c.median():+.2f} °C. Event lengths and contributor counts differ at later offsets; the average is descriptive.','ac_response')
                if model['ok']:
                    text=f'After conditioning on outdoor temperature change, outdoor RH, initial temperature difference and duration, the fitted improvement is {model["effect_c_per_kwh"]:.2f} °C/kWh (conditional 95% interval {model["low"]:.2f} to {model["high"]:.2f}). This is room temperature response, not AC COP.'
                    model_details(model)
                else:text=model['reason']+' Only the raw observed relationship is shown; weather-adjusted effectiveness is unavailable.'
                graph(px.scatter(ev,x='energy_kwh',y='temperature_drop_c',color='outdoor_rh'),ev,'AC energy and indoor-temperature improvement',text,'effectiveness')
            else:
                cards([('Complete isolated AC events',str(len(ev))),('Median RH change',fmt(ev.rh_change.median(),2)+' %-points'),('Median humidity-ratio change',fmt(ev.humidity_ratio_change_g_kg.median(),2)+' g/kg')])
                z=ev[['event_id','start','rh_change','dew_change_c','humidity_ratio_change_g_kg']]
                graph(px.scatter(ev,x='energy_kwh',y='humidity_ratio_change_g_kg',color='outdoor_rh'),ev,'Latent response during AC operation',f'Absolute moisture decreases in {(ev.humidity_ratio_change_g_kg<0).mean()*100:.1f}% of {len(ev)} complete isolated events. Median humidity-ratio change is {ev.humidity_ratio_change_g_kg.median():+.2f} g/kg; RH alone can rise as air cools. These responses also depend on ventilation and indoor moisture sources.','latent')
                z=traces.groupby('relative_minutes',as_index=False).agg(rh=('rh','mean'),dew_point_c=('dew_point_c','mean'),humidity_ratio_g_kg=('humidity_ratio_g_kg','mean'),events=('event_id','nunique'))
                for field,label in [('rh','Relative humidity (%)'),('dew_point_c','Dew point (°C)'),('humidity_ratio_g_kg','Humidity ratio (g/kg)')]:
                    graph(px.line(z,x='relative_minutes',y=field),z,label+' around AC starts',f'The mean event response uses {len(ev)} isolated events. Refer to the contributor count at each offset and compare dew point/humidity ratio with RH before concluding dehumidification.','latent_'+field)
        download(d.rename_axis('Timestamp').reset_index(),'indoor_outdoor_ac_15min','indoor_aligned')

elif view==VIEWS[9]:
    peak=st.slider('Peak-demand window (hours)',0,24,(18,22));window=st.slider('Candidate operating window (hours)',0,24,(22,24));delay=st.slider('Maximum deferral (hours)',1,24,6)
    include=st.checkbox('Screen AC events too (at most 15-minute deferral)',value=False);ctx.update(peak_window=peak,candidate_window=window,max_delay_hours=delay,screen_ac=include)
    ev=pd.read_parquet(DATA/f'{slug(apt)}_events.parquet');ev=ev[ev.start.ge(start)&ev.end_exclusive.le(end+pd.Timedelta(minutes=5))]
    if window[0]>=window[1] or peak[0]>=peak[1]:unavailable('Choose windows with a positive duration. For a window crossing midnight, assess each side separately.')
    else:
        eligible=flexibility_events(ev,peak,window,delay,include)
        if eligible.empty:unavailable('No complete geyser/washing-machine or selected AC events are available in these dates.')
        else:
            candidates=eligible[eligible.candidate];den=eligible.energy_kwh.sum();score=100*candidates.energy_kwh.sum()/den if den else 0
            cards([('Timing-screen score',fmt(score)+' / 100'),('Candidate event energy',fmt(candidates.energy_kwh.sum())+' kWh'),('Candidate / observed events',f'{len(candidates)} / {len(eligible)}')])
            summary=eligible.groupby('category',as_index=False).agg(events=('event_id','count'),candidate_events=('candidate','sum'),event_kwh=('energy_kwh','sum'),median_duration_minutes=('duration_minutes','median'))
            graph(px.bar(summary,x='category',y=['events','candidate_events'],barmode='group'),summary,'Events that fit the selected deferral window',f'{len(candidates)} of {len(eligible)} complete detected events fit the timing rule, representing {score:.1f}% of observed event energy in the selected categories. This screening score does not establish occupant acceptance, hot-water availability, safe appliance restart or energy savings.','flexibility',{'peak_window':peak,'candidate_window':window,'max_delay_hours':delay})
            graph(px.histogram(eligible,x='duration_minutes',color='category'),eligible,'Event-duration constraints',f'The median detected event is {eligible.duration_minutes.median():.0f} minutes. Every candidate fits wholly inside the requested window and its delay limit. Electrical cycling can split user sessions; confirm full operation cycles before control.','flex_durations')
            starts=eligible.groupby(['category','hour'],as_index=False).agg(events=('event_id','count'))
            graph(px.bar(starts,x='hour',y='events',color='category'),starts,'Observed appliance operation windows','These start-time counts identify established operating windows. Compare them with the proposed window when discussing acceptable shifts with residents.','flex_windows')
            download(eligible,'flexible_load_event_screen','flex_events','Download event-level candidate schedule')

elif view==VIEWS[10]:
    area=st.slider('Area match tolerance (%)',5,30,15,5)/100;floor=st.slider('Maximum floor difference',0,15,4);ctx.update(area_tolerance=area,floor_tolerance=floor)
    with st.spinner('Matching peers on simultaneous conditions…'):hourly={a:hours(a,str(start),str(end),cleaned,minimum) for a in meta.apartment}
    result=matched_benchmark(apt,meta,hourly,area,floor)
    st.caption('Peers must share BHK and orientation, fall inside the area/floor tolerances, and have measured demand at the same hours. Common station temperature and RH control weather by contemporaneous matching. Expected demand is the median peer demand scaled to the selected area. Occupancy and equipment remain uncontrolled.')
    if not result['ok']:unavailable(result['reason'],'At least two metadata-matched peers and 168 simultaneous valid demand/weather hours. Missing metadata is not treated as a match.')
    else:
        ratio=result['ratio'];cards([('Actual / expected',fmt(ratio,2)),('Shared comparison hours',str(result['hours'])),('Metadata-matched peers',str(len(result['peers'])))])
        z=result['rows'];daily=z[['total_kw','expected_kw']].resample('D').mean().rename_axis('date').reset_index()
        graph(px.line(daily,x='date',y=['total_kw','expected_kw']),daily,'Actual versus matched-apartment expectation',f'{apt} uses {(ratio-1)*100:+.1f}% energy relative to the area-scaled peer median over {result["hours"]} common hours. The ratio is {ratio:.2f}; it is a conditional benchmark, not measured equipment or envelope efficiency.','matched',{'peer_apartments':result['peers'],'area_tolerance':area,'floor_tolerance':floor})
        download(z.rename_axis('Timestamp').reset_index(),'matched_hourly_evidence','matched_hours')
    st.dataframe(result['audit'],hide_index=True);download(result['audit'],'peer_matching_audit','match_audit')

else:
    thresholds=csv('data/processed/thresholds.csv');selected=schedule_from_power(p,states,m,thresholds[thresholds.apartment.eq(apt)],apt)
    cat=st.selectbox('Schedule appliance',APPLIANCES);plot=selected[selected.category.eq(cat)&selected.season.eq('All seasons')]
    if len(plot):graph(px.line(plot,x='slot',y='fraction',color='day_type'),plot,'Selected-apartment operation schedule',f'{cat} profiles represent inferred electrical activity in these selected dates. Slot 0 is 00:00; slot 95 is 23:45. Unknown readings do not imply zero operation.','schedule')
    else:unavailable('No schedule observations are available for this appliance and duration.')
    download(selected,slug(apt)+'_operation_schedule','selected_schedule','Download selected apartment schedule')
    if st.button('Build selected-cohort average'):
        frames=[]
        for a in cohort:
            cp,_,ss=scoped(a,str(start),str(end),cleaned);am=mapping[mapping.apartment.eq(a)]
            frames.append(schedule_from_power(cp,ss,am,thresholds[thresholds.apartment.eq(a)],a))
        if frames:
            rep=representative_schedule(pd.concat(frames,ignore_index=True),cohort);download(rep,'representative_schedule','rep_schedule',f'Download {len(cohort)}-apartment mean schedule')
            graph(px.line(rep[rep.category.eq(cat)&rep.season.eq('All seasons')],x='slot',y='mean',color='day_type'),rep,f'{len(cohort)}-apartment schedule average','The mean weights apartments equally. Contributor counts can differ by appliance and time slot. Cross-household spread is not statistical uncertainty in the mean.','cohort_schedule')
        else:unavailable('No apartments are selected for the representative schedule.')
    st.info('Prediction remains a future-model placeholder. The weather and thermal regressions above are diagnostic analyses with recorded validation; no future energy forecast is presented.')
    st.caption('The original detailed phase, state/event and standby views remain available with: streamlit run legacy_dashboard.py')

st.divider()
st.download_button('Download this view: datasets + inferences (ZIP)',evidence_zip({'scope.json':json.dumps(ctx,indent=2),'channel_mapping.csv':m.to_csv(index=False)}),file_name=slug(apt)+'_'+slug(view)+'_evidence.zip',mime='application/zip',key='evidence_pack')
