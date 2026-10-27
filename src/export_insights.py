"""Reproducible default-scope insight pack: python -m src.export_insights."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .analytics import *
from .insights import *
from .ui import json_text
ROOT=Path(__file__).resolve().parents[1]

def run():
    out=ROOT/'exports/insights';out.mkdir(exist_ok=True,parents=True)
    meta=pd.read_csv(ROOT/'config/apartments.csv');mapping=pd.read_csv(ROOT/'config/channels.csv');qa=pd.read_csv(ROOT/'data/processed/source_qa.csv');weather=pd.read_parquet(ROOT/'data/processed/weather.parquet')
    start=pd.to_datetime(qa.start).min().normalize();end=pd.to_datetime(qa.end).max().normalize()-pd.Timedelta(minutes=5)
    idx=pd.date_range(start,end,freq='5min');panels={};profiles={};shares={};models=[];eligibility=[]
    for a in meta.apartment:
        sid=slug(a);folder=out/sid;folder.mkdir(exist_ok=True)
        p=pd.read_parquet(ROOT/f'data/processed/{sid}_power.parquet').reindex(idx).abs();f=pd.read_parquet(ROOT/f'data/processed/{sid}_flags.parquet').reindex(idx).eq(True);p=p.mask(f);m=mapping[mapping.apartment.eq(a)];t=phase_total(p,m)
        fp,week,seasons,days=fingerprint(t)
        events=pd.read_parquet(ROOT/f'data/processed/{sid}_events.parquet');events=events[events.start.ge(start)&events.end_exclusive.le(end+pd.Timedelta(minutes=5))]
        flexibility_events(events).to_csv(folder/'default_flexibility_event_screen.csv',index=False)
        for name,d in [('fingerprint',fp),('weekday_weekend',week),('season',seasons)]:d.to_csv(folder/f'{name}.csv',index=False)
        shape,n=load_shape(t);eligibility.append(dict(apartment=a,complete_profile_days=n,eligible=shape is not None))
        if shape is not None:profiles[a]=shape;shares[a]=appliance_shares(p,m)
        h=hourly_dataset(p,m,weather);panels[a]=h
        for target in ['total_kw','ac_kw']:
            model=weather_model(h,target=target);record={k:v for k,v in model.items() if k not in ['rows','coefficients','change_point_curve']};record.update(apartment=a,target=target)
            models.append(record)
            if model['ok']:
                model['rows'].rename_axis('Timestamp').reset_index().to_csv(folder/f'{target}_weather_model_hours.csv',index=False)
                model['coefficients'].to_csv(folder/f'{target}_coefficients.csv',index=False)
                model['change_point_curve'].to_csv(folder/f'{target}_change_point_selection.csv',index=False)
            (folder/f'{target}_inference.json').write_text(json_text(record))
        print(a,flush=True)
    cluster=cluster_apartments(profiles)
    if cluster['ok']:
        cluster['assignments'].to_csv(out/'cluster_assignments.csv',index=False);cluster['profiles'].to_csv(out/'cluster_normalized_profiles.csv',index=False)
    pd.DataFrame(eligibility).to_csv(out/'clustering_eligibility.csv',index=False)
    summary=[]
    for a in meta.apartment:
        r=matched_benchmark(a,meta,panels);folder=out/slug(a);r['audit'].to_csv(folder/'peer_matching_audit.csv',index=False)
        summary.append({'apartment':a,'available':r['ok'],'actual_expected_ratio':r.get('ratio',np.nan),'paired_hours':r.get('hours',0),'peers':'; '.join(r.get('peers',[])),'reason':r.get('reason','')})
        if r['ok']:r['rows'].rename_axis('Timestamp').reset_index().to_csv(folder/'matched_hourly_evidence.csv',index=False)
    pd.DataFrame(summary).to_csv(out/'matched_benchmark_summary.csv',index=False)
    pd.DataFrame([{k:v for k,v in m.items() if k not in ['features','reference_weather']} for m in models]).to_csv(out/'weather_model_summary.csv',index=False)
    (out/'scope.json').write_text(json_text(dict(start=str(start),end=str(end),timezone='Asia/Kolkata',review_filtered=True,coverage_threshold=.9,clustering='K-means mean-one complete daily load shapes; 14 days minimum',benchmark='area ±15%, identical BHK/orientation, floor ±4, >=2 peers and 168 simultaneous hours',indoor_status='No indoor time series supplied; import required',models=models)))
    write_review(out,meta)
    print(out,flush=True)
def write_review(out,meta):
    scope=json.loads((out/'scope.json').read_text());models=pd.read_csv(out/'weather_model_summary.csv');bench=pd.read_csv(out/'matched_benchmark_summary.csv')
    lines=['# Default-period evidence review','',f"Scope: {scope['start']} through {scope['end']} (Asia/Kolkata), review-filtered, 90% hourly/daily coverage threshold. Interactive filters recompute results.",'','Unsupported estimates must not be filled with synthetic scores.','']
    for a in meta.apartment:
        lines.extend(['## '+a,'']);fp=pd.read_csv(out/slug(a)/'fingerprint.csv')
        if len(fp):
            pk=fp.loc[fp.mean_kw.idxmax()];lines.append(f"- Fingerprint: typical hourly peak {int(pk.hour):02d}:00, {pk.mean_kw:.2f} kW. Use the exported eligible-hour counts.")
        else:lines.append('- Fingerprint: no eligible daily observations.')
        for r in models[models.apartment.eq(a)].itertuples():
            label='Apartment weather response' if r.target=='total_kw' else 'AC cooling response'
            if not r.ok:lines.append(f'- {label}: unavailable — {r.reason}')
            else:lines.append(f'- {label}: {"passes" if str(r.supported)=="True" else "fails"} the support screen; fitted base {r.base_c:.1f} °C, slope {r.slope:.4f} kW/°C, later-holdout R² {r.holdout_r2:.3f}.')
        r=bench[bench.apartment.eq(a)].iloc[0]
        lines.append(f"- Matched benchmark: Actual/Expected {r.actual_expected_ratio:.2f} on {int(r.paired_hours)} simultaneous hours. Peers: {r.peers}." if r.available else '- Matched benchmark: unavailable — '+str(r.reason))
        for r in models[models.apartment.eq(a)].itertuples():
            if str(getattr(r,'change_point_at_search_boundary',False))=='True':lines.append(f'- {r.target}: selected change point reaches the search boundary; threshold location is uncertain and slope intervals are conditional.')
        lines.append('- Indoor buffering/time constant/effectiveness/latent response: no indoor observations supplied; import matching room readings to calculate.')
        lines.append('')
    (out/'CLIENT_REVIEW.md').write_text('\n'.join(lines))
if __name__=='__main__':run()
