"""Create downloadable study-period schedules: python -m src.export_schedules."""
import json,zipfile
from pathlib import Path
import pandas as pd
from .analytics import *
ROOT=Path(__file__).resolve().parents[1]

def run():
    out=ROOT/'exports';out.mkdir(exist_ok=True);meta=pd.read_csv(ROOT/'config/apartments.csv');mapping=pd.read_csv(ROOT/'config/channels.csv');th=pd.read_csv(ROOT/'data/processed/thresholds.csv');qa=pd.read_csv(ROOT/'data/processed/source_qa.csv')
    start=pd.to_datetime(qa.start).min().normalize();end=pd.to_datetime(qa.end).max().normalize()-pd.Timedelta(minutes=5)
    schedules=[];management=[];totals=[];balances=[]
    for _,a in meta.iterrows():
        sid=slug(a.apartment);p=pd.read_parquet(ROOT/f'data/processed/{sid}_power.parquet');f=pd.read_parquet(ROOT/f'data/processed/{sid}_flags.parquet');s=pd.read_parquet(ROOT/f'data/processed/{sid}_states.parquet')
        cp=p.mask(f).loc[start:end];ss=s.loc[start:end];m=mapping[mapping.apartment==a.apartment]
        schedule=schedule_from_power(cp,ss,m,th[th.apartment==a.apartment],a.apartment)
        schedule.to_csv(out/f'{sid}_operation_schedule_15min.csv',index=False);schedules.append(schedule)
        source=pd.read_parquet(ROOT/f'data/processed/{sid}_source.parquet')
        for scope in ['monthly','daily','hourly']:
            d=coverage(source,f,start,end,scope);d['apartment']=a.apartment;d['scope']=scope;management.append(d)
        pt=phase_total(cp,m);dt=daily_total(cp,m);balance=electrical_balance(cp.reindex(pd.date_range(start,end,freq='5min')),m)
        balance.rename_axis('Timestamp').reset_index().to_csv(out/f'{sid}_residual_load_5min.csv',index=False)
        for scope in ['monthly','daily','hourly']:
            bd=balance_summary(balance,scope);bd['apartment']=a.apartment;bd['scope']=scope;balances.append(bd)
        totals.append(dict(apartment=a.apartment,recorded_phase_kwh=pt.sum(min_count=1)*DT_HOURS,eligible_days=int(dt.eligible.sum()),mean_eligible_day_kwh=dt.eligible_recorded_kwh.mean(),valid_phase_hours=pt.notna().sum()*DT_HOURS,paired_phase_kwh=balance.phase_kw_aligned.sum(min_count=1)*DT_HOURS,paired_appliance_kwh=balance.appliance_kw_aligned.sum(min_count=1)*DT_HOURS,residual_kwh=balance.residual_kw.sum(min_count=1)*DT_HOURS,positive_residual_kwh=balance.positive_residual_kw.sum(min_count=1)*DT_HOURS,negative_residual_kwh=balance.negative_residual_kw.sum(min_count=1)*DT_HOURS,valid_balance_hours=balance.valid_balance.sum()*DT_HOURS,balance_coverage_pct=100*balance.valid_balance.sum()/len(balance),negative_residual_slots=int(balance.negative_residual_flag.sum())))
    all_schedules=pd.concat(schedules,ignore_index=True);all_schedules.to_csv(out/'all_apartment_operation_schedules.csv',index=False)
    cohort=meta.loc[meta.default_cohort.astype(str).str.lower().eq('true'),'apartment'].tolist()
    rep=representative_schedule(all_schedules,cohort);rep.to_csv(out/'representative_10_apartment_schedule.csv',index=False)
    pd.concat(management,ignore_index=True).to_csv(out/'all_apartment_monthly_daily_hourly_missing_data.csv',index=False)
    pd.concat(balances,ignore_index=True).to_csv(out/'all_apartment_residual_monthly_daily_hourly.csv',index=False)
    pd.DataFrame(totals).to_csv(out/'management_energy_summary.csv',index=False)
    qa.to_csv(out/'source_data_audit.csv',index=False)
    (out/'export_scope.json').write_text(json.dumps(dict(start=str(start),end=str(end),timezone='Asia/Kolkata',cohort=cohort,apartment_count=len(meta),note='Residual = R+B+Y minus all non-phase electrical channels. Paired phase/appliance/residual energy uses the same complete timestamps; negative residual remains signed. Review-filtered calculations. Native 15-minute intervals may have marked held support. Original reading counts control missing-data audit. Candidate standby needs field verification.'),indent=2))
    standby=pd.read_csv(ROOT/'data/processed/standby.csv');standby.to_csv(out/'candidate_standby_load_library.csv',index=False)
    (out/'SCHEDULE_README.md').write_text('''# Operation schedule export\n\nEach individual CSV includes 96-slot weekday/weekend profiles for each available season and all seasons. Use `fraction` for normalized active load, `active_probability` for sampled electrical-activity availability, and `design_kw`, `inactive_kw` and `clipping_loss_kw` to reconstruct full measured typical power. States are inferred, not measured commands.\n\nThe representative CSV has household-equal mean/median/P25/P75. Missing appliances are excluded and contributor counts remain visible. `mean_kw` and `energy_preserving_kw` retain actual average power; one common rated power multiplied by the mean fraction need not reproduce it. P25–P75 is cross-household spread. Do not add historical AC kW as internal gains if the AC is also modelled as HVAC.\n\nResidual exports include signed power/energy, positive/negative components, complete-data coverage and matched phase/appliance totals. A positive residual may represent unmetered or unmapped loads. It is not a measured standby load. Negative residuals are retained for reconciliation review.\n\n`export_scope.json` records the cohort and reporting dates. Read the project's docs/METHODOLOGY.md before using these profiles for simulation or whitepaper claims.\n''')
    return out
if __name__=='__main__':print(run())
