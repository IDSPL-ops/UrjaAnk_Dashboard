# Operation schedule export

Each individual CSV includes 96-slot weekday/weekend profiles for each available season and all seasons. Use `fraction` for normalized active load, `active_probability` for sampled electrical-activity availability, and `design_kw`, `inactive_kw` and `clipping_loss_kw` to reconstruct full measured typical power. States are inferred, not measured commands.

The representative CSV has household-equal mean/median/P25/P75. Missing appliances are excluded and contributor counts remain visible. `mean_kw` and `energy_preserving_kw` retain actual average power; one common rated power multiplied by the mean fraction need not reproduce it. P25–P75 is cross-household spread. Do not add historical AC kW as internal gains if the AC is also modelled as HVAC.

Residual exports include signed power/energy, positive/negative components, complete-data coverage and matched phase/appliance totals. A positive residual may represent unmetered or unmapped loads. It is not a measured standby load. Negative residuals are retained for reconciliation review.

`export_scope.json` records the cohort and reporting dates. Read the project's docs/METHODOLOGY.md before using these profiles for simulation or whitepaper claims.
