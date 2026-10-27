> Version 2.0.0 adds the insight dashboard. Read [INSIGHT_METHODS.md](INSIGHT_METHODS.md) for the new model, clustering, thermal, flexibility and benchmarking definitions; this document describes the retained preprocessing and schedule methods.

# Measurement and inference definitions

## Sources and authority

The uploaded CSVs establish observed electrical/weather readings. The sensor workbook establishes phase/channel relationships. The user's metadata table establishes apartment area, floor, orientation and typology. The supplied Deliverable 1 and 2 documents establish the analytical aims; they do not establish new measured results. Source hashes are recorded in the prepared manifest. kW was explicitly confirmed by the user in this turn.

## Electrical energy

Negative electrical values are converted to absolute values before aggregating duplicate timestamps. Duplicate readings at one timestamp are averaged, never counted as separate intervals. For an interval represented at five-minute resolution, energy = absolute kW × 5/60 hours. No electrical sign conversion is applied to temperature or humidity.

A sustained 15-minute source cadence is inferred only when a day has at least 30 intervals, a median interval of 15 minutes and at least 70% of adjacent intervals exactly 15 minutes. Each native reading on those days represents a 15-minute interval using explicitly marked held support on the five-minute analytical grid. The original reading-only table is retained. Coverage counts original readings, not expanded values. The assumption that a sampled reading represents its native interval should be confirmed with the data provider before billing-grade use. Isolated gaps at ordinary five-minute cadence remain unfilled. Channel blanks at a native timestamp are not turned into measured values.

Apartment total power is the sum of R, B and Y phase kW, with all three required at the same timestamp. Every remaining electrical column is a submetered appliance/circuit reading. The appliance total includes the five named appliance types, combined lights/fans, mixed circuits and unnamed submetered electrical columns, whether or not their appliance/phase identity appears in the workbook. Timestamp and apartment-identity fields are not electrical channels.

`Residual kW = Total phase kW − Total appliance/circuit kW`

Positive residuals can represent loads that are not connected to the appliance Schneider EMD channels or are not mapped. That interpretation is a possible explanation, not identification of a particular appliance. Measurement timing, sensor/CT scaling and circuit overlap can also affect the difference. The residual is not labelled as appliance standby or verified avoidable consumption.

The residual is calculated only when all phase and all appliance/circuit readings are valid concurrently. A missing appliance reading never becomes zero. Phase-only apartment energy remains independently calculable where all phases are valid. Paired phase/appliance/residual energy totals use only common complete timestamps and cannot be compared directly to full phase-only energy without considering their different coverage.

Negative residuals remain signed and indicate that the appliance subtotal exceeds the phase total. They require reconciliation rather than conversion to positive consumption. Original electrical readings still use absolute values before aggregation, as requested. The derived difference is neither converted to absolute values nor clipped to zero. Exports retain separate positive/negative residual components, negative-interval counts, expected/valid intervals and coverage. Net residual can be near zero even where positive and negative intervals are substantial.

Daily/monthly residual energy = sum of signed residual kW × 5/60 hours on valid common intervals. Mean residual kW divides residual energy by valid common hours. Hourly residual patterns group valid intervals by hour of day. The paired identity `phase energy = appliance energy + signed residual energy` is checked in every exported period. An all-missing period retains null energy, not zero.

A701's `Main Incomer` is assigned to R phase in the supplied workbook and substitutes for the absent named R-phase column. This explicit alias is not counted as an appliance or whole-apartment meter in addition to R/B/Y. If a named R-phase column exists, Main Incomer is not promoted to a fourth phase. The Lagoona light/fan circuit repeated under multiple phases remains one electrical column; its mapping conflict is visible. An unmapped electrical column such as O1106's `P[5]` remains included once in the appliance total, rather than disappearing into residual merely because its name is unknown.

## Data coverage

Expected readings = requested five-minute timestamps × selected channel count. Missing readings = expected cells − non-null supplied cells, including completely absent rows. Zero is a valid reading. Usable = supplied readings − review-flagged supplied readings. Monthly and daily percentages sum numerator/denominator before division. Hourly means coverage across all dates at each hour of day. Reporting dates cover complete local days. The default excludes the final partial midnight-only day.

A selected reporting period can predate an apartment's monitoring installation. Those intervals are missing in the common study-period denominator. The source audit also provides coverage within each file's monitoring span. The reduced native cadence of O1106 is shown separately so a 15-minute source is not mistaken for random five-minute sensor loss.

Review flags use editable category-specific power caps in config/settings.json. These are screening rules rather than confirmed physical nameplates. Original readings remain downloadable/reproducible. Default energy analytics exclude flagged values, with an option to include them. State and schedule calculations always use review-filtered data, including when the energy display includes all recorded readings.

## State and event derivation

A deterministic two-cluster fit of the cleaned channel distribution establishes a low/high operating signature, subject to a category-specific noise floor. Separate ON and OFF thresholds provide hysteresis. A missing sample resets state to unknown; events cannot bridge missing intervals. Reviewed thresholds and design powers can override inferred values. Stable all-zero or very low-load channels are marked unresolved rather than declared unused with high confidence.

Electrical ON is a proxy for sampled power activity. Inverter operation, fan-only modes, thermostat cycles and multi-stage washing-machine activity prevent equating events with user switch commands. Refrigerator active state describes compressor-like electrical draw. User sessions and internal appliance cycles are not independently identified. Sustained 15-minute sampling reduces event precision even though the analytical grid is five minutes.

Events carry start/end, duration, energy, left/right censoring and single-sample flags. Events crossing the reporting boundary are excluded from complete-event duration/start summaries. Duty-cycle denominator is valid state hours. Coincidence requires valid states for all monitored appliance channels. The activity indicator is the valid-time share with a washing machine, geyser or oven active; it is not a validated occupancy probability.

## Operating schedules

Electrical data and states are averaged into 15-minute bins. A bin needs three valid represented five-minute intervals. Profiles have 96 slots per weekday/weekend and season. Month groups: Summer March–May; Monsoon June–September; Mild October–February. All-seasons profiles are also available, but seasonal AC profiles should be preferred for simulation.

The design reference is P99.5 of cleaned inferred-active power, bounded by the ON threshold. It is not a verified nameplate. Category design power is the sum of separate device references in that apartment. Active fraction = mean active category kW / category design kW, clipped to 0–1. Inactive kW is retained separately. Over-design power is exported in `clipping_loss_kw` and is required for full energy reconstruction:

`represented kW = design kW × fraction + inactive kW + clipping_loss_kw`

The validation table checks reconstruction. A reconstruction match verifies accounting only, not state-classification accuracy or the applicability of inferred design power. Full-day sums require all 96 slots. Missing slots never become zero.

Representative profiles average normalized apartment profiles with **equal household weight**, regardless of monitoring length or number of AC channels. Mean/median/P25/P75 are cross-apartment statistics, not pooled five-minute statistics. Contributor counts accompany every slot. If only nine households have an oven channel, the oven average is explicitly a nine-contributor estimate in the selected ten-flat cohort. Fractional average schedules describe annual/internal-gain patterns; they are not discrete start/stop schedules for peak modelling.

Different households have different design powers. `mean fraction × mean design power` need not reproduce mean household kW. The representative export includes mean active/inactive/total kW and an effective slot design reference for reconstruction. For physical HVAC simulation, use measured electrical-active probability as an availability proxy and calibrate the physical AC/control model. Do not input historical AC electrical kW as an internal gain while also modelling the same AC system.

## Standby and low-load measures

Low-power inactive residuals above 1 W on AC, washing machine and oven channels are candidate standby. They can include another connected load or meter offset. Candidate energy sums those valid intervals. Refrigerator cycles and geyser heating/thermal maintenance are excluded from vampire estimates. Avoidable standby savings remain unavailable until field verification; no savings factor is invented.

Annualized candidate kWh = observed candidate energy / valid day-equivalents × 365. This is an extrapolation scenario and can be unrepresentative of seasonality. The observed selected-period energy remains primary. Apartment P5 load includes necessary refrigeration/other services and is a low-load indicator, not a standby total.

## Weather and indoor data

Weather dates are day-first. Native hourly coverage counts hourly bins with temperature and humidity. Daily paired weather-energy comparisons require the selected coverage threshold for AC and weather. Pearson correlation requires at least ten variable paired days; it is descriptive, not causal. Cooling-degree-hours use 24°C as an uncalibrated diagnostic base and are included in exported weather pairs.

The integrated 15-minute dataset uses an as-of backward weather join with at most 60 minutes of hold and an explicit age column. Long gaps remain missing. Dew point, humidity ratio and enthalpy are descriptive calculations at standard atmospheric pressure. They do not constitute full comfort assessment.

No indoor time series was present in the supplied files. Optional per-apartment, per-room indoor data is supported; other apartments remain unavailable. Thermal time constants, cooling effectiveness and full comfort indices need additional data and model validation and are not fabricated.

## Cross-apartment comparison

Recorded daily energy divided by area is calculated only for days meeting the complete-phase coverage threshold. At least ten eligible days are required to plot an apartment. Coverage-filtered values are still recorded interval energy, not gap-filled daily totals. Broad comparisons can have different eligible dates; that limitation is shown. Matched pair comparisons require the same flat type, area within 10%, and at least ten identical valid dates. Household behaviour and equipment remain uncontrolled. Floor/orientation causality and Actual/Expected scores require validated prediction or physics models.

## Source issues retained for audit

- 12 unique apartment files, not ten. Default ten-apartment cohort explicitly selected by the user; the other two remain selectable.
- Two K502 files were byte-identical; one copy is excluded.
- Elite A401 has one Unix-epoch timestamp (1 January 1970). It is quarantined by the explicit May 2025–May 2026 electrical monitoring window. Raw data is preserved and the excluded-row count is retained.
- O1106 uses `DD-MM-YYYY HH:MM`, whereas most apartment files use ISO timestamps. Dates are parsed using mixed-format day-first handling.
- O1106 later changes to sustained 15-minute sampling. Original readings and analytical held-support flags remain separate.
- The O1106 column `P[5]` has no clear workbook mapping; it remains Other and the exception is disclosed.
- Lagoona has a mixed AC/washroom-light circuit and a repeated light/fan mapping under several phases.
- Indoor T/RH is unavailable in this uploaded source set.
- Marvella building type, flat type and orientation were not supplied; they remain blank and it is excluded from typology matching.
- Solar radiation and wind/weather extensions are placeholders. No synthetic values are inserted.

## Prediction placeholder

No forecast, expected-energy score or model-validation accuracy is displayed. Planned targets, a global gradient-boosting starting model, time-based splits and leakage controls are documented in the app and README. This remains an explicit placeholder as requested.
