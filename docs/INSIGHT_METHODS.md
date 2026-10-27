# Insight methods and interpretation — version 2.0.0

All results recompute for apartment, duration, coverage and review settings. Cohort charts use the same duration for every apartment. A changed date range can change cluster membership, fitted weather reference, accepted models and eligible peers. Diagnostic regressions are implemented; future energy prediction remains unavailable.

## Electrical accounting

Source electrical readings are converted to **absolute values before averaging duplicates or integrating energy**. Weather and indoor measurements retain physical signs. Apartment power is **R + B + Y**. Appliance power sums **every remaining electrical column once**, including unknown, combined and mixed circuits. The workbook-confirmed A701 `Main Incomer` aliases its absent named R-phase channel.

Residual = phase total − appliance total at shared complete timestamps. This derived difference remains signed: an absolute residual would hide over-counting and violate the accounting identity. Positive residuals may indicate unmetered/unmapped loads; negative residuals require reconciliation. Neither establishes avoidable standby. Missing appliance readings cannot be residual load.

Native 15-minute power uses explicitly flagged interval holds on the 5-minute integration grid. Coverage audits count original source readings separately. Model/schedule eligibility uses available interval support; this must not be cited as additional measured sampling resolution. Hourly power requires the selected minimum share of its 12 five-minute slots. It is a mean kW over observed support, not a fabricated missing-hour energy total.

## View coverage

| View | Implemented evidence and limits |
|---|---|
| Data coverage | Monthly, daily, hourly and channel missingness; complete phase and balance coverage. Common requested dates include pre-installation gaps. |
| Apartment energy fingerprint | Hourly mean, P25/P75, weekday/weekend and season curves on eligible days; typical peak timing, level ratio and common-timestamp appliance shares. Curves use hourly means and an explicit daily coverage threshold. |
| Lifestyle / behavior clustering | K-means of mean-one daily shapes. Each profile requires 14 complete 24-hour daily shapes; each day receives equal weight. Explore K=2…4 with silhouette, forbid singleton clusters, fixed seed. Shape features are divided by sqrt(24). Optional appliance shares form an equally weighted second block; unavailable shares exclude that apartment. No inference of resident identity, occupancy or lifestyle cause. Small 12-apartment sample makes groups exploratory. |
| Weather-normalized consumption | Per-apartment cooling hinge regression with moisture and calendar terms. Later holdout validation, calendar-only baseline and day-cluster conditional coefficient intervals. Weather reference is the training-period median within current filters. |
| Cooling sensitivity | Pure AC channels only. Apartment-specific change point, kW/°C slope (equivalent to kWh/CDH), conditional interval and acceptance screen. AC kWh/CDH scatter uses only complete 24-hour paired days. Enthalpy/RH are descriptive secondary views. Cross-apartment comparison includes supported models only and exports exclusions. |
| Indoor–outdoor buffering | Room-specific Tout−Tin, AC-off first-difference lag and continuous AC-off daytime/overnight rates. No isolated envelope-efficiency claim. |
| Thermal time constant | First-order room model conditional on Tin, Tout and selected room AC; physical-sign, interval and later-holdout checks. No accepted time constant if checks fail. |
| AC cooling effectiveness | Complete isolated AC events with indoor/outdoor context, before/during/after response, raw °C/kWh and event regression adjusted for outdoor T change, RH, initial Tout−Tin and duration when enough independent events exist. This is not COP. |
| Humidity / latent response | RH, dew point and humidity-ratio responses on the same isolated AC events, plus moisture-change-per-event evidence. No latent capacity or moisture-removal mass without airflow/condensate measurements. |
| Flexible load / demand response | Complete geyser/washing-machine events and optionally AC, durations/start windows, user-defined peak/window/delay scenario, event-level candidate starts. Timing-screen score = 100 × candidate event energy / all complete event energy in the selected categories. Comfort and actual controllability remain unverified. |
| Matched-apartment benchmark | Same BHK/orientation, area tolerance default ±15%, floor difference ≤4; at least two peers and 168 simultaneously valid target/peer/weather hours. Area-scaled hourly peer median gives expected demand. Same time and common weather station control weather/calendar by matching; household size, behavior and equipment remain uncontrolled. |

The duplicate “Tab 1” in the brief is represented as separate fingerprint and clustering navigation views. Coverage remains the first navigation item. The original detailed energy/state/standby charts remain in `legacy_dashboard.py` and use the same input datasets and accounting rules.

## Weather models

For hourly load y:

`y = intercept + hour harmonics (sin/cos at 24 h and 12 h) + weekend + b × max(Tout − Tb, 0) + moisture + available optional weather fields`.

T and RH come from the same station observation held backward no more than 60 minutes; future weather observations are never joined. RH is default; enthalpy may replace it in the total-consumption view, rather than adding collinear moisture representations together. Available complete, varying irradiance, wind speed, rain and pressure can enter the model. Missing station fields are not imputed. Wind direction stays in the source dataset; it is not entered as a linear 0–360-degree regressor. Optional irradiance, wind direction and other station fields retain their templates/placeholders.

Requirements: ≥240 paired hours over ≥30 distinct days, temperature span ≥4 °C and nonconstant demand. Chronological whole dates split 80% training / 20% later holdout. Within training, the first 75% of dates fit candidate change points (18–32.5 °C at 0.5 °C increments); the last 25% choose the minimum-MSE candidate. Both sides of a candidate require ≥24 training hours. Final coefficients fit all training hours only. No holdout observations select the change point or fit coefficients. A selected search-boundary change point is explicitly flagged as not well localized; its slope interval remains conditional on that threshold.

A model is supported only if later R² > 0, MAE improves on the calendar-only baseline, the lower conditional 95% interval on cooling slope is >0, and fewer than 5% of holdout predictions are negative. Candidate models that fail remain diagnostic and explicitly labelled. These are practical screens, not validation against a national energy-rating standard. A high fitted training correlation cannot override a failed holdout. No supported model is forced when data are insufficient.

Adjusted demand = observed − modeled current-weather demand + modeled reference-weather demand. Calendar terms are preserved. Displayed daily curves average available paired hours and preserve null days; their means must not be described as complete daily kWh. Reference weather and paired-hour rows are exported. No causal conclusion is made from weather association. Confidence intervals cluster errors by date and are conditional on the chosen change point; they do not include change-point-selection uncertainty or unobserved confounding.

## Indoor alignment and methods

No indoor time series is included in the recovered source project. The app accepts `Timestamp,Apartment,Room,temperature,humidity` CSV, or preprocessing reads `data/raw/indoor.csv`. Invalid temperature/RH ranges become null. Negative Celsius temperatures are retained. Duplicate room/time records are averaged. Select the room and its actual serving AC explicitly; never substitute whole-apartment AC for another room's sensor.

Indoor values aggregate to 15-minute bins without interpolation. AC power requires three valid 5-minute supported slots. Outdoor observation age is exported. Tin-next response is known only for adjacent valid bins. Psychrometrics use the existing Magnus saturation approximation and 101.325 kPa reference pressure; resulting dew point and moisture metrics are diagnostic approximations, not a calibrated station-pressure calculation.

Buffer lag scans 0–6 h in 15-minute steps on first differences; all relevant intervals must be AC-off, with ≥48 paired transitions. A maximum correlation ≤0.2 is not called an identified lag. Daytime rise uses 10:00–16:59, overnight uses 22:00–05:59 and assigns after-midnight points to the preceding night. Rate windows require ≥12 observations over ≥3 h, and no intervening gap >15 min. Rates remain signed; overnight warming is not reported as cooling.

Thermal model: `Tin(t+15m) − Tin(t) = c + beta × (Tout − Tin) + gamma × AC kW`. Apparent `tau = −0.25/log(1−beta)` hours. Require ≥200 transitions over seven days including ≥30 AC-off hours and enough independent gap/AC variation. The first 80% of transitions fit; later 20% validate. A reported accepted tau requires the whole conditional 95% beta interval inside (0,1), negative gamma interval and positive holdout R². Infiltration, gains and ventilation remain bundled into this apparent response. This is not a calibrated building-envelope time constant.

AC event analysis uses complete uncensored 30–360 minute inferred electrical events, 30 minutes of AC-off context before and after, and complete paired indoor/weather data. Energy is summed on the aligned event bins. The timeline mean has different contributors at later offsets; counts are exported. Effectiveness adjustment requires ≥20 independent complete events and full predictor rank. Its day-cluster interval is conditional; no randomization or counterfactual validation is implied. RH change must be interpreted with dew point and humidity ratio because cooling itself can raise RH.

## Flexibility and benchmark interpretation

Candidate starts must fit the whole detected event inside the chosen operating window and maximum delay. Default peak = 18:00–22:00, target = 22:00–24:00, maximum delay = six hours. AC is excluded by default and is restricted to ≤15 minutes deferral if explicitly included. The candidate schedule is a resident-discussion scenario, not an automatic control instruction or a guarantee of comfort. Storage temperature, hot-water needs, appliance restart rules and permissions need separate verification. No peak kW or bill savings are inferred from shifted event energy.

Peer metadata is checked explicitly. Missing BHK, orientation, area or floor is a failed match. Every accepted hour requires target demand, outdoor T/RH and at least two metadata-matched peer demands. Expected kW is the median of their area-scaled kW for that exact hour. Actual/Expected uses sums on those same hours, not the average of hourly ratios. No whole-period extrapolation is made from the shared subset. Counts and exclusions are downloadable.

## Reproduction and primary references

`python -m src.preprocess` rebuilds power/states; `python -m src.export_schedules` rebuilds the original exports; `python -m src.export_insights` rebuilds the new default-duration model, fingerprint, clustering and benchmark evidence. Interactive filtering recomputes these functions rather than reading fixed model scores.

Primary technical background (the project-specific estimator/screening choices above are our implementation choices):

- [scikit-learn K-means and silhouette selection](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html).
- [EnergyPlus zone air energy balance](https://energyplus.readthedocs.io/en/stable/guides/engineering-reference/2.1-basis-for-the-zone-and-air-system-integration.html), background for first-order thermal-response interpretation.
- [PsychroLib psychrometric API](https://psychrometrics.github.io/psychrolib/api_docs.html), definitions of dew point, humidity ratio and moist-air enthalpy. The project uses its existing diagnostic Magnus approximation rather than a dependency on PsychroLib.
