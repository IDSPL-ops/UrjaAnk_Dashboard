# Palava apartment energy insights — Streamlit project 2.0.0

This project provides an apartment- and duration-specific management dashboard with data-driven inference text beside every graph, downloadable graph datasets, model evidence and per-view evidence ZIPs. The new insight dashboard is `streamlit_app.py`. The previous detailed phase, appliance event and standby views remain in `legacy_dashboard.py`.

## Run locally

Use Python 3.12. Extract this archive and open a terminal **inside `urjaank_dashboard/`**:

```bash
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Prepared data is included. No preprocessing is needed for the first launch. Select **Apartment**, **Duration** and **Insight tab** in the sidebar. Coverage and review settings also recalculate results. Selected-apartment schedule downloads remain in the sidebar; the schedule view builds a household-equal cohort average.

## Upload to GitHub and show the client

1. Extract the ZIP. Upload the **contents of `urjaank_dashboard/`** as the repository root so `streamlit_app.py` and `requirements.txt` are at the top level.
2. Include `config/`, `src/`, `data/templates/` and `data/processed/` for a runnable app. Include `data/raw/` for reproducibility and `exports/` for ready-made evidence; these are not required merely to run prepared data.
3. In your chosen Streamlit hosting environment, select the repository, Python 3.12 and entrypoint `streamlit_app.py`. Install `requirements.txt`. The included Dockerfile is an alternative for an internal host.
4. Open the app, confirm the scope and show coverage first. Then select fingerprint, clustering or another insight. An unsupported result states the missing data/validation requirement rather than inventing a score.

This delivery contains the project folder. It has not been pushed to a repository or deployed. GitHub numerical checks and Streamlit application tests are included. No credentials or external API keys are needed for the supplied prepared datasets. The largest included individual files are below GitHub's 100 MiB per-file limit; extract the folder instead of committing the whole ZIP as one file.

## Insight navigation

The duplicate “Tab 1” in the request is separated into two named views. Missing-data coverage remains the first view.

| Navigation view | What the inference answers |
|---|---|
| 0 · Data coverage | How much of the requested analysis period is actually observed, monthly/daily/hourly and per sensor? |
| 1 · Apartment energy fingerprint | When does demand typically peak, how do weekday/weekend and seasonal shapes differ, and which monitored uses contribute most? |
| 2 · Energy lifestyle / behavior clustering | Which apartments have similar normalized shapes despite different size/demand levels? K-means and optional appliance-share features. |
| 3 · Weather-normalized consumption | Does weather add predictive information beyond time-of-day/weekend, and what would the observed load look like at reference weather? |
| 4 · Cooling sensitivity index | How fast does AC demand rise above this apartment's fitted change point, including RH/enthalpy evidence and cross-apartment comparison? |
| 5 · Indoor–outdoor thermal buffering | What are room temperature differences, AC-off lag and daytime/overnight rates? |
| 6 · Apartment thermal time constant | Is a physically plausible, validated first-order room thermal response identifiable? |
| 7 · AC cooling effectiveness | What temperature improvement accompanies observed AC energy, and can weather-adjusted response be estimated? |
| 8 · Humidity / latent cooling performance | Do absolute moisture and dew point fall around AC operation, even if RH behaves differently? |
| 9 · Flexible load / demand response | Which complete appliance events fit a proposed operating window and delay limit? |
| 10 · Matched-apartment benchmark | What is Actual/Expected demand relative to metadata-matched apartments during the same weather and hours? |
| Operation schedules & evidence | Selected/cohort 15-minute operation schedules and a clearly untrained future-prediction placeholder. |

Every graph includes a specific inference, CSV export, inference/method JSON and PNG camera control. The bottom ZIP contains the current view's chart data, inferences, scope and selected channel mapping. Models export coefficient/selection evidence. To collect all views for a whitepaper, select the required scope and download each view, or rebuild the default-scope exports below.

## Definitions retained throughout

- **Apartment total kW = R phase + B phase + Y phase.**
- **Appliance total kW = all remaining electrical channels**, including lights/fans, mixed circuits and unknown appliance names.
- **Input negative consumption readings become positive before averaging or energy integration.** Physical indoor/weather signs are retained.
- **Residual kW = apartment total − appliance total** at matching complete timestamps. The derived residual remains signed. A positive residual may represent unmetered/unmapped loads; a negative residual is a reconciliation issue, not additional consumption.
- A701 uses the workbook-confirmed `Main Incomer` R-phase alias; it is counted once.
- AC uses word-boundary naming, so “Washing Machine” cannot match “AC”. Mixed AC/light circuits do not enter pure-AC sensitivity.
- Lights/fans remain combined. Missing readings are unknown, never zero. Original supplied-reading coverage is distinct from marked native 15-minute interval support.

## Source availability and limits

The recovered source set contains **12 unique apartments**. The default operation-schedule cohort has the confirmed ten, excluding Elite A401 and Lakeside B101. Each available apartment remains selectable; no unprovided apartment is manufactured from the longer metadata list.

**No indoor temperature/RH time series was supplied in the saved project.** Views 5–8 contain working calculations and indoor import, but no real indoor result is claimed until matching data is supplied. Upload the template CSV in the sidebar for session analysis, or save it as `data/raw/indoor.csv` and rerun preprocessing for deployment. Select the indoor room and actual AC channel serving it. Indoor availability can differ by apartment and dates.

The station provides temperature and RH. Irradiance, wind direction/speed, rain and pressure remain explicit missing fields until an extension CSV is supplied. Weather models are diagnostic regressions, not future forecasts or proof of causation. Weak holdout validation, missing metadata or too few peers can legitimately make results unavailable. The matched benchmark exports every exclusion rather than silently relaxing matches. Flexibility is a timing screen requiring resident/equipment verification, not verified comfortable load shifting or savings.

Default-scope exports show that supported weather models and matched peers exist for some apartments, while many apartment/metric combinations fail validation or coverage checks. See `exports/insights/` for actual outcomes and the method document for thresholds. A useful client walkthrough is fingerprint/clustering on Lakeside K502, supported diagnostic examples on Urbano M501, and peer matching on Urabno Q606 or Lakeside A701. Always retain the displayed dates, coverage and validation caveats.

## Rebuild and verify

After changing raw data, indoor/weather inputs, mappings, review caps or thresholds:

```bash
python -m src.preprocess
python -m src.export_schedules
python -m src.export_insights
```

Restart Streamlit or clear its cache after rebuilding inputs. The app recomputes interactive scope; packaged insight CSVs document the default scope only.

```bash
python -m pip install -r requirements-dev.txt
python -m pytest tests -q
```

## Project folder structure

| Path | Purpose |
|---|---|
| `streamlit_app.py` | New insight dashboard and filters |
| `legacy_dashboard.py` | Retained detailed energy, state/event and standby dashboard |
| `src/analytics.py` | Common accounting, coverage, states, events and schedules |
| `src/insights.py` | Fingerprint, K-means, weather regression, cooling sensitivity, peer matching and flexibility |
| `src/indoor_insights.py` | Indoor import, room alignment, buffering, thermal response and AC/moisture event analysis |
| `src/ui.py` | Chart inference, CSV/JSON and ZIP exports |
| `src/preprocess.py` | Rebuild prepared datasets from sources |
| `src/export_schedules.py` | Rebuild schedules, coverage and residual exports |
| `src/export_insights.py` | Rebuild default-period fingerprint, model, cluster and benchmark evidence |
| `config/` | Apartment metadata, channel roles, review/threshold settings |
| `data/raw/` | Original electrical CSVs, weather and mapping workbook; optional indoor/weather extension |
| `data/processed/` | Prepared power, source coverage, hold flags, states, events, weather and manifests |
| `data/templates/` | Indoor, weather-extension and threshold-override input templates |
| `exports/` | Original schedules and accounting exports |
| `exports/insights/` | Reproducible new view evidence and model/peer eligibility summaries |
| `docs/INSIGHT_METHODS.md` | Formulas, assumptions, model screens, limitations and primary references |
| `docs/METHODOLOGY.md` | Original pipeline and schedule methodology |
| `docs/VALIDATION.md` | Verification performed and remaining limits |
| `docs/CHANGELOG.md` | Version history |
| `tests/` | Numerical regressions, synthetic estimator checks and Streamlit interaction tests |
| `.github/workflows/checks.yml` | Automated repository checks |
| `.streamlit/config.toml` | Theme and runtime settings |
| `requirements.txt`, `requirements-dev.txt`, `Dockerfile` | Runtime, development and hosting setup |
