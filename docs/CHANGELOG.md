# Project updates

## 2.0.0 — 27 October 2026

Rebuilt the primary app around filter-specific insight views: energy fingerprints, K-means behavior groups, weather-normalized load, apartment AC change points and cooling sensitivity, indoor buffering and apparent thermal time constants, AC effectiveness and latent response, flexible-event screening and matched-apartment Actual/Expected benchmarks. Each graph has a computed inference, data and method downloads.

Coverage remains the opening view; original phase/appliance/residual definitions and absolute source inputs are retained. Models use chronological validation, failed-result explanations and exports of eligibility/exclusions. No indoor sensor readings are invented. Room-specific CSV import and serving-AC selection activate the indoor calculations.

Added reproducible insight exports, synthetic estimator tests and updated interaction tests; retained the original detailed dashboard as legacy_dashboard.py. Updated all setup/method documentation and GitHub checks.

## 1.1.0 — 4 October 2026

The apartment R, B and Y channels represent phase power. Every other electrical column represents a submetered appliance/circuit. Apartment demand is the sum of phases, and submetered demand sums all remaining electrical channels once, including unnamed and mixed circuits.

Signed residual power is the phase total minus the appliance total on concurrent complete readings. Positive residuals may represent loads outside Schneider EMD appliance metering or mapping. Negative residuals stay signed for reconciliation. Missing appliance readings do not become residual load.

Added monthly phase/appliance/residual energy, daily/hourly residual patterns, cross-apartment residual comparison, 5-minute balance downloads, integrated 15-minute balance fields, and residual columns in the management export. Updated the calculation library, channel roles, prepared data, whitepaper evidence, exports, tests and method/setup documentation.

A701 retains its workbook-confirmed Main Incomer alias for the missing named R-phase channel. The alias does not create a fourth phase. Residual load remains distinct from candidate appliance standby and verified avoidable savings.
