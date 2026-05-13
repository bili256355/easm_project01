# UPDATE_LOG_V7_Z_JE_AUDIT_A

## V7-z-je-audit-a: Je layer-split audit for W45

Purpose: audit why Je has an early shape-pattern peak around day33 but a late raw/profile peak around day46 in the W45 multi-object validation results.

Scope:
- Adds a new standalone entry script and module.
- Does not modify V7-z main pipeline, evidence gate, detector logic, bootstrap logic, or outputs.
- Focuses only on Je over day0–70.

Main checks:
- Rebuild Je raw 2-degree profile from u200 over 120–150E, 25–45N.
- Build shape-normalized Je profile.
- Audit shape-normalization norm around day33.
- Compare raw/profile and shape-pattern detector scores from V7-z when available.
- Compute Je feature time series: strength, amplitude, norm, axis latitude, centroid latitude, spread, north-south contrast, skewness.
- Compare day33 and day46 before/after raw and shape profiles.
- Read V7-z bootstrap return-day distribution to summarize raw-vs-shape peak separation when available.
- Write a markdown summary with allowed and forbidden interpretations.
