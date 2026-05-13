# V6-b selection-frequency experiment

This experiment layer is intentionally isolated from the current `V6-a` point-level screening chain.

## Purpose

Generate a bootstrap **selection-frequency-by-day** curve that answers:

- which day positions are repeatedly selected by the detector across bootstrap replicates
- where interval-level existence evidence is strong

## What this experiment includes

- the same baseline detector backbone used by `V6-a`
- bootstrap resampling
- raw day-by-day selection frequency
- lightly smoothed selection frequency
- optional local maxima extraction from the frequency curve

## What this experiment does NOT include

- window judgement or interval adjudication
- competition
- parameter-path
- final judgement
- replacement of the current `V6-a` point-level bootstrap main table

## Interpretation boundary

This layer provides **interval-level existence evidence** only.
It does not decide a unique breakpoint day and does not replace the current point-level bootstrap screening chain.


Update: this replacement raises the default bootstrap replicate count for the isolated selection-frequency experiment from 200 to 1000 and writes the selection-frequency curve PNG by default.
