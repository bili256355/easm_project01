# V3_b stability logger hotfix

## Purpose
Fixes a runtime error in `lead_lag_screen/V3/src/lead_lag_screen_v3b/pipeline_stability.py`:

```text
TypeError: setup_logger() got an unexpected keyword argument 'log_name'
```

## Change
The existing `lead_lag_screen_v3.logging_utils.setup_logger` accepts the logger name as the second positional argument, not a `log_name=` keyword. This hotfix changes:

```python
setup_logger(settings.log_dir, log_name="lead_lag_screen_v3_b_stability.log")
```

to:

```python
setup_logger(settings.log_dir, "lead_lag_screen_v3_b_stability")
```

## Scope
- Replacement-file hotfix only.
- No scientific logic changed.
- No output rules, bootstrap settings, stability judgement, or data paths changed.

## Files replaced
```text
lead_lag_screen/V3/src/lead_lag_screen_v3b/pipeline_stability.py
```

## Check performed
```text
python -m py_compile pipeline_stability.py
```
