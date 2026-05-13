# UPDATE_LOG_V7_V_HOTFIX_01

## Change
Removed the unintended dependency on `stage_partition_v6.io.load_smoothed_fields` from V7-v.

## Reason
The user's V7 runtime raised `ModuleNotFoundError: No module named 'stage_partition_v6'` when importing the V7-v module. The V7-v branch should be self-contained and should not require importing older V6 modules.

## Implementation
Added an internal loader:

```python
def _load_smoothed_fields_npz(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}
```

and replaced:

```python
smoothed = load_smoothed_fields(smoothed_path)
```

with:

```python
smoothed = _load_smoothed_fields_npz(smoothed_path)
```

## Scope
No scientific logic, event definitions, baseline windows, or output directory semantics were changed.
