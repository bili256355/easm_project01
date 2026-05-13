# Same-family conditioning sensitivity status

This V2_b audit reads the completed V2_a outputs. It does not rerun PCMCI+.

Therefore, it **does not** test the alternative design in which same-family variables are excluded from the conditioning pool. That test is a separate computational experiment, because it changes the PCMCI+ search/conditioning semantics.

Current V2_a design:

```text
Reported source-target edges: cross-family only
Conditioning pool: all variables, including same-family variables
```

Why this matters:

```text
Allowing same-family controls is statistically cleaner but may partial out object-internal signals.
Excluding same-family controls may recover broader object-level signals but increases redundancy/confounding risk.
```

If this audit shows that key V1 physical expectations disappear before or during graph selection, a next patch can add an explicit `v2_c_same_family_conditioning_sensitivity` rerun rather than silently changing V2_a.
