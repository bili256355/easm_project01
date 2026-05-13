# UPDATE_LOG_V7_Z_MULTIWIN_A_HOTFIX_02

## Purpose
Fix the conservative multi-window runner so it does not require or auto-search an `accepted_windows` registry by default. The current project workflow uses the confirmed significant windows directly.

## Changes
- Default window source is now hardcoded, not registry/auto-search.
- Hardcoded windows:
  - W045: anchor 45, system 40-48
  - W081: anchor 81, system 75-87
  - W113: anchor 113, system 108-118
  - W160: anchor 160, system 155-165
- Default run mode remains W45 only and profile-only.
- External registry remains optional via `V7_MULTI_ACCEPTED_WINDOW_REGISTRY`.
- Auto-search is opt-in via `V7_MULTI_WINDOW_SOURCE=auto`.
- No scientific definitions, detector logic, baseline logic, or 2D metrics were changed.

## Why
The user does not use an accepted_windows table. The previous hotfix still attempted to locate a registry, which reintroduced the wrong workflow assumption.
