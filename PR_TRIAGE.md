# Pull request triage

Reviewed 2026-11-23 against `main` at `20268bd`.

| PR | Change | Recommendation | Outcome |
| --- | --- | --- | --- |
| #1 | `actions/checkout` v4 -> v7 | Merge | Merged 2026-11-23 after `@dependabot recreate` (it conflicted with #2 on adjacent lines) |
| #2 | `actions/setup-python` v5 -> v7 | Merge | Merged 2026-11-23 after a Dependabot rebase onto the new `main` |
| #3 | `actions/upload-artifact` v4 -> v7 | Merge | Merged 2026-11-23 |

All three were Dependabot updates touching only `.github/workflows/ci.yml`, with all five CI jobs green before merging. `ci.yml` now uses checkout v7, setup-python v7 and upload-artifact v7.

Note for future Dependabot PRs on this file: updates to `checkout` and `setup-python` edit adjacent lines in every job, so merging one makes the other conflict. Use `@dependabot recreate` on the second one rather than `rebase`, which replays the old commit and stops at the conflict.
