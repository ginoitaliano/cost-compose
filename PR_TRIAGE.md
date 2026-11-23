# Open pull request triage

Reviewed 2026-11-23 against `main` at `20268bd`. Nothing has been merged.

All three are Dependabot updates to GitHub Actions versions, opened 2026-11-02 from the `github-actions` entry in `.github/dependabot.yml`. Each changes only `.github/workflows/ci.yml`, and each has all five CI jobs passing (lint, test on 3.10 / 3.11 / 3.12, and `full`, which covers LangGraph parity, Redis and the dry run). None touches code, configs or experimental values.

| PR | Change | Decision | Reason |
| --- | --- | --- | --- |
| #1 | `actions/checkout` v4 -> v7 in all three jobs | Merge | CI green on every job. The workflow uses only default checkout options and GitHub-hosted runners. |
| #2 | `actions/setup-python` v5 -> v7 in all three jobs | Merge | CI green on every job, including the 3.10 to 3.12 matrix and the `cache: pip` option the workflow relies on. |
| #3 | `actions/upload-artifact` v4 -> v7 in the `full` job | Merge | CI green. The step uploads one directory under a fixed name, and the `full` job uploaded the dry-run artifact successfully. |

Notes for merging:

- #1 and #2 edit adjacent lines in the same steps (`checkout` is directly above `setup-python`). After one is merged, the other will likely show a conflict. Comment `@dependabot rebase` on the second one rather than resolving it by hand, then merge once CI is green again.
- The branch for the repo cleanup (`repo-cleanup-paper-alignment`) does not modify `.github/workflows/ci.yml`, so it does not conflict with any of the three.
