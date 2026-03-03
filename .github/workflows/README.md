# GitHub Actions CI Overview

## Workflow

### `ci.yml`

This controls how the CI jobs are run and `ci.yml` is only run if one of the following conditions are met:

- Pushes
- Pull requests
- Manual dispatch
- Daily at 6:00 AM UTC

If one of these conditions are met, than the following occurs:

- **`check`**: Always runs and determines if the branch is part of a PR.
- **`lint`, `test`, `docs`**: Run only when the branch is **not** already in an open PR — to avoid duplicate checks.
- **`pypi`**: Publishes to PyPI if all jobs are successful and if the event is "release"
- **`scheduled-job`**: Runs `_testing.yml` on a specific schedule.

This structure avoids duplicate CI jobs when both a PR and a branch push happen.

### `_check.yml`

Used to see if the current branch (on `push`) is already part of an open pull request.

- If the branch **is part of a PR**, it outputs the PR number.
- If **not**, it outputs an empty value.
- This output is used to **conditionally skip jobs** in `ci.yml` when they would otherwise run redundantly.

## Notes & Tips

- If you push a branch before opening a PR, `branch-pr` will initially be empty. Once the PR is created, future pushes to that branch will correctly populate `branch-pr`.
- `check` must always run before the conditional jobs (`lint`, `test`, `docs`) to provide the correct `branch-pr` context.
