# Contributing to Blop

Thanks for contributing to Blop. We aim to keep the process lightweight, practical, and focused on correct,
tested changes.

## Process

Open an issue before opening a pull request, unless the change is trivial, such as a typo fix, broken link fix,
or small documentation cleanup.

After there is an issue, open a small, focused pull request that addresses one topic at a time.

## Pull Requests

Good pull requests are:

- Small and focused
- Linked to an issue when the change is not trivial
- Covered by tests
- Passing CI
- Clear about what changed and why

Tests are required for code changes. Bug fixes should include a regression test when practical.

Unit tests should remain fast. The full unit test suite, run with `pixi run tests`, should stay under 10 seconds.

Integration tests, documentation tests, and tutorials may take longer, but should still run in a reasonable time.

## Local Checks

Before opening a pull request, please run:

```bash
pixi run -e dev-cpu check
pixi run -e dev-cpu tests
```

## Public API Changes

Changes to public APIs should be discussed lightly with reviewers before merge. This includes adding, removing,
renaming, or changing behavior of public classes, functions, arguments, or documented workflows.

## Optimization Behavior Changes

Changes that affect optimization behavior need clear evidence that the new behavior is better. Include benchmarks,
comparisons, or examples that explain the improvement.

## Review and Merge Policy

Maintainers should approve and merge readily when a pull request is small, correct, tested, and passes CI.

After approval, maintainers are responsible for merging.
