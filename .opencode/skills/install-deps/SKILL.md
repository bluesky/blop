---
name: install-deps
description: Install and manage blop dependencies using pixi or pip
compatibility: opencode
---

## What I do

Install project dependencies.

## Commands

**Using pixi:**

```bash
pixi install
```

## Adding dependencies

Add to `pyproject.toml` as the single source of truth. `pixi.toml` should install most dependencies via the `pyproject.toml` with few exceptions.

## When to use me

Use me when setting up the project or syncing dependencies.
