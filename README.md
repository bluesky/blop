# Blop - Bluesky Optimization Package

> [!IMPORTANT]
> **Blop is now [Bluesky Adaptive](https://github.com/bluesky/bluesky-adaptive).**
> Blop v1.2.0 is the final release, and this repository is archived.
> bluesky-adaptive 1.0.0 is Blop 1.2.0 under a new name; development continues there.
> Please open new issues and pull requests in
[bluesky/bluesky-adaptive](https://github.com/bluesky/bluesky-adaptive).

## Migrating to Bluesky Adaptive

| Blop | Bluesky Adaptive |
|---|---|
| `pip install blop[...]` | `pip install bluesky-adaptive[...]` |
| `conda install -c conda-forge blop` | `conda install -c conda-forge bluesky-adaptive` |
| `import blop`, `from blop.ax import Agent` | `import bluesky_adaptive`, `from bluesky_adaptive.ax import Agent` |
| <https://blueskyproject.io/blop> | <https://blueskyproject.io/bluesky-adaptive> |

Extra names (`ax`, `queueserver`, `scipy`, `xopt`, `all`, `cpu`, `dev`) and the run 
metadata keys (`blop_suggestions`, `blop_acquisition_order`, `blop_correlation_uid`) are unchanged, 
so data recorded with Blop works with Bluesky Adaptive. See 
the [1.0.0 release notes](https://blueskyproject.io/bluesky-adaptive/v1.0.0/release-history.html).

To keep using Blop, pin `blop==1.2.0`.

## Citation

If you use this package in your work, please cite the following paper:

```bibtex
@Article{Morris2024,
  author   = {Morris, Thomas W. and Rakitin, Max and Du, Yonghua and Fedurin, Mikhail and Giles, Abigail C. and Leshchev, Denis and Li, William H. and Romasky, Brianna and Stavitski, Eli and Walter, Andrew L. and Moeller, Paul and Nash, Boaz and Islegen-Wojdyla, Antoine},
  journal  = {Journal of Synchrotron Radiation},
  title    = {{A general Bayesian algorithm for the autonomous alignment of beamlines}},
  year     = {2024},
  month    = {Nov},
  number   = {6},
  pages    = {1446--1456},
  volume   = {31},
  doi      = {10.1107/S1600577524008993},
  keywords = {Bayesian optimization, automated alignment, synchrotron radiation, digital twins, machine learning},
  url      = {https://doi.org/10.1107/S1600577524008993},
}
```
