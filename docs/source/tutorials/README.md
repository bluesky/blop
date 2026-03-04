# How to run and contribute to the tutorials

This repository contains tutorials that are designed to be run using Jupyter Lab, following what was done in [executable tutorials](https://github.com/scientific-python/executable-tutorials) on Github. This guide provides instructions on how to run the tutorials as well as what to do if a new tutorial is added.

## Running the Tutorials

To run and experiment with the tutorial code, you'll need to open them inside Jupyter Lab. To do this, run:

```bash
pixi run start-jupyter
```

This will launch Jupyter Lab in your browser, where you can navigate to and run the tutorial notebooks.

If you prefer to work with the tutorials in your local editor as standard notebooks, you can convert them back to the .ipynb format:

```bash
jupytext --to notebook <path_to_tutorial>
```

## Adding a tutorial

If you create a new tutorial, you must convert it to an .md file format. You can do this using the below command:

```bash
jupytext --to myst <path_to_tutorial>
```

To than confirm that your tutorial is still working as expected, you can run the following:

```bash
pixi run test-tutorials
```

or

```bash
pixi run build-docs
```
