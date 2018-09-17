# GASpy Regressions

[GASpy](https://github.com/ulissigroup/GASpy/tree/v0.1) is able to create
various catalyst-adsorbate systems and then use DFT to simulate the adsorption
energies of these systems.
[GASpy_regressions](https://github.com/ktran9891/GASpy_regressions) analyzes
GASpy's results to create surrogate models that can make predictions on DFT
calculations that we have not yet performed. We then store these predictions in
the Mongo collections that we set up in GASpy. Refer to our
[Jupyter](http://jupyter.org/)
[notebooks](https://github.com/ulissigroup/GASpy_regressions/tree/master/notebooks)
for examples/specifics.

# Installation

You will need to first install GASpy. Then to use GASpy_regressions, you will need
to make sure that this repository is cloned into your local repository of GASpy
as a [submodule](https://git-scm.com/book/en/v2/Git-Tools-Submodules). Then run
via Docker, e.g. `docker run -v "/local/path/to/GASpy:/home/GASpy"
ulissigroup/gaspy_regressions:latest foo`.

# Reference

[Active learning across intermetallics to guide discovery of electrocatalysts
for CO2 reduction and H2
evolution](https://www.nature.com/articles/s41929-018-0142-1). Note that the
repository which we reference in this paper is version 0.1 of GASpy_feedback,
which can stil be found
[here](https://github.com/ulissigroup/GASpy_regressions/tree/v0.1).

# Versions

Current GASpy_regressions version: 0.20

For an up-to-date list of our software dependencies, you can simply check out
how we build our docker image
[here](https://github.com/ulissigroup/GASpy_regressions/blob/master/dockerfile/Dockerfile).
