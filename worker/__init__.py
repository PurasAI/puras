"""Puras runtime (the open-source runner).

Same package the hosted worker runs; also the dependency-light **offline runner**
behind `puras run --local` (see `worker.local_run`). The hosted-only modules
(db / queue / storage / memory) pull the platform stack, but they're never
imported on the offline path, so `pip install puras[local]` stays thin.
"""

__version__ = "0.1.1"
