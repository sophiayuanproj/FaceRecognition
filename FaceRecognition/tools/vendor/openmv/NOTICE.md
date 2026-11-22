# Vendored OpenMV tools

The unmodified `mkromfs.py`, `modelc.py`, and `haar2c.py` were downloaded from
the official OpenMV **v5.0.1** tag on October 14, 2026. Their original copyright
and MIT license declarations are retained in each file. Upstream's repository
at this tag does not contain the top-level LICENSE referenced by those headers.

Sources:

- https://github.com/openmv/openmv/blob/v5.0.1/tools/mkromfs.py
- https://github.com/openmv/openmv/blob/v5.0.1/tools/modelc.py
- https://github.com/openmv/openmv/blob/v5.0.1/tools/haar2c.py
- https://github.com/openmv/openmv/blob/v5.0.1/README.md#licensing

These notices apply to the vendored tools, not a new license decision for the
team's project code. For Nicla, leave accelerator compiler arguments empty;
the model builder only needs these standard-library Python helpers.
