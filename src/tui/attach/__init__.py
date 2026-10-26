"""What the OMH TUI runs on Hermes' interpreter, never on OMH's: see `gateway_attach`.

Nothing in `omh` imports this package. `gateway_attach.py` runs by path, which
puts this directory on its import path, so the directory holds no module whose
name could stand in for one of Hermes' own.
"""
