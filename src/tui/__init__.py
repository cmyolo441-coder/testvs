"""The OMH TUI: the terminal frontend bare `omh` opens.

Hermes' launcher starts it in place of its own Ink app (through `HERMES_TUI_DIR`),
and it speaks Hermes' `tui_gateway` stdio JSON-RPC to its gateway child, which
reaches Hermes' gateway (`attach/gateway_attach.py`). Layout, input, rendering
and the OMH panels are this package's; every model call is the gateway's.
"""
