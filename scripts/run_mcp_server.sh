#!/bin/sh
# Launch the IQC MCP server (stdio) with an interpreter that has iqc and mcp.
# Tries $IQC_PYTHON, then this checkout's .venv, then python3/python on PATH.
# Diagnostics go to stderr; stdout is reserved for JSON-RPC.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
check='import importlib.util as u, sys; sys.exit(not (u.find_spec("mcp") and u.find_spec("iqc")))'
for py in "${IQC_PYTHON:-}" "$root/.venv/bin/python" python3 python; do
    [ -n "$py" ] || continue
    if "$py" -c "$check" >/dev/null 2>&1; then
        exec "$py" -m iqc.mcp_server "$@"
    fi
done
echo "iqc MCP: no Python with iqc and mcp found. Set IQC_PYTHON to an" \
    "interpreter, or run: python3 -m venv $root/.venv &&" \
    "$root/.venv/bin/pip install -e '$root[mcp,xtb]'" >&2
exit 1
