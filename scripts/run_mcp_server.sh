#!/bin/sh
# Launch the IQC MCP server (stdio) with an interpreter that has iqc and mcp.
# Tries $IQC_PYTHON, then this checkout's .venv, then python3/python on PATH.
# Diagnostics go to stderr; stdout is reserved for JSON-RPC.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Package presence is insufficient: MCP 2.x has no FastMCP v1 API, and
# incomplete installations may be discoverable but fail during import. The
# repo root is on sys.path here, so also import the scientific stack the
# compute tools need rather than only iqc.mcp_server (which needs only mcp).
check='import iqc.mcp_server, iqc.asetools, ase, rdkit'
if [ -n "${IQC_PYTHON:-}" ]; then
    if err=$("$IQC_PYTHON" -c "$check" 2>&1 >/dev/null); then
        exec "$IQC_PYTHON" -m iqc.mcp_server "$@"
    fi
    # An explicit choice is never replaced silently: say why it was skipped.
    echo "iqc MCP: skipping IQC_PYTHON=$IQC_PYTHON, which cannot run the" \
        "server:" >&2
    printf '%s\n' "$err" | tail -n 5 >&2
fi
for py in "$root/.venv/bin/python" python3 python; do
    if "$py" -c "$check" >/dev/null 2>&1; then
        [ -z "${IQC_PYTHON:-}" ] || echo "iqc MCP: using $py instead" >&2
        exec "$py" -m iqc.mcp_server "$@"
    fi
done
echo "iqc MCP: no Python with iqc, its dependencies, and mcp found. Set" \
    "IQC_PYTHON to an interpreter, or run: python3 -m venv $root/.venv &&" \
    "$root/.venv/bin/pip install -e '$root[mcp,xtb]'" >&2
exit 1
