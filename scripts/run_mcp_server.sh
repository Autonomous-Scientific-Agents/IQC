#!/bin/sh
# Launch the IQC MCP server (stdio) with an interpreter that has iqc and mcp.
# Tries $IQC_PYTHON, then this checkout's .venv, then python3/python on PATH.
# Diagnostics go to stderr; stdout is reserved for JSON-RPC.
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
# Serve this checkout's iqc whatever the client's working directory. Do not
# cd: relative XYZ paths in tool calls are resolved from the client's cwd.
PYTHONPATH="$root${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONPATH
# Package presence is insufficient for mcp: MCP 2.x has no FastMCP v1 API,
# so import iqc.mcp_server (cheap; it loads only mcp). The scientific stack
# is only located, not imported, so a probe stays fast on slow filesystems;
# the server imports it once, lazily, on the first compute call.
check='import iqc.mcp_server, importlib.util as u, sys
missing = [m for m in ("ase", "rdkit", "numpy") if u.find_spec(m) is None]
sys.exit("missing: " + ", ".join(missing) if missing else 0)'
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
