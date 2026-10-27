# IQC with coding agents and MCP

IQC works through its CLI, Python API, or nine MCP tools. The same MCP server
can be used by Codex, Claude Code, Claude Desktop, and other MCP clients.
Agent configuration is versioned in this repository; no account credentials
or global client settings are needed in the checkout.

## Local quick start

Use Python 3.11 or newer (the optional MCP SDK requires at least Python 3.10).
From the repository root:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[mcp,xtb,test]'
python -m iqc.mcp_server --help
IQC_DISABLE_MPI=1 python -m pytest iqc/tests/test_mcp_server.py
```

The server is also available as `iqc-mcp`. Both entry points default to stdio:
they wait for an MCP client on stdin. `--help` checks installation; starting
the process alone does not verify an MCP handshake or a calculator.
For shared machines, set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, and
`OPENBLAS_NUM_THREADS=1` before starting the client.

### Codex

Start Codex from the activated environment in this checkout. The project
`.codex/config.toml` registers IQC; trusted project configuration must be enabled
by the client. `AGENTS.md` supplies repository guidance and
`.agents/skills/iqc/SKILL.md` is the discoverable chemistry skill.

To register IQC for use outside this checkout, use the absolute interpreter
path of the installed environment:

```bash
codex mcp add iqc --env IQC_DISABLE_MPI=1 --env OMP_NUM_THREADS=1 -- /absolute/path/IQC/.venv/bin/python -m iqc.mcp_server
```

Or merge the following into your user `~/.codex/config.toml`:

```toml
[mcp_servers.iqc]
command = "/absolute/path/IQC/.venv/bin/python"
args = ["-m", "iqc.mcp_server"]
startup_timeout_sec = 30
tool_timeout_sec = 600

[mcp_servers.iqc.env]
IQC_DISABLE_MPI = "1"
OMP_NUM_THREADS = "1"
MKL_NUM_THREADS = "1"
OPENBLAS_NUM_THREADS = "1"
```

Avoid registering both a global and project server with different environments
unless intentional. To use the chemistry skill in other repositories, copy the
entire `.agents/skills/iqc` directory to `~/.agents/skills/iqc`.

### Claude Code

Start `claude` from the activated environment at the repository root. The
checked-in `.mcp.json` registers IQC after accepting the client's project MCP
trust prompt. `CLAUDE.md` imports `AGENTS.md`, and `.claude/skills/iqc` links to
the canonical chemistry skill. Invoke it with `/iqc` or ask a chemistry question.

For global registration instead:

```bash
claude mcp add --scope user --transport stdio --env IQC_DISABLE_MPI=1 --env OMP_NUM_THREADS=1 iqc -- /absolute/path/IQC/.venv/bin/python -m iqc.mcp_server
```

### Claude Desktop and other clients

Desktop apps often do not inherit your shell's environment. Merge this into
Claude Desktop's `claude_desktop_config.json` (on macOS, under
`~/Library/Application Support/Claude/`), replacing the interpreter path:

```json
{
  "mcpServers": {
    "iqc": {
      "command": "/absolute/path/IQC/.venv/bin/python",
      "args": ["-m", "iqc.mcp_server"],
      "env": {
        "IQC_DISABLE_MPI": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1"
      }
    }
  }
}
```

Clients such as Cursor can use the same command/args/env configuration in their
own MCP settings. Enable the skill separately if the client supports Agent
Skills; an MCP connection does not automatically install a skill.

## Container quick start

The `mcp` target extends IQC's existing light image with the MCP SDK, xTB,
PySCF, and NWChem. It runs as a non-root user over stdio and starts no notebook
or HTTP listener. It needs no host Python environment or model downloads.
Build it once before configuring a client:

```bash
docker build -f Dockerfile.uv --target mcp -t iqc:mcp .
mkdir -p work
docker run --rm iqc:mcp python -m iqc.mcp_server --help
```

The client launches one container per connection. Use `-i` and omit `-t`, which
would mix terminal output into protocol messages. Example MCP configuration:

```json
{
  "mcpServers": {
    "iqc": {
      "command": "docker",
      "args": ["run", "--rm", "-i", "--mount", "type=bind,src=/absolute/path/IQC/work,dst=/work", "iqc:mcp"]
    }
  }
}
```

For Codex, use the same `command = "docker"` and `args` array in its IQC
server table. The Docker daemon must be running, and the host `work` directory
must be writable by container UID 10001 (or override with `--user UID:GID` on
Linux). Only mounted paths are accessible: `/work/input.xyz` means a file
inside the mounted directory, not an arbitrary host path. Calculator scratch
files are written under `/work`; keep each connection's workspace separate.

Compose is also available:

```bash
docker compose --profile agent build mcp
docker compose --profile agent run --rm -T mcp
```

The container does not include licensed ORCA/VASP binaries or ML checkpoints.
See `docs/docker.md` for the larger MLIP and HPC targets. For long calculations,
use CLI dispatch rather than keeping an MCP client waiting.

## Tools and scientific behavior

| Tool | Purpose |
| --- | --- |
| `smiles_to_xyz` | RDKit 3D conformer guess |
| `xyz_to_smiles` | Perceive connectivity from XYZ |
| `list_calculators` | Supported calculator catalog, not availability detection |
| `run_single_point` | Energy and optional forces |
| `run_optimization` | Relax geometry and return convergence/XYZ |
| `run_vibrations` | Finite-difference frequencies and imaginary-mode count |
| `run_thermo` | Ideal-gas G, H, S, and ZPE at 298.15 K / 1 atm |
| `run_ir` | Frequencies and IR intensities; supports calculator role overrides |
| `inspect_data_file` | Tabular schema and statistics |

Compute tools take exactly one of `smiles` or `xyz`. Omitted charge preserves
input metadata; explicit charge overrides it. Specify multiplicity for known
spin states. Check result `error`, task-specific errors, warnings, and convergence
before treating a tool response as success. Energies are eV, frequencies cm⁻¹,
and entropy eV/K. Read the chemistry skill for calculator selection and caveats.
MCP thermochemistry currently exposes a single calculator, while the CLI/Python
API also support composite thermochemistry.

Calculator instances are mutable, so tool calls are serialized within each
server. ML model instances are cached; file-backed calculators are recreated.
Native backend output and Python diagnostics are routed to stderr in stdio
mode. Heavy imports occur only when a tool needs them.

## Transports and troubleshooting

`python -m iqc.mcp_server --transport streamable-http` enables the modern HTTP
transport on FastMCP's default localhost address/port (typically 8000, `/mcp`).
`--transport sse` remains available for older clients. These modes are intended
for local use; authentication and remote deployment are not configured here.

- Server missing: install `.[mcp]` with the same interpreter used by the client.
- Wrong interpreter: replace `python` with an absolute environment interpreter.
- Calculator missing: install that backend; the calculator catalog does not
  probe executables, checkpoints, or licenses.
- BLAS startup failures: set the three thread-limit variables above.
- Diagnostic logging: set `IQC_MCP_LOG=INFO` (or `DEBUG`); logs go to stderr.
- Timeouts: the project Codex example allows 600 seconds. Other clients have
  their own limits. A timeout may leave work running; check before retrying.
- Files missing in Docker: pass `/work/...` paths and mount the containing folder.

Configuration references: [Codex MCP](https://developers.openai.com/codex/mcp),
[Codex skills](https://developers.openai.com/codex/skills),
[Claude MCP](https://code.claude.com/docs/en/mcp),
[Claude skills](https://code.claude.com/docs/en/skills), and
[MCP stdio logging](https://modelcontextprotocol.io/docs/develop/build-server).
