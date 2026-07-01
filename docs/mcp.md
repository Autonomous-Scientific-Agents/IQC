# IQC MCP Server

IQC ships an MCP (Model Context Protocol) server so an LLM agent — Claude
Desktop, Claude Code, Cursor, or any other MCP-aware client — can run
quantum-chemistry workflows directly.

## What it exposes

| Tool                | Returns                                              |
| ------------------- | ---------------------------------------------------- |
| `smiles_to_xyz`     | 3D coordinates (RDKit MMFF94/UFF)                    |
| `xyz_to_smiles`     | Canonical SMILES perceived from XYZ                  |
| `list_calculators`  | Calculator names and dipole/IR compatibility         |
| `run_single_point`  | Energy (+ optional forces)                           |
| `run_optimization`  | Optimized geometry, energy, convergence, opt SMILES  |
| `run_vibrations`    | Vibrational frequencies (cm⁻¹), imag-mode count      |
| `run_thermo`        | G, H, S, ZPE at 298.15 K / 1 atm                     |
| `run_ir`            | IR frequencies + intensities                          |
| `inspect_data_file` | Schema/stats for parquet/CSV/JSONL/Excel/Feather     |

The default calculator is `xtb` (fast and has dipoles for IR). See
`SKILL.md` for guidance on choosing calculators.

## Install

```bash
pip install -e '.[mcp,xtb]'           # base + MCP + xtb (recommended)
pip install -e '.[mcp,mlip,xtb]'      # add MACE / UMA support
```

The MCP server binary is `iqc-mcp` (declared as a console script). Run it
manually to confirm it starts:

```bash
iqc-mcp --help
```

## Register with an MCP client

### Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "iqc": {
      "command": "iqc-mcp",
      "args": [],
      "env": {
        "IQC_DISABLE_MPI": "1",
        "OMP_NUM_THREADS": "1"
      }
    }
  }
}
```

If `iqc-mcp` isn't on your shell PATH for the Claude Desktop process,
substitute the absolute path (`which iqc-mcp` inside your IQC venv).

### Claude Code

Register globally:

```bash
claude mcp add iqc --env IQC_DISABLE_MPI=1 -- iqc-mcp
```

Or add to `.mcp.json` at the project root:

```json
{
  "mcpServers": {
    "iqc": {
      "command": "iqc-mcp",
      "env": {"IQC_DISABLE_MPI": "1"}
    }
  }
}
```

### Cursor / Continue / other clients

Most clients accept the same `command`/`args`/`env` shape. Point them at the
`iqc-mcp` binary inside the venv where IQC is installed.

## Transports

`iqc-mcp` defaults to **stdio**, which is what Claude Desktop / Claude Code
expect. For HTTP/SSE, run:

```bash
iqc-mcp --transport sse        # host/port come from FastMCP defaults
```

## Environment variables

- `IQC_DISABLE_MPI=1` — strongly recommended; the server is a long-lived
  single process and should not try to initialize MPI.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` — cap BLAS
  threads on shared machines to avoid `pthread_create failed` startup errors.
- `IQC_MCP_LOG` — Python log level (default `WARNING`). Set to `INFO` or
  `DEBUG` while diagnosing tool calls.
- `ASE_ORCA_COMMAND` — required only if you want the `orca` calculator and
  the binary isn't on PATH.

## First-call latency

The MCP server is intentionally lazy: importing `iqc.mcp_server` does not
load torch / MACE / UMA. The first call to `run_*(calculator="mace")` will
download checkpoints (hundreds of MB) and load PyTorch — expect a one-time
delay of tens of seconds.
