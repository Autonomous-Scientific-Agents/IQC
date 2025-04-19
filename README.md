# IQC
Interactive Quantum Chemistry

[![CI](https://github.com/Autonomous-Scientific-Agents/IQC/actions/workflows/ci.yml/badge.svg)](https://github.com/Autonomous-Scientific-Agents/IQC/actions/workflows/ci.yml)

## Installation

### Clone the repository
   ```bash
   git clone git@github.com:Autonomous-Scientific-Agents/IQC.git
   cd IQC
   ```

### Option 1: Using Conda

1. First, install a package manager (Conda, Miniconda, Mamba, or MicroMamba)
   - Download Miniconda from the [official page](https://docs.conda.io/en/latest/miniconda.html)
   - Follow the installation instructions for your operating system

2. Create and activate the environment:
   ```bash
   conda env create -f env.yml
   conda activate iqc-env
   ```

3. Install IQC:
   ```bash
   pip install .
   ```

### Option 2: Using Docker

1. Build the Docker image:
   ```bash
   docker build -t iqc .
   ```

2. Run the container with Jupyter Lab:
   ```bash
   docker run -p 8888:8888 -it iqc
   ```

3. Access Jupyter Lab by opening `http://localhost:8888` in your web browser

To persist your notebooks, you can mount a local directory:
```bash
docker run -p 8888:8888 -v $(pwd)/notebooks:/app/notebooks -it iqc
```

# Troubleshooting
If you see (a possible OpenMPI error):

```bash
shmem: mmap: an error occurred while determining whether or not /tmp/ompi.yv.1001/jf.0/3074883584/sm_segment.yv.1001.b7470000.0 could be created
```

Try: `export OMPI_MCA_btl_sm_backing_directory=/tmp`

## Available Tasks

IQC provides several computational tasks that can be performed on molecular systems:

- `single`: Single-point energy and force calculation
- `vib`: Vibrational frequency calculation
- `opt`: Geometry optimization
- `thermo`: Thermochemical analysis

These tasks can be specified when running calculations. For example:

```python
from iqc.asetools import run_single_point, run_vibrations, run_optimization, run_thermo

# Single-point calculation
results = run_single_point(atoms)

# Vibrational frequency calculation
results = run_vibrations(atoms)

# Geometry optimization
results = run_optimization(atoms)

# Thermochemical analysis
results = run_thermo(atoms)
```

Each task returns a dictionary containing the results and timing information in milliseconds.
