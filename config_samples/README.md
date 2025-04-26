# IQC Configuration Samples

This directory contains sample configuration and script files for running IQC (Interactive Quantum Chemistry) calculations.

## Files

1.  **`config.yaml`**
    *   **Purpose:** Provides an example YAML configuration file for customizing IQC runs.
    *   **Content:** Shows how to set parameters for:
        *   Calculator selection (`calculator`: mace, xtb, emt)
        *   Calculator-specific settings (`calculator_params`: e.g., model, dispersion for MACE; method for XTB).
        *   Optimization settings (`optimization_params`: e.g., `fmax`, `max_steps`).
        *   (Placeholders for future thermo and vibration settings).
    *   **Usage:** Use the `-p` or `--params` command-line argument with `iqc.main` to specify a configuration file, e.g., `python -m iqc.main xyz/water.xyz -p config_samples/config.yaml`.

2.  **`crux_threads_test.sh`**
    *   **Purpose:** Example shell script demonstrating how to run IQC on ALCF's Crux supercomputer for performance testing.
    *   **Content:** The script runs IQC with different numbers of threads and reports the performance.
    *   **Usage:** Adapt the script for your specific system (paths, modules, batch scheduler commands) and execute it, e.g., `qsub crux_threads_test.sh`.