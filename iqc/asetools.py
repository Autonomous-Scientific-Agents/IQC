import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import ase
from ase import Atoms, build
from ase.calculators.calculator import PropertyNotImplementedError, PropertyNotPresent
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.optimize import BFGS
from ase.thermochemistry import IdealGasThermo
from ase.vibrations import Vibrations
from ase.visualize import view
from rdkit import Chem
from rdkit.Chem import AllChem, rdmolops
import io

# Optional dependencies with informative messages
XTB = None
try:
    from xtb.ase.calculator import XTB
except ImportError:
    logging.warning(
        "XTB calculator not available. Install with 'pip install xtb' if you need quantum chemistry calculations with XTB."
    )


def get_ase_version():
    """
    Returns the version of the ASE library.
    """
    return ase.__version__


def _normalize_calculator_compatibility(calculator):
    """Expose stable compatibility attributes across ASE calculator versions."""

    if calculator is None:
        return None

    mixer = getattr(calculator, "mixer", None)
    mixer_calcs = getattr(mixer, "calcs", None)
    if mixer_calcs is not None and not hasattr(calculator, "calcs"):
        calculator.calcs = mixer_calcs

    return calculator


def _patch_e3nn_codegen_legacy_state(codegen_mixin=None):
    """Allow newer e3nn to load MACE checkpoints saved with older codegen state."""

    if codegen_mixin is None:
        try:
            from e3nn.util.codegen import _mixin as codegen_mixin
        except Exception:
            return False

    codegen_cls = getattr(codegen_mixin, "CodeGenMixin", None)
    if codegen_cls is None or getattr(codegen_cls, "_iqc_legacy_state_patch", False):
        return False

    original_setstate = getattr(codegen_cls, "__setstate__", None)
    code = getattr(original_setstate, "__code__", None)
    if original_setstate is None or code is None:
        return False

    if "buffer_type" not in code.co_varnames:
        return False

    def _iqc_setstate(self, state):
        if isinstance(state, dict):
            codegen_state = state.get("__codegen__")
            if isinstance(codegen_state, dict):
                legacy_codegen_state = {
                    name: (
                        ("torchscript", buffer) if isinstance(buffer, bytes) else buffer
                    )
                    for name, buffer in codegen_state.items()
                }
                if legacy_codegen_state != codegen_state:
                    state = state.copy()
                    state["__codegen__"] = legacy_codegen_state
        return original_setstate(self, state)

    codegen_cls.__setstate__ = _iqc_setstate
    codegen_cls._iqc_legacy_state_patch = True
    codegen_cls._iqc_original_setstate = original_setstate
    return True


def _make_e3nn_spherical_harmonics_func():
    """Build the callable e3nn 0.5 expects on SphericalHarmonics instances."""

    import torch
    from e3nn import get_optimization_defaults
    from e3nn.o3._spherical_harmonics import _spherical_harmonics

    jit_mode = get_optimization_defaults().get("jit_mode")
    if jit_mode == "script":
        return torch.jit.script(_spherical_harmonics)
    if jit_mode == "inductor":
        return torch.compile(_spherical_harmonics, fullgraph=True)
    return _spherical_harmonics


def _restore_e3nn_spherical_harmonics_sph_func(module, sph_func_factory=None):
    """Restore e3nn 0.5 SphericalHarmonics state missing from old checkpoints."""

    if module.__class__.__name__ != "SphericalHarmonics":
        return False
    if hasattr(module, "sph_func"):
        return False

    if sph_func_factory is None:
        sph_func_factory = _make_e3nn_spherical_harmonics_func
    module.sph_func = sph_func_factory()
    return True


def _patch_e3nn_spherical_harmonics_legacy_state(
    spherical_harmonics_cls=None, sph_func_factory=None
):
    """Patch e3nn SphericalHarmonics restored from older MACE checkpoints."""

    if spherical_harmonics_cls is None:
        try:
            from e3nn.o3._spherical_harmonics import SphericalHarmonics

            spherical_harmonics_cls = SphericalHarmonics
        except Exception:
            return False

    if getattr(spherical_harmonics_cls, "_iqc_sph_func_patch", False):
        return False

    original_forward = getattr(spherical_harmonics_cls, "forward", None)
    original_setstate = getattr(spherical_harmonics_cls, "__setstate__", None)
    if original_forward is None:
        return False

    def _iqc_forward(self, *args, **kwargs):
        _restore_e3nn_spherical_harmonics_sph_func(self, sph_func_factory)
        return original_forward(self, *args, **kwargs)

    spherical_harmonics_cls.forward = _iqc_forward
    spherical_harmonics_cls._iqc_original_forward = original_forward

    if original_setstate is not None:

        def _iqc_setstate(self, state):
            result = original_setstate(self, state)
            _restore_e3nn_spherical_harmonics_sph_func(self, sph_func_factory)
            return result

        spherical_harmonics_cls.__setstate__ = _iqc_setstate
        spherical_harmonics_cls._iqc_original_setstate = original_setstate

    spherical_harmonics_cls._iqc_sph_func_patch = True
    return True


def _restore_e3nn_activation_paths(module):
    """Restore e3nn 0.5 Activation paths missing from old checkpoints."""

    if module.__class__.__name__ != "Activation":
        return False
    if hasattr(module, "paths"):
        return False
    if not hasattr(module, "irreps_in") or not hasattr(module, "acts"):
        return False

    module.paths = [
        (mul, (l, p), act) for (mul, (l, p)), act in zip(module.irreps_in, module.acts)
    ]
    return True


def _patch_e3nn_activation_legacy_state(activation_cls=None):
    """Patch e3nn Activation restored from older MACE checkpoints."""

    if activation_cls is None:
        try:
            from e3nn.nn._activation import Activation

            activation_cls = Activation
        except Exception:
            return False

    if getattr(activation_cls, "_iqc_paths_patch", False):
        return False

    original_forward = getattr(activation_cls, "forward", None)
    original_setstate = getattr(activation_cls, "__setstate__", None)
    if original_forward is None:
        return False

    def _iqc_forward(self, *args, **kwargs):
        _restore_e3nn_activation_paths(self)
        return original_forward(self, *args, **kwargs)

    activation_cls.forward = _iqc_forward
    activation_cls._iqc_original_forward = original_forward

    if original_setstate is not None:

        def _iqc_setstate(self, state):
            result = original_setstate(self, state)
            _restore_e3nn_activation_paths(self)
            return result

        activation_cls.__setstate__ = _iqc_setstate
        activation_cls._iqc_original_setstate = original_setstate

    activation_cls._iqc_paths_patch = True
    return True


def _patch_e3nn_mace_compatibility():
    """Apply e3nn compatibility patches needed by MACE foundation checkpoints."""

    mace_safe_globals_patched = False
    try:
        import importlib
        import inspect
        import torch

        safe_globals = []
        for module_name in (
            "e3nn.math",
            "e3nn.nn",
            "e3nn.nn._activation",
            "e3nn.nn._batchnorm",
            "e3nn.nn._fc",
            "e3nn.nn._gate",
            "e3nn.nn._normact",
            "e3nn.o3",
            "e3nn.o3._irreps",
            "e3nn.o3._linear",
            "e3nn.o3._spherical_harmonics",
            "e3nn.o3._tensor_product._sub",
            "e3nn.o3._tensor_product._tensor_product",
            "mace.modules.blocks",
            "mace.modules.models",
            "mace.modules.radial",
            "mace.modules.symmetric_contraction",
            "mace.modules.utils",
            "torch.nn",
            "torch.nn.functional",
            "torch.nn.modules.activation",
            "torch.nn.modules.container",
            "torch.nn.modules.linear",
            "torch.nn.modules.normalization",
            "torch.fx._symbolic_trace",
            "torch.fx.graph_module",
        ):
            module = importlib.import_module(module_name)
            safe_globals.extend(
                obj
                for obj in vars(module).values()
                if (inspect.isclass(obj) or inspect.isfunction(obj))
                and getattr(obj, "__module__", "").startswith(
                    ("e3nn.", "mace.modules.", "torch.nn.", "torch.fx.")
                )
            )
        if safe_globals:
            torch.serialization.add_safe_globals(safe_globals)
            mace_safe_globals_patched = True
    except Exception:
        pass

    codegen_patched = _patch_e3nn_codegen_legacy_state()
    spherical_harmonics_patched = _patch_e3nn_spherical_harmonics_legacy_state()
    activation_patched = _patch_e3nn_activation_legacy_state()
    return (
        mace_safe_globals_patched
        or codegen_patched
        or spherical_harmonics_patched
        or activation_patched
    )


def _patch_ase_orca_dipole():
    """Make `ase.io.orca.read_orca_output` emit a dipole for ORCA 6 outputs.

    ORCA 6 dropped the "The origin for moment calculation is the CENTER OF
    MASS = (...)" line that ASE's parser scrapes. Without it, the parser
    reads the dipole successfully but discards it because its
    `if com is not None and dipole is not None` guard fails. The dipole as
    printed by ORCA 6 is in the COM frame; for neutral molecules and for
    finite-difference IR intensities (which only see derivatives) the COM
    correction term is zero, so we can safely emit the dipole as-is when the
    COM line is missing.
    """

    try:
        from ase.io import orca as _orca_io
        from ase.utils import reader as _ase_reader
    except ImportError:
        return False

    if getattr(_orca_io, "_iqc_dipole_patched", False):
        return False

    _original_read_orca_output = getattr(
        _orca_io, "_iqc_original_read_orca_output", _orca_io.read_orca_output
    )
    _orca_io._iqc_original_read_orca_output = _original_read_orca_output

    def _read_orca_dipole_with_com_fallback(lines):
        dipole = _orca_io.read_dipole(lines)
        if dipole is None:
            return None
        com = _orca_io.read_center_of_mass(lines)
        charge = _orca_io.read_charge(lines)
        if com is None:
            # ORCA 6: dipole is already in COM frame; for neutral systems
            # the lab-frame value is identical, and IR derivatives are
            # frame-independent regardless of charge.
            return dipole
        return dipole + com * (charge if charge is not None else 0)

    def _attach_dipole(parsed, dipole):
        if dipole is None:
            return parsed
        images = parsed if isinstance(parsed, list) else [parsed]
        for image in images:
            calc = getattr(image, "calc", None)
            calc_results = getattr(calc, "results", None)
            if isinstance(calc_results, dict) and calc_results.get("dipole") is None:
                calc_results["dipole"] = dipole
        return parsed

    @_ase_reader
    def _read_orca_output_with_dipole_fallback(fd, index=slice(None)):
        # Let the original parser run first via its own path (so engrad and
        # other branches stay unchanged), then patch the dipole in if it was
        # dropped because ORCA 6 omitted the COM line.
        try:
            fd.seek(0)
            lines = fd.readlines()
        except Exception:
            lines = None
        fd.seek(0)
        parsed = _original_read_orca_output(fd, index=index)
        if lines is None:
            return parsed
        return _attach_dipole(parsed, _read_orca_dipole_with_com_fallback(lines))

    _orca_io.read_orca_output = _read_orca_output_with_dipole_fallback
    _orca_io._iqc_dipole_patched = True
    return True


def _is_orca_calculator(calculator):
    return calculator is not None and type(calculator).__name__ == "ORCA"


def _ensure_orca_engrad_for_forces(calculator):
    """Request ORCA gradient output when ASE will need forces."""

    if not _is_orca_calculator(calculator):
        return False

    parameters = getattr(calculator, "parameters", None)
    if parameters is None:
        return False

    simpleinput = str(
        parameters.get("orcasimpleinput") or "B3LYP def2-TZVP"
    ).strip()
    if re.search(r"(?i)(^|\s)engrad($|\s)", simpleinput):
        return False

    parameters["orcasimpleinput"] = f"{simpleinput} ENGRAD".strip()
    logging.info("Added ENGRAD to ORCA simple input for ASE force evaluation.")
    return True


def _safe_path_component(value):
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return safe.strip("._") or "calc"


def _assign_orca_work_directory(calculator, unique_name, purpose="calc", base_dir=None):
    """Give auto-managed ORCA calculators a per-record work directory."""

    if not _is_orca_calculator(calculator):
        return False

    current = Path(getattr(calculator, "directory", "."))
    auto_managed = getattr(calculator, "_iqc_auto_directory", False)
    if not auto_managed and current != Path("."):
        return False

    root = Path(base_dir) if base_dir else Path(".")
    directory = root / f"{_safe_path_component(unique_name)}_{purpose}_orca"
    calculator.directory = directory
    calculator._iqc_auto_directory = True
    logging.debug("Using ORCA work directory: %s", directory)
    return True


def _read_file_tail(path, max_lines=40):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _orca_failure_context(calculator):
    if not _is_orca_calculator(calculator):
        return ""

    directory = Path(getattr(calculator, "directory", "."))
    template = getattr(calculator, "template", None)
    names = []
    for attr, default in (("errorname", "orca.err"), ("outputname", "orca.out")):
        names.append(getattr(template, attr, default))

    parts = [f"ORCA work directory: {directory}"]
    for name in names:
        path = directory / name
        tail = _read_file_tail(path)
        if tail:
            parts.append(f"Tail of {path}:\n{tail}")
    return "\n" + "\n".join(parts) if len(parts) > 1 else "\n" + parts[0]


UMA_DEFAULT_MODEL_BY_SIZE = {
    "s": "uma-s-1p2",
    "m": "uma-m-1p1",
}
UMA_TASKS = {"omol", "omat", "odac"}
UMA_PREDICTOR_KWARGS = {
    "cache_dir",
    "device",
    "inference_settings",
    "overrides",
    "seed",
    "workers",
}


def _parse_uma_calculator_name(name):
    """Return the UMA model name and task encoded in an IQC calculator name."""

    if name == "uma":
        return UMA_DEFAULT_MODEL_BY_SIZE["s"], "omol"

    parts = name.split("-")
    if len(parts) != 3 or parts[0] != "uma":
        raise ValueError(
            "UMA calculator names must be 'uma' or 'uma-{s,m}-{omol,omat,odac}'."
        )

    _, size, task = parts
    if size not in UMA_DEFAULT_MODEL_BY_SIZE:
        raise ValueError("UMA calculator size must be 's' or 'm'.")
    if task not in UMA_TASKS:
        raise ValueError(f"UMA task must be one of {', '.join(sorted(UMA_TASKS))}.")

    return UMA_DEFAULT_MODEL_BY_SIZE[size], task


def _get_uma_calculator(name, **kwargs):
    """Initialize a FAIRChem UMA calculator with IQC's compact name aliases."""

    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor_name, task = _parse_uma_calculator_name(name)
    uma_kwargs = dict(kwargs)
    predictor_name = uma_kwargs.pop("model", predictor_name)
    task = uma_kwargs.pop("task_name", task)
    predictor_kwargs = {
        key: uma_kwargs.pop(key)
        for key in list(uma_kwargs)
        if key in UMA_PREDICTOR_KWARGS
    }

    predictor = pretrained_mlip.get_predict_unit(predictor_name, **predictor_kwargs)
    calculator = FAIRChemCalculator(predictor, task_name=task, **uma_kwargs)
    calculator.model_name = predictor_name
    logging.info(
        "Using UMA calculator with model=%s, task=%s, predictor_args=%s, calculator_args=%s",
        predictor_name,
        task,
        predictor_kwargs,
        uma_kwargs,
    )
    return calculator


def get_calculator(name="mace", **kwargs):
    """Initializes and returns the specified ASE calculator.

    Args:
        name (str): The name of the calculator ('mace', 'xtb', 'emt', 'uma',
                    'uma-s-omol', 'uma-s-omat', 'uma-s-odac',
                    'uma-m-omol', 'uma-m-omat', or 'uma-m-odac').
        **kwargs: Additional keyword arguments passed to the calculator constructor.

    Returns:
        ase.calculators.calculator.Calculator: The initialized calculator instance.
                                                 Returns EMT as a fallback if the requested
                                                 calculator is not available or fails to initialize.
    """
    name = name.lower()
    calculator = None

    if name == "mace":
        try:
            _patch_e3nn_mace_compatibility()
            from mace.calculators import mace_mp

            mace_kwargs = {
                "model": "large",
                "dispersion": True,
                "default_dtype": "float64",
                "device": "cpu",
                **kwargs,
            }
            try:
                # Attempt with specified/default dispersion
                calculator = mace_mp(**mace_kwargs)
                calculator.model_name = mace_kwargs["model"]
                logging.info(f"Using MACE calculator with arguments: {mace_kwargs}")
            except Exception as e:
                # Try without dispersion if the first attempt failed
                logging.warning(
                    f"Failed to initialize MACE with dispersion={mace_kwargs.get('dispersion')}: {str(e)}. Trying with dispersion=False."
                )
                mace_kwargs["dispersion"] = False
                calculator = mace_mp(**mace_kwargs)
                calculator.model_name = mace_kwargs["model"]
                logging.info(f"Using MACE calculator with arguments: {mace_kwargs}")
        except ImportError:
            logging.warning(
                "MACE not found. Install with 'pip install mace'. Falling back to EMT."
            )
        except RuntimeError as e:
            logging.warning(f"MACE initialization failed: {e}. Falling back to EMT.")

    elif name == "xtb":
        try:
            from xtb.ase.calculator import XTB

            xtb_kwargs = {"method": "GFN2-xTB", **kwargs}
            calculator = XTB(**xtb_kwargs)
            logging.info(f"Using XTB calculator with arguments: {xtb_kwargs}")
        except ImportError:
            logging.warning(
                "XTB not found. Install with 'pip install xtb' or 'pip install iqc[xtb]'. Falling back to MACE."
            )
        except Exception as e:
            logging.warning(f"XTB initialization failed: {e}. Falling back to MACE.")

    elif name == "emt":
        try:
            from ase.calculators.emt import EMT

            calculator = EMT(**kwargs)
            logging.info(f"Using EMT calculator with arguments: {kwargs}")
        except ImportError:
            # This case should ideally not happen if ASE is installed correctly
            logging.warning(
                "EMT not found, but it's usually built-in with ASE. Problem with ASE install? Returning None."
            )
            return None

    elif name.startswith("uma"):
        try:
            calculator = _get_uma_calculator(name, **kwargs)
        except ImportError as e:
            logging.warning(
                "FAIRChem UMA calculator import failed: %s. Install fairchem-core "
                "with compatible dependencies and follow the MACE/UMA install "
                "workaround. Falling back to MACE.",
                e,
            )
        except Exception as e:
            logging.warning(f"UMA initialization failed: {e}. Falling back to MACE.")

    elif name == "orca":
        import shutil

        try:
            # ORCA 6 dropped the COM line ASE keys off of when extracting
            # dipoles; without this patch IR intensities silently come back
            # zero/missing.
            _patch_ase_orca_dipole()
            from ase.calculators.orca import ORCA, OrcaProfile

            orca_kwargs = dict(kwargs)
            # Resolve ORCA executable: explicit `command:` kwarg →
            # ASE_ORCA_COMMAND env → first `orca` on PATH. ASE's config file
            # is only consulted if all three of those miss.
            command = (
                orca_kwargs.pop("command", None)
                or os.environ.get("ASE_ORCA_COMMAND")
                or shutil.which("orca")
            )
            profile = orca_kwargs.pop("profile", None)
            if profile is None and command:
                profile = OrcaProfile(command=command)
            orca_kwargs.setdefault("orcasimpleinput", "B3LYP def2-SVP")
            orca_kwargs.setdefault("orcablocks", "%pal nprocs 1 end")
            if profile is not None:
                calculator = ORCA(profile=profile, **orca_kwargs)
                logging.info(
                    f"Using ORCA calculator (command={command}) with "
                    f"arguments: {orca_kwargs}"
                )
            else:
                # ASE will look up the executable in its config file.
                calculator = ORCA(**orca_kwargs)
                logging.info(
                    f"Using ORCA calculator (ASE config) with "
                    f"arguments: {orca_kwargs}"
                )
        except ImportError:
            logging.warning(
                "ASE ORCA calculator not available. Falling back to MACE."
            )
        except Exception as e:
            logging.warning(f"ORCA initialization failed: {e}. Falling back to MACE.")

    else:
        logging.warning(f"Unknown calculator '{name}'. Falling back to MACE.")

    # Fallback to MACE if the requested calculator failed or was unknown
    # MACE is a required dependency, so it should be available
    if calculator is None:
        logging.warning(
            f"Calculator '{name}' failed or not found. Attempting fallback to MACE."
        )
        try:
            _patch_e3nn_mace_compatibility()
            from mace.calculators import mace_mp

            mace_kwargs = {
                "model": "large",
                "dispersion": True,
                "default_dtype": "float64",
                "device": "cpu",
            }
            try:
                calculator = mace_mp(**mace_kwargs)
                calculator.model_name = mace_kwargs["model"]
                logging.info("Using MACE calculator as fallback.")
            except Exception as e:
                # Try without dispersion if the first attempt failed
                logging.warning(
                    f"Failed to initialize MACE fallback with dispersion: {str(e)}. Trying without dispersion."
                )
                mace_kwargs["dispersion"] = False
                calculator = mace_mp(**mace_kwargs)
                calculator.model_name = mace_kwargs["model"]
                logging.info("Using MACE calculator as fallback (without dispersion).")
            # Tag so callers can detect that they got a fallback rather than
            # the calculator they asked for. SumCalculator(MACE+D3) hides the
            # MACE class name from a naive `type()` check.
            try:
                calculator._iqc_fallback_from = name
            except (AttributeError, TypeError):
                pass
        except ImportError:
            logging.error(
                "MACE fallback calculator could not be imported. MACE is a required dependency."
            )
            raise RuntimeError(
                "No suitable ASE calculator found. MACE (required dependency) is not available."
            )
        except Exception as e:
            logging.error(f"MACE fallback initialization failed: {e}")
            # Last resort: try EMT
            try:
                from ase.calculators.emt import EMT

                calculator = EMT()
                logging.warning("Using EMT calculator as last resort fallback.")
            except ImportError:
                logging.error(
                    "All fallback calculators failed. No calculator available."
                )
                raise RuntimeError(
                    "No suitable ASE calculator found or could be initialized."
                )

    return _normalize_calculator_compatibility(calculator)


def save_atoms(atoms, prefix="", suffix="", file_format="xyz", directory=None):
    """
    Save an ASE Atoms object to a file with a name composed of:
    [prefix]_[chemical_formula]_[timestamp]_[suffix].[file_format]

    Args:
        atoms (ase.Atoms): The ASE Atoms object to be saved.
        prefix (str): Optional prefix for the filename.
        suffix (str): Optional suffix for the filename.
        file_format (str): The format in which to save the file (default is "xyz").
        directory (str): Optional directory where the file will be saved. If not provided, the file will be saved in the current directory.

    Returns:
        str: The path to the saved file.
    """
    # Get chemical formula in a canonical form
    formula = atoms.get_chemical_formula(mode="hill")

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Construct the filename
    base_name_parts = [part for part in [prefix, formula, timestamp, suffix] if part]
    base_name = "_".join(base_name_parts) + "." + file_format

    if directory:
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, base_name)
    else:
        file_path = base_name

    # Write the atoms to file
    write(file_path, atoms, format=file_format)

    return file_path


def print_atoms_info(atoms):
    """
    Prints the index, symbol, and coordinates of each atom in an ASE Atoms object.

    Args:
        atoms (ase.Atoms): The ASE Atoms object containing the atoms.

    Returns:
        None
    """
    print(f"{'Index':<6}{'Symbol':<8}{'Coordinates':<30}")
    print("-" * 44)
    for i, atom in enumerate(atoms):
        print(
            f"{i:<6}{atom.symbol:<8}{atom.position[0]:<10.3f}{atom.position[1]:<10.3f}{atom.position[2]:<10.3f}"
        )


def translate_atoms(atoms, indices, reference_index, target_index, distance):
    """
    Translates specified atoms in the direction of a vector defined by two reference atoms.

    Args:
        atoms (ase.Atoms): The ASE Atoms object containing the atoms.
        indices (list): List of indices of the atoms to be translated.
        reference_index (int): Index of the atom defining the origin of the direction vector.
        target_index (int): Index of the atom defining the target of the direction vector.
        distance (float): Distance to translate the atoms in the specified direction (in angstroms).

    Returns:
        ase.Atoms: A new ASE Atoms object with the translated atoms.
    """
    # Calculate the direction vector from reference_index to target_index
    direction_vector = atoms[target_index].position - atoms[reference_index].position
    # Normalize the direction vector
    direction_unit_vector = direction_vector / np.linalg.norm(direction_vector)
    # Compute the translation vector
    translation_vector = direction_unit_vector * distance

    # Translate the specified atoms
    for idx in indices:
        atoms[idx].position += translation_vector

    return atoms


def get_rdmol_from_smiles(smiles: str, optimize=False, seed=0xF00D):
    """Convert a SMILES string to an RDKit molecule.

    Args:
        smiles (str): The SMILES string representing the molecule.
        optimize (bool, optional): Whether to optimize the molecule geometry using MMFF. Defaults to False.
        seed (int, optional): The random seed for molecule embedding. Defaults to 0xF00D.

    Returns:
        rdkit.Chem.rdchem.Mol: The RDKit molecule object.
    """
    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    rdmol = Chem.AddHs(rdmol)
    if not optimize:
        status = AllChem.EmbedMolecule(rdmol, randomSeed=seed)
        if status != 0:
            raise ValueError(f"Failed to generate a 3D geometry from SMILES: {smiles}")
        return rdmol

    num_conformers = 20
    params = AllChem.ETKDGv3() if hasattr(AllChem, "ETKDGv3") else AllChem.ETKDG()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.35
    params.enforceChirality = True
    conf_ids = list(
        AllChem.EmbedMultipleConfs(rdmol, numConfs=num_conformers, params=params)
    )
    if not conf_ids:
        status = AllChem.EmbedMolecule(rdmol, randomSeed=seed)
        if status != 0:
            raise ValueError(f"Failed to generate a 3D geometry from SMILES: {smiles}")
        conf_ids = [0]

    props = None
    force_field_name = "UFF"
    if AllChem.MMFFHasAllMoleculeParams(rdmol):
        props = AllChem.MMFFGetMoleculeProperties(rdmol)
        if props is not None:
            force_field_name = "MMFF94"

    best_conf_id = int(conf_ids[0])
    best_energy = None
    for conf_id in conf_ids:
        try:
            if props is not None:
                AllChem.MMFFOptimizeMolecule(
                    rdmol, mmffVariant="MMFF94", confId=conf_id, maxIters=500
                )
                force_field = AllChem.MMFFGetMoleculeForceField(
                    rdmol, props, confId=conf_id
                )
            else:
                AllChem.UFFOptimizeMolecule(rdmol, confId=conf_id, maxIters=500)
                force_field = AllChem.UFFGetMoleculeForceField(rdmol, confId=conf_id)
            energy = (
                float(force_field.CalcEnergy()) if force_field is not None else None
            )
        except Exception:
            energy = None
        if energy is not None and (best_energy is None or energy < best_energy):
            best_energy = energy
            best_conf_id = int(conf_id)

    rdmol.SetIntProp("_IQCLowestEnergyConformerId", best_conf_id)
    rdmol.SetProp(
        "_IQCCanonicalSmiles",
        Chem.MolToSmiles(Chem.RemoveHs(Chem.Mol(rdmol)), canonical=True),
    )
    rdmol.SetProp("_IQCForceField", force_field_name)
    if best_energy is not None:
        rdmol.SetDoubleProp("_IQCLowestEnergy", best_energy)
    return rdmol


def get_rdmol_from_inchi(inchi: str, optimize=False, seed=0xF00D):
    """
    Convert an InChI string to an RDKit molecule.

    Args:
        inchi (str): The InChI string representing the molecule.
        optimize (bool, optional): Whether to optimize the molecule geometry using MMFF. Defaults to False.
        seed (int, optional): The random seed for molecule embedding. Defaults to 0xF00D.

    Returns:
        rdkit.Chem.rdchem.Mol: The RDKit molecule object.
    """
    rdmol = AllChem.MolFromInchi(inchi)
    rdmol = AllChem.AddHs(rdmol)
    AllChem.EmbedMolecule(rdmol, randomSeed=seed)
    if optimize:
        AllChem.MMFFOptimizeMolecule(rdmol)
    return rdmol


def get_rdmol_from_xyz(xyz: str):
    """Return RDKit molecule from xyz formatted string or a path to an xyz file

    Parameters
    ----------
    xyz : str
        xyz formatted string or a path to an xyz file

    Returns
    -------
    rdkit.Chem.rdchem.Mol
    """
    import rdkit
    from os.path import isfile

    if isfile(xyz):
        return rdkit.Chem.rdmolfiles.MolFromXYZFile(xyz)
    return rdkit.Chem.AllChem.MolFromXYZBlock(xyz)


def get_xyz_from_rdmol(rdmol, conf_id=None):
    """
    Convert an RDKit molecule to an XYZ formatted string.

    Args:
        rdmol (rdkit.Chem.rdchem.Mol): The RDKit molecule object.
        conf_id (int, optional): The conformer id to export. If not provided,
            the lowest-energy conformer selected by IQC is used when available.

    Returns:
        str: The XYZ formatted string representing the molecule.
    """
    if conf_id is None and rdmol.HasProp("_IQCLowestEnergyConformerId"):
        conf_id = rdmol.GetIntProp("_IQCLowestEnergyConformerId")
    conf = rdmol.GetConformer(conf_id) if conf_id is not None else rdmol.GetConformer()
    xyz = str(rdmol.GetNumAtoms()) + "\n\n"
    for i in range(rdmol.GetNumAtoms()):
        at = rdmol.GetAtomWithIdx(i)
        symbol = str(at.GetSymbol())
        pos = conf.GetAtomPosition(i)
        x, y, z = pos.x, pos.y, pos.z
        xyz += f"{symbol} {x} {y} {z}\n"
    return xyz


def get_atoms_from_smiles(smiles: str, seed=0xF00D):
    """Generate an ASE Atoms object from a SMILES string using RDKit.

    The geometry is built from the lowest-energy embedded conformer using MMFF94
    when available, otherwise UFF.
    """

    rdmol = get_rdmol_from_smiles(smiles, optimize=True, seed=seed)
    atoms = get_atoms_from_xyz(get_xyz_from_rdmol(rdmol))
    atoms.info["smiles_input"] = smiles
    if rdmol.HasProp("_IQCCanonicalSmiles"):
        atoms.info["canonical_smiles"] = rdmol.GetProp("_IQCCanonicalSmiles")
    if rdmol.HasProp("_IQCForceField"):
        atoms.info["rdkit_force_field"] = rdmol.GetProp("_IQCForceField")
    if rdmol.HasProp("_IQCLowestEnergy"):
        atoms.info["rdkit_conformer_energy"] = rdmol.GetDoubleProp("_IQCLowestEnergy")
    return atoms


def convert_extended_xyz_to_standard(file_path):
    """
    Reads an extended XYZ format file and converts it to standard XYZ format.

    Args:
        file_path (str): Path to the extended XYZ file.

    Returns:
        str: Standard XYZ format string.
    """
    with open(file_path, "r") as file:
        lines = file.readlines()

    # Extract the number of atoms and comment line (first two lines of XYZ format)
    num_atoms = int(lines[0].strip())
    comment_line = lines[1].strip()

    # Extract the atom data lines
    atom_data = lines[2 : num_atoms + 2]

    # Convert to standard XYZ format
    standard_xyz_lines = [
        f"{line.split()[0]} {line.split()[1]} {line.split()[2]} {line.split()[3]}"
        for line in atom_data
    ]

    # Combine into the standard XYZ format string
    standard_xyz_string = f"{num_atoms}\n{comment_line}\n" + "\n".join(
        standard_xyz_lines
    )

    return standard_xyz_string


def get_canonical_smiles(molecule):
    """
    Generates the canonical SMILES string for an RDKit molecule object.

    Args:
        molecule (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        str: The canonical SMILES string, or None if the molecule is invalid.
    """
    if molecule is None:
        return None
    try:
        # Use RDKit's MolToSmiles to generate canonical SMILES
        return Chem.MolToSmiles(molecule, canonical=True)
    except Exception as e:
        print(f"Error generating SMILES: {e}")
        return None


def get_bonding_info(molecule):
    """
    Extracts bonding information for a given RDKit molecule object.

    Args:
        molecule (rdkit.Chem.Mol): The RDKit molecule object.

    Returns:
        list of dict: A list of dictionaries where each dictionary represents a bond with:
            - 'atom1': Index of the first atom in the bond
            - 'atom2': Index of the second atom in the bond
            - 'atom1_symbol': Symbol of the first atom
            - 'atom2_symbol': Symbol of the second atom
            - 'bond_type': Type of the bond (e.g., SINGLE, DOUBLE)
    """
    if molecule is None:
        return []

    bonding_info = []
    for bond in molecule.GetBonds():
        atom1_idx = bond.GetBeginAtomIdx()
        atom2_idx = bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        bonding_info.append(
            {
                "atom1": atom1_idx,
                "atom2": atom2_idx,
                "atom1_symbol": molecule.GetAtomWithIdx(atom1_idx).GetSymbol(),
                "atom2_symbol": molecule.GetAtomWithIdx(atom2_idx).GetSymbol(),
                "bond_type": bond_type.name,  # Get bond type as a string
            }
        )

    return bonding_info


def ase2rdkit2(atoms):
    """
    Converts an ASE Atoms object to an RDKit Mol object using RDKit's MolFromXYZBlock.

    Parameters:
        atoms (ase.Atoms): ASE Atoms object.

    Returns:
        rdkit.Chem.Mol: RDKit molecule object, or None if conversion fails.
    """
    try:
        # Export ASE object to XYZ format and read with RDKit
        xyz = f"{len(atoms)}\n\n"
        for atom, position in zip(atoms.get_chemical_symbols(), atoms.get_positions()):
            xyz += f"{atom} {position[0]} {position[1]} {position[2]}\n"

        raw_mol = Chem.MolFromXYZBlock(xyz)
        if raw_mol is None:
            raise ValueError("Failed to convert XYZ to RDKit molecule.")

        mol = Chem.Mol(raw_mol)

        # Set initial charges to 0 if missing
        if not atoms.has("initial_charges"):
            atoms.set_initial_charges([0] * len(atoms))

        # Determine bonds using RDKit
        Chem.rdDetermineBonds.DetermineBonds(
            mol, charge=int(sum(atoms.get_initial_charges()))
        )
        return mol
    except Exception as e:
        print(f"Error in ase2rdkit: {e}")
        return None


def ase2rdkit_manual(atoms, bond_threshold=1.2):
    """
    Converts an ASE Atoms object to an RDKit Mol object, manually determining bonds.

    Args:
        atoms (ase.Atoms): ASE Atoms object.
        bond_threshold (float): Bonding threshold (multiplied by covalent radii).

    Returns:
        rdkit.Chem.Mol: RDKit molecule object.
    """
    try:
        # Create an editable molecule
        mol = Chem.RWMol()

        # Add atoms to the molecule
        atomic_numbers = atoms.get_atomic_numbers()
        for atomic_num in atomic_numbers:
            mol.AddAtom(Chem.Atom(int(atomic_num)))  # Convert to Python int

        # Get positions and determine bonds
        positions = atoms.get_positions()
        radii = np.array(
            [Chem.GetPeriodicTable().GetRcovalent(int(z)) for z in atomic_numbers]
        )

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i >= j:
                    continue  # Avoid double-counting bonds

                distance = np.linalg.norm(pos_i - pos_j)
                max_bond_distance = bond_threshold * (radii[i] + radii[j])

                if distance <= max_bond_distance:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        # Convert to RDKit Mol object
        mol = mol.GetMol()
        rdmolops.SanitizeMol(mol)  # Sanitize the molecule

        return mol
    except Exception as e:
        print(f"Error in ase2rdkit_manual: {e}")
        return None


def ase2rdkit_with_bond_orders(atoms, bond_threshold=1.2):
    """
    Converts an ASE Atoms object to an RDKit Mol object, with higher-order bond detection.

    Args:
        atoms (ase.Atoms): ASE Atoms object.
        bond_threshold (float): Base bonding threshold (multiplied by covalent radii).

    Returns:
        rdkit.Chem.Mol: RDKit molecule object.
    """
    try:
        # Create an editable molecule
        mol = Chem.RWMol()

        # Add atoms to the molecule
        atomic_numbers = atoms.get_atomic_numbers()
        for atomic_num in atomic_numbers:
            mol.AddAtom(Chem.Atom(int(atomic_num)))  # Convert to Python int

        # Get positions and determine bonds
        positions = atoms.get_positions()
        radii = np.array(
            [Chem.GetPeriodicTable().GetRcovalent(int(z)) for z in atomic_numbers]
        )

        # Define bond distance thresholds for different bond types
        bond_multipliers = {
            Chem.BondType.SINGLE: 1.2,
            Chem.BondType.DOUBLE: 1.1,
            Chem.BondType.TRIPLE: 1.0,
        }

        for i, pos_i in enumerate(positions):
            for j, pos_j in enumerate(positions):
                if i >= j:
                    continue  # Avoid double-counting bonds

                distance = np.linalg.norm(pos_i - pos_j)
                bond_added = False

                # Check bond type based on thresholds
                for bond_type, multiplier in bond_multipliers.items():
                    max_bond_distance = multiplier * (radii[i] + radii[j])
                    if distance <= max_bond_distance:
                        mol.AddBond(i, j, bond_type)
                        bond_added = True
                        break  # Add the highest-order bond that fits

                if not bond_added and distance <= bond_threshold * (
                    radii[i] + radii[j]
                ):
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        # Convert to RDKit Mol object
        mol = mol.GetMol()
        rdmolops.SanitizeMol(mol)  # Sanitize the molecule

        return mol
    except Exception as e:
        print(f"Error in ase2rdkit_with_bond_orders: {e}")
        return None


def get_canonical_smiles_from_atoms(atoms):
    """Convert ASE Atoms object to canonical SMILES string.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        str: Canonical SMILES string, or None if conversion fails
    """
    try:
        # Convert atoms to RDKit mol using manual bond detection
        rdmol = ase2rdkit_manual(atoms)

        if rdmol is None:
            return None

        # Generate canonical SMILES
        return Chem.MolToSmiles(rdmol, canonical=True)

    except Exception as e:
        print(f"Error converting atoms to SMILES: {e}")
        return None


atoms2smiles = get_canonical_smiles_from_atoms
rdmol2smiles = get_canonical_smiles


def ase_atoms_to_tuple(atoms):
    """
    Convert ASE Atoms object to a tuple of atom symbols and coordinates.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        tuple: A tuple containing tuples of atom symbols and their coordinates
    """
    atom_data = []
    for i in range(len(atoms)):
        atom_symbol = atoms.get_chemical_symbols()[i]
        coordinates = tuple(atoms.get_positions()[i])
        atom_data.append((atom_symbol, coordinates))
    return tuple(atom_data)


atoms2tuple = ase_atoms_to_tuple


def get_symmetry_info(atoms):
    """
    Calculate symmetry information for an ASE Atoms object using pymatgen and spglib.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        tuple: (str, int): (Point group symbol, rotational symmetry number)
    """
    try:
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer
        from pymatgen.io.ase import AseAtomsAdaptor

        aaa = AseAtomsAdaptor()
        molecule = aaa.get_molecule(atoms)
        pga = PointGroupAnalyzer(molecule)
        symmetrynumber = pga.get_rotational_symmetry_number()
        pointgroup = pga.get_pointgroup()
        return (pointgroup, symmetrynumber)
    except Exception as e:
        logging.warning(f"Error getting symmetry info: {e}")
        return ("C1", 1)


def ase_to_rdkit_mol(atoms):
    """
    Convert ASE Atoms to an RDKit Mol.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        rdkit.Chem.Mol: RDKit molecule object
    """
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    # Create an RDKit molecule with no bonds first.
    mol = Chem.RWMol()
    atom_indices = []
    for sym in symbols:
        a = Chem.Atom(sym)
        idx = mol.AddAtom(a)
        atom_indices.append(idx)

    # Add bonds heuristically based on distance criteria
    # This is naive and may need improvement:
    # For a real system, you'd implement a proper bond-guessing function.
    conf = Chem.Conformer(len(symbols))
    for i, pos in enumerate(positions):
        conf.SetAtomPosition(i, tuple(pos))
    mol.AddConformer(conf)

    # Simple bond guess: if distance < some cutoff, add a bond
    # You would refine these rules depending on your chemistry.
    distance_matrix = atoms.get_all_distances(mic=False)
    # Rough covalent radius guess table for common elements:
    cov_radii = {
        "H": 0.31,
        "C": 0.76,
        "N": 0.71,
        "O": 0.66,
        "F": 0.57,
        "Cl": 1.02,
        # Add more if needed
    }
    for i in range(len(symbols)):
        for j in range(i + 1, len(symbols)):
            ri = cov_radii.get(symbols[i], 0.7)
            rj = cov_radii.get(symbols[j], 0.7)
            cutoff = ri + rj + 0.4  # some margin
            if distance_matrix[i, j] < cutoff:
                mol.AddBond(i, j, Chem.BondType.SINGLE)

    rd_mol = mol.GetMol()
    # Try to sanitize molecule
    Chem.SanitizeMol(rd_mol)
    # Generate 3D coords if needed (usually we already have them from ASE)
    # But we can just keep as is, since we have coordinates from ASE.
    return rd_mol


def atoms2xyz(atoms):
    """
    Converts an ASE Atoms object to an XYZ string.

    Args:
        atoms (ase.Atoms): The ASE Atoms object to be converted.

    Returns:
        str: A string in XYZ format.
    """
    try:
        # Get the number of atoms
        num_atoms = len(atoms)

        # Create the header (number of atoms and a blank/comment line)
        xyz_str = f"{num_atoms}\n\n"

        # Add atom positions and symbols
        for symbol, position in zip(
            atoms.get_chemical_symbols(), atoms.get_positions()
        ):
            xyz_str += (
                f"{symbol} {position[0]:.8f} {position[1]:.8f} {position[2]:.8f}\n"
            )

        return xyz_str
    except Exception as e:
        print(f"Error in atoms_to_xyz: {e}")
        return ""


def decode_complex(dct):
    """
    Decode a complex number from a dictionary.

    Args:
        dct (dict): A dictionary containing "real" and "imag" keys.

    Returns:
        complex: The decoded complex number.
    """
    if "real" in dct and "imag" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


def get_total_electrons(atoms):
    """
    Get the total number of electrons from an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object

    Returns:
        int: Total number of electrons
    """
    atomic_numbers = atoms.get_atomic_numbers()

    # Sum up all electrons
    total_electrons = sum(atomic_numbers)

    return total_electrons


def get_multiplicity(atoms, multiplicity=None):
    """Resolve spin multiplicity (2S+1, integer >= 1).

    Defaults to 2 for an odd electron count (doublet), 1 for an even count
    (singlet). Pass an explicit `multiplicity` to override (e.g. 3 for
    triplet O2).
    """
    if multiplicity is None:
        return 2 if get_total_electrons(atoms) % 2 else 1
    multiplicity = int(multiplicity)
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    return multiplicity


def get_spin(atoms, multiplicity=None):
    """Return total spin S = (multiplicity - 1) / 2 (ASE thermo convention).

    If `multiplicity` is None, defaults from electron count parity (singlet
    for even, doublet for odd).

    Args:
        atoms: ASE Atoms object
        multiplicity (int, optional): Spin multiplicity 2S+1. When provided,
            overrides the parity-based default (e.g. for triplet O2 pass 3).

    Returns:
        float: Spin S (e.g. 0.0 for singlet, 0.5 for doublet, 1.0 for triplet).
    """
    return (get_multiplicity(atoms, multiplicity) - 1) / 2.0


def apply_spin_charge(atoms, calculator, multiplicity=None, charge=0):
    """Apply spin state and charge to `atoms` using the calculator's convention.

    User-facing convention is **multiplicity** 2S+1 (singlet=1, doublet=2,
    triplet=3, ...) and integer **charge**. Defaults: multiplicity from
    electron-count parity, charge=0.

    Translation per calculator:
      - XTB (`xtb.ase.calculator.XTB`): reads sums of
        `atoms.get_initial_charges()` and `atoms.get_initial_magnetic_moments()`.
        XTB's `uhf` is the number of unpaired electrons (= multiplicity - 1).
        We place the totals on atom 0, leave others at 0.
      - FAIRChem UMA (`fairchem.core.FAIRChemCalculator`): reads
        `atoms.info["charge"]` and `atoms.info["spin"]` where its `spin` is
        the spin multiplicity. We set `atoms.info["spin"] = multiplicity`.
      - MACE (`MACECalculator` from mace_mp) and ASE EMT: no spin/charge
        support; a warning is logged if non-default values are requested.

    Returns:
        int: Resolved multiplicity that was applied (after defaulting).
    """
    multiplicity = get_multiplicity(atoms, multiplicity)
    unpaired = multiplicity - 1
    charge = int(charge)
    calc_class = type(calculator).__name__
    n = len(atoms)

    if calc_class == "XTB":
        charges = [0.0] * n
        magmoms = [0.0] * n
        if n:
            charges[0] = float(charge)
            magmoms[0] = float(unpaired)
        atoms.set_initial_charges(charges)
        atoms.set_initial_magnetic_moments(magmoms)
    elif calc_class == "FAIRChemCalculator":
        atoms.info["charge"] = charge
        atoms.info["spin"] = multiplicity
    elif calc_class == "ORCA":
        parameters = getattr(calculator, "parameters", None)
        if parameters is not None:
            parameters["charge"] = charge
            parameters["mult"] = multiplicity
    elif calc_class in {"MACECalculator", "EMT"}:
        default_mult = 2 if get_total_electrons(atoms) % 2 else 1
        if charge != 0 or multiplicity != default_mult:
            logging.warning(
                "%s does not support spin/charge; ignoring charge=%d, "
                "multiplicity=%d.",
                calc_class,
                charge,
                multiplicity,
            )
    else:
        default_mult = 2 if get_total_electrons(atoms) % 2 else 1
        if charge != 0 or multiplicity != default_mult:
            logging.warning(
                "Spin/charge convention for calculator '%s' is unknown; "
                "values not applied (charge=%d, multiplicity=%d).",
                calc_class,
                charge,
                multiplicity,
            )
    return multiplicity


def get_inchikey(atoms):
    """Convert ASE Atoms object to InChIKey using RDKit.

    Args:
        atoms (ase.Atoms): ASE Atoms object to convert

    Returns:
        str: InChIKey string, or empty string if conversion fails
    """
    try:
        # Convert to RDKit mol using existing function
        mol = ase2rdkit_manual(atoms)
        if mol is None:
            return ""

        # Generate InChIKey
        inchikey = Chem.MolToInchiKey(mol)
        return inchikey

    except Exception as e:
        print(f"Error in atoms2inchikey: {e}")
        return ""


atoms2inchikey = get_inchikey


def _prepare_calculation(
    atoms, calculator=None, unique_name="", multiplicity=None, charge=0
):
    """
    Prepare atoms and calculator for a calculation.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator): The calculator instance to use.
                                                            If None, get_calculator() is called.
        unique_name (str): Unique name for the molecule
        multiplicity (int, optional): Spin multiplicity 2S+1. Defaults from
            electron-count parity (1 for even, 2 for odd).
        charge (int): Total molecular charge. Defaults to 0.

    Returns:
        tuple: (calculator, initial_data, results_dict)
    """
    if unique_name == "":
        unique_name = get_inchikey(atoms)

    # Get calculator if not provided
    if calculator is None:
        logging.debug("No calculator provided, getting default.")
        calculator = get_calculator()

    if calculator is None:
        # This should not happen if get_calculator raises RuntimeError correctly
        raise ValueError("Failed to obtain a valid calculator.")

    calc = calculator

    _assign_orca_work_directory(calc, unique_name)

    # Apply spin/charge in the convention expected by this calculator before
    # attaching it; XTB and FAIRChem read these from the atoms object.
    multiplicity = apply_spin_charge(
        atoms, calc, multiplicity=multiplicity, charge=charge
    )

    # Get initial data
    initial_smiles = atoms2smiles(atoms)
    initial_xyz = atoms2xyz(atoms)
    try:
        initial_sym, initial_sym_number = get_symmetry_info(atoms)
    except Exception as e:
        logging.warning(
            f"Error getting symmetry number: {e}. Using default value of 1."
        )
        initial_sym = "C1"  # Default point group
        initial_sym_number = 1

    # Set calculator and get initial energy
    atoms.calc = calc
    try:
        initial_energy = atoms.get_potential_energy()
    except Exception as e:
        context = _orca_failure_context(calc)
        message = f"Failed to get initial potential energy with {str(calc)}: {e}{context}"
        logging.error(message)
        if context:
            raise RuntimeError(message) from e
        raise

    # Prepare results dictionary
    results = {
        "number_of_atoms": len(atoms),
        "number_of_electrons": get_total_electrons(atoms),
        "spin": get_spin(atoms, multiplicity),
        "multiplicity": multiplicity,
        "charge": charge,
        "formula": atoms.get_chemical_formula(mode="hill"),
        "unique_name": unique_name,
        "initial_smiles": initial_smiles,
        "initial_xyz": initial_xyz,
        "initial_symmetry": str(initial_sym),
        "initial_sym_number": initial_sym_number,
        "initial_energy_eV": initial_energy,
        "warnings": [],
        "error": "",
        "calculator_name": str(calc),
        "model": getattr(calc, "model_name", ""),
    }

    return calc, results


def run_single_point(
    atoms,
    calculator=None,
    unique_name="",
    multiplicity=None,
    charge=0,
):
    """
    Run a single point energy calculation for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        unique_name (str): Unique name for the molecule
        multiplicity (int, optional): Spin multiplicity 2S+1. Defaults from
            electron-count parity. Translated per-calculator by `apply_spin_charge`.
        charge (int): Total molecular charge. Defaults to 0.

    Returns:
        tuple: A tuple containing the atoms and a dictionary with calculated properties
    """
    logging.info(f"Starting single point calculation for {unique_name}")

    calc, results = _prepare_calculation(
        atoms, calculator, unique_name, multiplicity=multiplicity, charge=charge
    )

    start_time = time.time()
    # Energy is already calculated in _prepare_calculation. Forces are useful
    # when present, but single-point energy should not fail for energy-only
    # backends or ORCA inputs without ENGRAD.
    energy = results["initial_energy_eV"]
    results["energy_eV"] = energy
    results["forces"] = []
    try:
        implemented_properties = getattr(calc, "implemented_properties", None)
        if implemented_properties is not None and "forces" not in implemented_properties:
            warning = (
                f"Forces are not implemented by {calc}; single-point energy was saved."
            )
            results["warnings"].append(warning)
            logging.warning(warning)
        else:
            forces = atoms.get_forces()
            results["forces"] = forces.tolist()
    except (PropertyNotImplementedError, PropertyNotPresent) as e:
        warning = (
            f"Forces were not available for the single-point calculation ({e}); "
            "single-point energy was saved."
        )
        if calc.__class__.__name__ == "ORCA":
            warning += " For ORCA forces, include ENGRAD in orcasimpleinput."
        results["warnings"].append(warning)
        logging.warning(warning)
    except Exception as e:
        error = f"Error in single point calculation: {e}\n"
        results["error"] += error
        logging.error(error)
    finally:
        results["calc_time"] = time.time() - start_time

    if not results["error"]:
        logging.debug(
            f"Single point calculation completed in {results['calc_time']} seconds."
        )

    logging.info(f"Single point calculation for {unique_name} completed")
    return atoms, results


def run_optimization(
    atoms,
    calculator=None,
    fmax=0.001,
    unique_name="",
    max_steps=500,
    trajectory=None,
    save_geometry=False,
    output_dir=None,
    multiplicity=None,
    charge=0,
):
    """
    Run geometry optimization for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        fmax (float): Maximum force for geometry optimization
        unique_name (str): Unique name for the molecule
        max_steps (int): Maximum number of optimization steps
        trajectory (str): Path to save trajectory file
        save_geometry (bool): Whether to save the final optimized geometry to xyz file
        output_dir (str, optional): Directory for optimized geometry output.
        multiplicity (int, optional): Spin multiplicity 2S+1. Defaults from
            electron-count parity. Translated per-calculator by `apply_spin_charge`.
        charge (int): Total molecular charge. Defaults to 0.

    Returns:
        tuple: A tuple containing the optimized atoms and a dictionary with calculated properties
    """
    _ensure_orca_engrad_for_forces(calculator or atoms.calc)
    calc, results = _prepare_calculation(
        atoms, calculator, unique_name, multiplicity=multiplicity, charge=charge
    )
    logging.info(f"Starting geometry optimization for {unique_name} with {str(calc)}")
    # Log optimization parameters
    logging.debug(f"Optimization parameters: fmax={fmax}, max_steps={max_steps}")

    # Add optimization-specific fields
    results.update(
        {
            "opt_smiles": "",
            "opt_xyz": "",
            "opt_sym_number": 0,
            "opt_energy_eV": 0,
            "smiles_changed": None,
            "opt_time": 0,
            "opt_steps": 0,
            "opt_converged": False,
            "opt_forces": [],
            "trajectory_file": trajectory if trajectory else "",
            "optimized_geometry_file": "",
        }
    )

    try:
        if trajectory:
            trajectory_dir = os.path.dirname(os.path.abspath(trajectory))
            if trajectory_dir:
                os.makedirs(trajectory_dir, exist_ok=True)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Ensure atoms positions and cell are float64 before optimization
        # This can help prevent type mismatches within the optimizer
        atoms.positions = atoms.positions.astype(np.float64)
        if atoms.pbc.any():  # Check if periodic boundary conditions are set
            atoms.cell = atoms.cell.astype(np.float64)

        start_time = time.time()
        dyn = BFGS(atoms, trajectory=trajectory)
        converged = dyn.run(fmax=fmax, steps=max_steps)
        results["opt_time"] = time.time() - start_time
        results["opt_steps"] = dyn.get_number_of_steps()
        results["opt_converged"] = converged
        results["opt_forces"] = atoms.get_forces().tolist()

        if trajectory:
            logging.info(f"Optimization trajectory saved to {trajectory}")

        logging.debug(f"Optimization completed in {results['opt_time']} seconds.")
    except Exception as e:
        error = f"Error in optimization: {e}"
        # Add more context to the error log
        logging.error(
            f"Optimization failed for {unique_name} using {str(calc)}. Error: {e}"
        )
        if "did not contain a loop with signature matching types" in str(e):
            logging.error(
                "Potential type mismatch detected. Check calculator dtype and input geometry precision."
            )
        results["error"] = error
        # No need to log again here, already logged above
    if not results["error"]:
        try:
            opt_sym, opt_sym_number = get_symmetry_info(atoms)
        except Exception as e:
            logging.warning(f"Error getting symmetry info: {e}")
            opt_sym = "C1"
            opt_sym_number = 1
        results["opt_smiles"] = atoms2smiles(atoms)
        results["opt_energy_eV"] = atoms.get_potential_energy()
        results["opt_xyz"] = atoms2xyz(atoms)
        results["opt_sym"] = str(opt_sym)
        results["opt_sym_number"] = opt_sym_number
        results["smiles_changed"] = results["initial_smiles"] != results["opt_smiles"]

        # Save optimized geometry if requested and optimization converged
        if save_geometry and results["opt_converged"]:
            try:
                geometry_file = f"{unique_name}_optimized.xyz"
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    geometry_file = os.path.join(output_dir, geometry_file)
                write(geometry_file, atoms, format="xyz")
                results["optimized_geometry_file"] = geometry_file
                logging.info(f"Optimized geometry saved to {geometry_file}")
            except Exception as e:
                logging.warning(f"Failed to save optimized geometry: {e}")

    logging.info(f"Geometry optimization for {unique_name} completed")
    return atoms, results


def _optimization_extra_params(params):
    """Return only params accepted by run_optimization beyond common arguments."""

    return {
        key: params[key]
        for key in ("max_steps", "output_dir")
        if key in params
    }


def run_vibrations(
    atoms,
    calculator=None,
    optimize=True,
    unique_name="",
    vib_dir=None,
    indices=None,
    fmax=0.01,
    delta=0.01,
    max_trans_rot=100,
    max_vib_imag=50,
    trajectory=None,
    save_geometry=False,
    multiplicity=None,
    charge=0,
    **params,
):
    """
    Run vibrational frequency calculations for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        unique_name (str): Unique name for the molecule
        vib_dir (str, optional): Directory to store vibration files. Defaults to None.
        indices (list): List of atom indices to include in vibration calculation
        fmax (float): Maximum force for geometry optimization
        delta (float): Displacement for finite difference calculation
        max_trans (float): Max abs. value in cm-1 for translation modes
        max_rot (float): Max abs. value in cm-1 for rotation modes
        max_vib_imag (float): Max abs. value for the imaginary part in cm-1 for vibrational modes
        trajectory (str): Path to save trajectory file during optimization
        save_geometry (bool): Whether to save the final optimized geometry to xyz file

    Returns:
        tuple: A tuple containing the atoms and a dictionary with calculated properties
    """
    results = {
        "warnings": [],
        "error": "",
    }  # Initialize results dictionary with warning and error fields

    try:
        if calculator is None:
            if atoms.calc is None:
                calc, calc_results = _prepare_calculation(
                    atoms, calculator, unique_name, multiplicity=multiplicity, charge=charge
                )
                results.update(calc_results)  # Update results with calculator results
            else:
                # use the calculator from the atoms object
                calc = atoms.calc
        else:
            calc = calculator
    except Exception as e:
        error = f"Error in calculator preparation: {e}"
        results["error"] += error
        logging.error(error)
        return None, results

    _assign_orca_work_directory(calc, unique_name, purpose="vib")
    _ensure_orca_engrad_for_forces(calc)
    logging.debug(
        f"Starting vibrational calculations for {unique_name} with {str(calc)}"
    )
    if optimize:
        atoms, opt_results = run_optimization(
            atoms,
            calculator=calc,
            unique_name=unique_name,
            fmax=fmax,
            trajectory=trajectory,
            save_geometry=save_geometry,
            multiplicity=multiplicity,
            charge=charge,
            **_optimization_extra_params(params),
        )
        if opt_results.get("error"):  # Use get() to safely check for error
            results["error"] += opt_results["error"]  # Append optimization error
            logging.error("Optimization failed, cannot proceed with vibrations.")
            return None, results
        results.update(opt_results)
    else:
        logging.warning(
            "No optimization requested, using given geometry for the vibrations."
        )

    try:
        start_time = time.time()
        vib_name = f"tmp_vib_{unique_name}"
        if vib_dir:
            os.makedirs(vib_dir, exist_ok=True)
            vib_name = os.path.join(vib_dir, vib_name)
        vib = Vibrations(atoms, name=vib_name, indices=indices, delta=delta)
        vib.run()
        vib_data = vib.get_vibrations()  # Get the VibrationsData object
        results["vib_time"] = time.time() - start_time

        # Get frequencies and energies from vib_data
        frequencies = vib_data.get_frequencies()  # cm^-1
        logging.debug(f"Frequencies in cm^-1: {frequencies}")

        logging.debug(
            f"Vibrational energies (ev) and modes (3N, N, 3) as tuple (energies, modes)"
        )
        try:
            vib_energies, vib_modes = vib_data.get_energies_and_modes()  # eV
        except (AttributeError, TypeError, ValueError):
            vib_energies = (
                vib_data.get_energies() if hasattr(vib_data, "get_energies") else []
            )
            vib_modes = []
        results["frequencies_cm^-1"] = (
            frequencies.tolist() if hasattr(frequencies, "tolist") else frequencies
        )
        results["vib_energies"] = (
            vib_energies.tolist() if hasattr(vib_energies, "tolist") else vib_energies
        )  # Store energies for thermo
        results["vib_modes"] = (
            vib_modes.tolist() if hasattr(vib_modes, "tolist") else vib_modes
        )
        try:
            # In-memory text file:
            buffer = io.BytesIO()
            f = io.TextIOWrapper(buffer, encoding="utf-8", write_through=True)

            # Write Jmol XYZ+vectors into the in-memory "file"
            vib._write_jmol(f)  # <-- accepts any TextIO-like object

            # Rewind and read the string
            f.seek(0)
            xyz_with_modes = buffer.getvalue().decode("utf-8")

            # Clean up
            f.close()
            buffer.close()
            results["jmol_vib_modes_xyz"] = xyz_with_modes
        except Exception as e:
            warning = f"Could not export Jmol vibrational modes: {e}"
            logging.warning(warning)
            results["warnings"].append(warning)
        nrot = 3
        if is_linear_by_inertia(atoms):
            nrot = 2
        # Check translational and rotational modes
        if np.any(np.abs(frequencies[: 3 + nrot]) > max_trans_rot):
            logging.warning(
                f"Translational or rotational modes are too high: {frequencies[:3+nrot]}"
            )
            results["warnings"].append("Translational or rotational modes are too high")
        img_freqs = [f for f in frequencies[3 + nrot :] if abs(f.imag) > max_vib_imag]
        results["number_of_imaginary"] = len(img_freqs)
        results["vibrational_frequencies_cm^-1"] = [
            f.real for f in frequencies[3 + nrot :]
        ]

        logging.debug(
            f"Vibrational analysis completed in {results['vib_time']} seconds."
        )
        logging.debug(vib.summary())
        vib.clean()
    except AttributeError as ae:
        # Catch specific errors related to missing methods
        error = f"Error accessing vibration data (possibly ASE version issue?): {ae}\n"
        results["error"] += error
        logging.error(error)
    except Exception as e:
        error = f"Error in vibrational analysis: {e}\n"
        results["error"] += error
        logging.error(error)

    logging.info(f"Vibrational analysis for {unique_name} completed")
    return atoms, results


def run_ir(
    atoms,
    calculator=None,
    optimization_calculator=None,
    vibration_calculator=None,
    dipole_calculator=None,
    optimize=True,
    unique_name="",
    vib_dir=None,
    indices=None,
    fmax=0.01,
    delta=0.01,
    trajectory=None,
    save_geometry=False,
    ir_spectrum_start=300,
    ir_spectrum_end=4000,
    sparse_spectrum=False,
    intensity_threshold=0.0,
    multiplicity=None,
    charge=0,
    **params,
):
    """
    Run infrared spectrum calculations for an ASE Atoms object.

    Three independent calculators may be supplied: one for geometry
    optimization, one for the vibrational forces (Hessian / normal modes),
    and one for the dipole moments (IR intensities). Each falls back to
    `calculator` if not provided. A typical mixed workflow is MACE for
    optimization+vibrations and an electronic-structure backend for dipoles.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator: Default calculator used as a fallback for any of the
            three roles below if they are not specified.
        optimization_calculator: Calculator used for the optional geometry
            optimization step.
        vibration_calculator: Calculator used to compute forces at each
            finite-difference displacement (Hessian / frequencies).
        dipole_calculator: Calculator used to compute the dipole moment at
            each displacement (IR intensities).
        optimize (bool): Whether to optimize geometry before IR calculation.
        unique_name (str): Unique name for the molecule
        vib_dir (str, optional): Directory to store IR files. Defaults to None.
        indices (list): List of atom indices to include in IR calculation
        fmax (float): Maximum force for geometry optimization
        delta (float): Displacement for finite difference calculation
        trajectory (str): Path to save trajectory file during optimization
        save_geometry (bool): Whether to save the final optimized geometry to xyz file
        ir_spectrum_start (float): Start of IR spectrum range in cm^-1
        ir_spectrum_end (float): End of IR spectrum range in cm^-1
        sparse_spectrum (bool): Store only points above threshold instead of full spectrum
        intensity_threshold (float): Absolute intensity cutoff used with sparse_spectrum

    Returns:
        tuple: A tuple containing the atoms and a dictionary with calculated properties
    """
    results = {
        "warnings": [],
        "error": "",
        "spectrum_frequencies": [],
        "spectrum_frequencies_units": "cm-1",
        "spectrum_intensities": [],
        "spectrum_intensities_units": "D/A^2 amu^-1",
    }

    opt_calc = optimization_calculator or calculator
    vib_calc = vibration_calculator or calculator
    dip_calc = dipole_calculator or calculator

    try:
        if opt_calc is None and vib_calc is None and dip_calc is None:
            if atoms.calc is None:
                fallback, calc_results = _prepare_calculation(
                    atoms,
                    None,
                    unique_name,
                    multiplicity=multiplicity,
                    charge=charge,
                )
                results.update(calc_results)
            else:
                fallback = atoms.calc
            opt_calc = vib_calc = dip_calc = fallback
        else:
            for role_calc in (opt_calc, vib_calc, dip_calc):
                if role_calc is not None:
                    apply_spin_charge(
                        atoms,
                        role_calc,
                        multiplicity=multiplicity,
                        charge=charge,
                    )
    except Exception as e:
        error = f"Error in calculator preparation: {e}"
        results["error"] += error
        logging.error(error)
        return None, results

    seen_calculators = set()
    for role, role_calc in (
        ("ir_opt", opt_calc),
        ("ir_vib", vib_calc),
        ("ir_dipole", dip_calc),
    ):
        if role_calc is None or id(role_calc) in seen_calculators:
            continue
        seen_calculators.add(id(role_calc))
        _assign_orca_work_directory(role_calc, unique_name, purpose=role)

    if optimize:
        _ensure_orca_engrad_for_forces(opt_calc)
    _ensure_orca_engrad_for_forces(vib_calc)

    results["calculator_optimization"] = str(opt_calc) if opt_calc else ""
    results["calculator_vibration"] = str(vib_calc) if vib_calc else ""
    results["calculator_dipole"] = str(dip_calc) if dip_calc else ""

    # Fail fast if the dipole calculator does not actually expose dipoles.
    # Every ASE Calculator subclass has the `get_dipole_moment` *method*, but
    # it raises PropertyNotImplementedError unless 'dipole' is in
    # `implemented_properties`. Without this check, ASE only complains after
    # optimization and the first displacement, with a cryptic
    # "dipole property not implemented".
    if dip_calc is not None:
        dip_props = getattr(dip_calc, "implemented_properties", None)
        if dip_props is not None and "dipole" not in dip_props:
            error = (
                f"Dipole calculator {type(dip_calc).__name__} does not "
                "implement the 'dipole' property; IR intensities cannot be "
                "computed. MACE and EMT do not support dipoles. Use a "
                "calculator that does (e.g. xtb), or pass an instance via the "
                "Python API.\n"
            )
            results["error"] += error
            logging.error(error)
            return None, results

    logging.debug(
        f"Starting IR calculations for {unique_name}: "
        f"opt={opt_calc}, vib={vib_calc}, dip={dip_calc}"
    )

    if optimize:
        if opt_calc is None:
            error = (
                "Optimization requested but no optimization_calculator or "
                "calculator was provided.\n"
            )
            results["error"] += error
            logging.error(error)
            return None, results
        atoms, opt_results = run_optimization(
            atoms,
            calculator=opt_calc,
            unique_name=unique_name,
            fmax=fmax,
            trajectory=trajectory,
            save_geometry=save_geometry,
            multiplicity=multiplicity,
            charge=charge,
            **_optimization_extra_params(params),
        )
        if opt_results.get("error"):
            results["error"] += opt_results["error"]
            logging.error("Optimization failed, cannot proceed with IR.")
            return None, results
        results.update(opt_results)
    else:
        logging.warning("No optimization requested, using given geometry for IR.")

    if vib_calc is None or dip_calc is None:
        error = (
            "IR requires both a vibration calculator (for forces) and a "
            "dipole calculator. Provide vibration_calculator/dipole_calculator "
            "or a fallback calculator.\n"
        )
        results["error"] += error
        logging.error(error)
        return None, results

    try:
        from ase.vibrations import Infrared
    except ImportError as e:
        error = f"Infrared module is not available in ASE: {e}\n"
        results["error"] += error
        logging.error(error)
        return None, results

    class _DualCalcInfrared(Infrared):
        """Infrared with separate calculators for forces and dipole moments."""

        def __init__(self, atoms, vib_calc, dip_calc, **kwargs):
            super().__init__(atoms, **kwargs)
            self._vib_calc = vib_calc
            self._dip_calc = dip_calc

        def calculate(self, atoms, disp):
            results = {"forces": self._vib_calc.get_forces(atoms)}
            if self.ir:
                results["dipole"] = self._dip_calc.get_dipole_moment(atoms)
            return results

    try:
        start_time = time.time()
        ir_name = f"tmp_ir_{unique_name}"
        if vib_dir:
            os.makedirs(vib_dir, exist_ok=True)
            ir_name = os.path.join(vib_dir, ir_name)

        # Attach the vibration calculator so atoms.calc is non-None for any
        # ASE internals that inspect it; calculate() uses both calculators
        # explicitly via the override above.
        atoms.calc = vib_calc
        ir = _DualCalcInfrared(
            atoms, vib_calc, dip_calc, name=ir_name, indices=indices, delta=delta
        )
        ir.clean()
        ir.run()

        freq_intensity = ir.get_spectrum(
            start=ir_spectrum_start,
            end=ir_spectrum_end,
        )
        frequencies = np.asarray(freq_intensity[0], dtype=float)
        intensities = np.asarray(freq_intensity[1], dtype=float)
        if sparse_spectrum:
            mask = np.abs(intensities) > intensity_threshold
            frequencies_to_store = frequencies[mask]
            intensities_to_store = intensities[mask]
            results["spectrum_storage"] = "sparse"
            results["spectrum_points_total"] = int(frequencies.size)
            results["spectrum_points_stored"] = int(frequencies_to_store.size)
            results["intensity_threshold"] = float(intensity_threshold)
        else:
            frequencies_to_store = frequencies
            intensities_to_store = intensities
            results["spectrum_storage"] = "full"
            results["spectrum_points_total"] = int(frequencies.size)
            results["spectrum_points_stored"] = int(frequencies.size)

        results["spectrum_frequencies"] = frequencies_to_store.tolist()
        results["spectrum_intensities"] = intensities_to_store.tolist()
        results["ir_time"] = time.time() - start_time

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            ax.plot(frequencies, intensities)
            ax.set_xlabel("Frequency (cm^-1)")
            ax.set_ylabel("Intensity (a.u.)")
            ax.set_title("Infrared Spectrum")
            ax.grid(True)

            ir_plot_path = f"{ir_name}_spectrum.png"
            fig.savefig(ir_plot_path, format="png", dpi=300)
            plt.close(fig)

            results["ir_plot"] = os.path.abspath(ir_plot_path)
        except Exception as e:
            warning = f"IR spectrum plot could not be generated: {e}"
            logging.warning(warning)
            results["warnings"].append(warning)

        results["normal_mode_data"] = (
            "Normal modes saved as .traj files with prefix "
            f"{os.path.abspath(ir_name)}"
        )
    except Exception as e:
        error = f"Error in IR analysis: {e}\n"
        results["error"] += error
        logging.error(error)

    logging.info(f"IR analysis for {unique_name} completed")
    return atoms, results


def run_thermo(
    atoms,
    calculator=None,
    ignore_imag_modes=True,
    unique_name="",
    trajectory=None,
    save_geometry=False,
    multiplicity=None,
    charge=0,
    **params,
):
    """
    Run thermochemistry calculations for an ASE Atoms object.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator (ase.calculators.calculator.Calculator, optional): Calculator instance. Defaults to None (uses get_calculator).
        ignore_imag_modes (bool): Whether to ignore imaginary vibrational modes
        unique_name (str): Unique name for the molecule
        trajectory (str): Path to save trajectory file during optimization
        save_geometry (bool): Whether to save the final optimized geometry to xyz file
        **opt_params: Additional keyword arguments passed to run_optimization.

    Returns:
        tuple: A tuple containing the thermochemistry results and a dictionary with calculated properties
    """

    atoms, results = run_vibrations(
        atoms,
        calculator=calculator,
        optimize=True,
        unique_name=unique_name,
        trajectory=trajectory,
        save_geometry=save_geometry,
        multiplicity=multiplicity,
        charge=charge,
        **params,
    )
    if results["error"]:
        logging.error(
            "Vibrational analysis failed, cannot proceed with thermochemistry.\n"
        )
        return None, results

    thermo = None

    try:
        start_time = time.time()
        # Get energies directly from vib_results dictionary
        vib_energies = results.get("vib_energies", None)
        if vib_energies is None:
            # This case indicates an issue in run_vibrations not storing energies
            logging.error(
                "Vibrational energies not found in vibration results. Cannot calculate thermo properties."
            )
            results["error"] += "Missing vibrational energies for thermochemistry.\n"
            return None, results

        # Check for imaginary frequencies if not ignoring them
        if not ignore_imag_modes:
            n_imag = results.get("number_of_imaginary", 0)
            if n_imag > 0:
                error = f"Imaginary vibrational energies are present: ({n_imag} imaginary modes).\n"
                results["error"] = error
                logging.error(error)
                return None, results

        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry=get_geometry_type(atoms),
            atoms=atoms,
            potentialenergy=atoms.get_potential_energy(),
            spin=get_spin(atoms, results.get("multiplicity")),
            symmetrynumber=results.get("opt_sym_number", 1),  # Use optimized symmetry
            ignore_imag_modes=ignore_imag_modes,
        )
        results["thermo_time"] = time.time() - start_time
        results["G_eV"] = thermo.get_gibbs_energy(temperature=298.15, pressure=101325.0)
        results["H_eV"] = thermo.get_enthalpy(temperature=298.15)
        results["S_eV/K"] = thermo.get_entropy(temperature=298.15, pressure=101325.0)
        results["E_ZPE_eV"] = thermo.get_ZPE_correction()
        logging.debug(
            f"Thermochemistry calculations completed in {results['thermo_time']} seconds."
        )
    except Exception as e:
        error = f"Error in thermochemistry calculations: {e}\n"
        results["error"] += error
        logging.error(error)
        return None, results

    logging.info(f"Thermochemistry calculation for {unique_name} completed")
    return thermo, results


_XYZ_COMMENT_KV_RE = re.compile(
    r"(?:^|[\s,;])(multiplicity|mult|uhf|charge|chrg|q)\s*[=:]\s*(-?\d+)",
    re.IGNORECASE,
)


def parse_multiplicity_charge_from_comment(comment):
    """Extract spin multiplicity and charge from an XYZ comment line.

    Recognized tokens (case-insensitive, separator `=` or `:`):
      - `multiplicity=N` or `mult=N`: spin multiplicity 2S+1 (preferred form)
      - `uhf=N`: number of unpaired electrons (XTB spelling); converted to N+1
      - `charge=N`, `chrg=N`, or `q=N`: total molecular charge

    Returns:
        tuple[Optional[int], Optional[int]]: (multiplicity, charge). Either
        may be None if not present. If both `multiplicity` and `uhf` are given
        and disagree, `multiplicity` wins (a warning is logged).
    """
    if not comment:
        return None, None
    mult = None
    uhf = None
    charge = None
    for key, value in _XYZ_COMMENT_KV_RE.findall(comment):
        key = key.lower()
        n = int(value)
        if key in ("multiplicity", "mult"):
            mult = n
        elif key == "uhf":
            uhf = n
        elif key in ("charge", "chrg", "q"):
            charge = n
    if mult is None and uhf is not None:
        if uhf < 0:
            raise ValueError(f"uhf must be >= 0, got {uhf}")
        mult = uhf + 1
    elif mult is not None and uhf is not None and (uhf + 1) != mult:
        logging.warning(
            "XYZ comment has both multiplicity=%d and uhf=%d (mismatched); "
            "using multiplicity=%d.",
            mult,
            uhf,
            mult,
        )
    if mult is not None and mult < 1:
        raise ValueError(f"multiplicity must be >= 1, got {mult}")
    return mult, charge


def get_atoms_from_xyz(xyz, parallel=False, index=-1):
    """
    Generate ASE Atoms object from XYZ input.

    The XYZ comment line is parsed for `multiplicity`/`uhf` and `charge`/`q`
    tokens (see `parse_multiplicity_charge_from_comment`). When found, they
    are stored under `atoms.info["multiplicity"]` and `atoms.info["charge"]`
    for downstream consumption.

    Args:
        xyz (str): Either path to an XYZ file or XYZ content as string
        parallel (bool): Whether to use parallel reading
        index (int): Index of the configuration to read
    Returns:
        ase.Atoms: ASE Atoms object
    """
    logging.debug(f"Reading atoms from XYZ input: {xyz}")
    if os.path.isfile(xyz):
        with open(xyz, "r") as fh:
            xyz_text = fh.read()
        atoms = read(io.StringIO(xyz_text), format="xyz", parallel=parallel, index=index)
    elif isinstance(xyz, str):
        xyz_text = xyz
        atoms = read(io.StringIO(xyz_text), format="xyz", parallel=parallel, index=index)
    else:
        logging.error(f"Invalid input type for ase.io.read: {type(xyz)}")
        return None

    try:
        comment = _extract_xyz_comment(xyz_text, index)
        mult, charge = parse_multiplicity_charge_from_comment(comment)
        if mult is not None:
            atoms.info["multiplicity"] = mult
        if charge is not None:
            atoms.info["charge"] = charge
    except Exception as e:
        logging.debug(f"Could not parse multiplicity/charge from XYZ comment: {e}")

    logging.debug(f"Successfully read atoms from {xyz}")
    return atoms


def _extract_xyz_comment(xyz_text, index=-1):
    """Return the comment (line 2) of the requested XYZ frame, or empty string."""
    lines = xyz_text.splitlines()
    frames = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        try:
            n = int(line)
        except ValueError:
            i += 1
            continue
        comment = lines[i + 1] if i + 1 < len(lines) else ""
        frames.append(comment)
        i += 2 + n
    if not frames:
        return ""
    try:
        return frames[index]
    except IndexError:
        return frames[-1]


xyz2atoms = get_atoms_from_xyz


def is_linear_by_inertia(atoms, tol=1e-3):
    """
    Determine if a molecule is linear by checking its moments of inertia.

    Parameters:
    atoms : ase.Atoms
        The molecule to check.
    tol : float
        Tolerance for treating a moment of inertia as zero.

    Returns:
    bool : True if molecule is linear, False otherwise.
    """
    moments = sorted(atoms.get_moments_of_inertia())  # ascending order
    if moments[0] > tol:
        return False  # First moment should be (near) zero
    return abs(moments[1] - moments[2]) / max(moments[1], moments[2]) < tol


def get_geometry_type(atoms):
    """Return the geometry type (monatomic, linear, nonlinear) of the atoms object"""
    if len(atoms) == 1:
        return "monatomic"
    elif len(atoms) == 2:
        return "linear"
    else:
        symmetry, symmetry_number = get_symmetry_info(atoms)
        if "*" in symmetry:
            return "linear"
        else:
            return "nonlinear"
