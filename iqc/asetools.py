import json
import logging
import os
import re
import shutil
import socket
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
import numpy as np
import ase
from ase import Atoms, build
from ase.calculators.calculator import PropertyNotImplementedError, PropertyNotPresent
from ase.calculators.emt import EMT
from ase.io import read, write
from ase.optimize import BFGS, FIRE, LBFGS
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


def _orca_requests_engrad(calculator):
    """Return True when the ORCA input already asks for the gradient."""

    parameters = getattr(calculator, "parameters", None) or {}
    simpleinput = str(parameters.get("orcasimpleinput") or "")
    return bool(re.search(r"(?i)(^|\s)engrad($|\s)", simpleinput))


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


MACE_POLAR_DEFAULT_MODEL = "polar-1-m"
MACE_POLAR_MODEL_URLS = {
    "polar-1-s": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-S.model",
    "polar-1-m": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-M.model",
    "polar-1-l": "https://github.com/ACEsuit/mace-foundations/releases/download/mace_polar_1/MACE-POLAR-1-L.model",
}


def _default_mace_cache_dir():
    """Return MACE's checkpoint cache directory without requiring a new MACE API."""

    try:
        from mace.tools.utils import get_cache_dir

        return Path(get_cache_dir()).expanduser()
    except Exception:
        cache_home = os.environ.get("XDG_CACHE_HOME")
        if cache_home:
            return Path(cache_home).expanduser() / "mace"
        return Path.home().expanduser() / ".cache" / "mace"


def _mace_polar_cache_name(checkpoint_url):
    """Match MACE's sanitized checkpoint cache filename convention."""

    return "".join(
        c for c in os.path.basename(str(checkpoint_url)) if c.isalnum() or c in "_"
    )


def _mace_polar_cached_model_path(model):
    """Return the cached checkpoint path for a MACE-Polar model key or URL.

    Picks the path that *mace itself* uses (sanitized URL basename) when it
    exists on disk; only falls back to ``polar_model_paths`` (which reports
    the un-sanitized name like ``MACE-POLAR-1-S.model``) if the sanitized
    cache is missing. This avoids the spurious-download trap: the previous
    order checked the un-sanitized path first, found nothing (mace caches
    under the sanitized name), and acquired the download lock every time —
    which on Aurora compute nodes (no github) just hangs.
    """

    model = str(model)

    sanitized_path = None
    checkpoint_url = MACE_POLAR_MODEL_URLS.get(model)
    if checkpoint_url is None and model.startswith("https:"):
        checkpoint_url = model
    if checkpoint_url is not None:
        sanitized_path = _default_mace_cache_dir() / _mace_polar_cache_name(checkpoint_url)
    if sanitized_path is not None and sanitized_path.exists():
        return sanitized_path

    try:
        from mace.calculators import foundations_models

        polar_model_paths = getattr(foundations_models, "polar_model_paths", {})
        cached_path = polar_model_paths.get(model)
        if cached_path is not None:
            return Path(cached_path).expanduser()
    except Exception:
        pass

    return sanitized_path


def _download_mace_polar_checkpoint(model):
    """Download or locate a MACE-Polar checkpoint using MACE's own resolver."""

    from mace.calculators import foundations_models

    return foundations_models.download_mace_polar_checkpoint(model)


@contextmanager
def _file_lock(lock_path, poll_interval=0.25):
    """Directory-based lock for filesystems where `flock` is unreliable."""

    lock_path = Path(lock_path).expanduser()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stale_seconds = float(os.environ.get("IQC_MACE_POLAR_LOCK_STALE_SECONDS", 43200))
    acquired = False
    while not acquired:
        try:
            lock_path.mkdir()
            acquired = True
        except FileExistsError:
            if lock_path.is_file():
                lock_path.unlink()
                continue
            try:
                age = max(0.0, time.time() - lock_path.stat().st_mtime)
            except OSError:
                age = 0.0
            if age > stale_seconds:
                logging.warning(
                    "Removing stale MACE-Polar checkpoint lock older than %.0f s: %s",
                    stale_seconds,
                    lock_path,
                )
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            time.sleep(poll_interval)

    owner_file = lock_path / "owner"
    try:
        owner_file.write_text(
            f"host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n",
            encoding="utf-8",
        )
    except OSError:
        pass

    try:
        yield
    finally:
        shutil.rmtree(lock_path, ignore_errors=True)


def _ensure_mace_polar_model_cached(model):
    """Avoid MPI download races by prefetching a missing checkpoint under a lock."""

    cached_path = _mace_polar_cached_model_path(model)
    if cached_path is None or Path(str(model)).expanduser().exists():
        return
    if cached_path.exists():
        return

    lock_path = cached_path.with_name(f"{cached_path.name}.iqc-download.lock")
    logging.info(
        "Waiting for MACE-Polar checkpoint cache lock: %s", lock_path
    )
    with _file_lock(lock_path):
        if cached_path.exists():
            return
        logging.info(
            "Caching MACE-Polar checkpoint '%s' at %s", model, cached_path
        )
        _download_mace_polar_checkpoint(model)


def _enable_mace_polar_metadata(calculator, model_name):
    """Tag a MACE-Polar calculator with IQC-specific behavior hints."""

    calculator.model_name = model_name
    calculator._iqc_spin_charge_convention = "mace_polar"
    calculator._iqc_calculator_family = "mace-polar"

    implemented = getattr(calculator, "implemented_properties", None)
    if implemented is not None and "dipole" not in implemented:
        calculator.implemented_properties = list(implemented) + ["dipole"]
    return calculator


def _get_mace_polar_calculator(**kwargs):
    """Initialize the MACE-Polar ASE calculator when the loader is available."""

    _patch_e3nn_mace_compatibility()
    from mace.calculators import mace_polar

    polar_kwargs = {
        "model": MACE_POLAR_DEFAULT_MODEL,
        "default_dtype": "float64",
        "device": "cpu",
        **kwargs,
    }
    model_name = polar_kwargs["model"]
    _ensure_mace_polar_model_cached(model_name)
    calculator = mace_polar(**polar_kwargs)
    _enable_mace_polar_metadata(calculator, model_name)
    logging.info(f"Using MACE-Polar calculator with arguments: {polar_kwargs}")
    return calculator


UMA_DEFAULT_MODEL_BY_SIZE = {
    "s": "uma-s-1p2",
    "m": "uma-m-1p1",
}
UMA_TASKS = {"omol", "omat", "odac"}
UMA_PREDICTOR_KWARGS = {
    "atom_refs",
    "cache_dir",
    "device",
    "form_elem_refs",
    "inference_settings",
    "overrides",
    "seed",
    "workers",
}
UMA_HOSTED_PREDICTOR_KWARGS = UMA_PREDICTOR_KWARGS - {"atom_refs", "form_elem_refs"}
UMA_LOCAL_PREDICTOR_KWARGS = UMA_PREDICTOR_KWARGS - {"cache_dir", "seed"}
UMA_SUPPORTED_DEVICES = {"cpu", "cuda", "xpu"}


class CalculatorConfigurationError(RuntimeError):
    """Raised when calculator parameters are valid YAML but unsupported."""


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


def _validate_uma_device(kwargs):
    """Fail early for FAIRChem UMA device values IQC cannot support."""

    if "device" not in kwargs:
        return

    device = str(kwargs["device"])
    if device in UMA_SUPPORTED_DEVICES:
        return

    supported = ", ".join(sorted(UMA_SUPPORTED_DEVICES))
    raise CalculatorConfigurationError(
        f"FAIRChem UMA does not support device={device!r} in this IQC path. "
        f"Use one of: {supported}. On Intel GPU systems, install the XPU "
        "PyTorch wheel to avoid NVIDIA dependencies, but set UMA "
        "calculator_params.device to 'cpu' unless your FAIRChem version "
        "explicitly supports XPU."
    )


def _get_uma_calculator(name, **kwargs):
    """Initialize a FAIRChem UMA calculator with IQC's compact name aliases."""

    _validate_uma_device(kwargs)

    from fairchem.core import FAIRChemCalculator, pretrained_mlip

    predictor_name, task = _parse_uma_calculator_name(name)
    uma_kwargs = dict(kwargs)
    checkpoint_path = uma_kwargs.pop("checkpoint_path", None)
    predictor_name = uma_kwargs.pop("model", predictor_name)
    task = uma_kwargs.pop("task_name", task)
    predictor_kwargs = {
        key: uma_kwargs.pop(key)
        for key in list(uma_kwargs)
        if key in UMA_PREDICTOR_KWARGS
    }

    model_path = Path(str(checkpoint_path or predictor_name)).expanduser()
    if checkpoint_path or model_path.exists():
        from fairchem.core.units.mlip_unit import load_predict_unit

        local_predictor_kwargs = {
            key: value
            for key, value in predictor_kwargs.items()
            if key in UMA_LOCAL_PREDICTOR_KWARGS
        }
        ignored_keys = sorted(set(predictor_kwargs) - set(local_predictor_kwargs))
        if ignored_keys:
            logging.info(
                "Ignoring hosted UMA predictor option(s) for local checkpoint: %s",
                ignored_keys,
            )
        predictor = load_predict_unit(model_path, **local_predictor_kwargs)
        predictor_name = str(model_path)
        logging.info(
            "Loaded UMA predictor from local checkpoint: %s", predictor_name
        )
    else:
        hosted_predictor_kwargs = {
            key: value
            for key, value in predictor_kwargs.items()
            if key in UMA_HOSTED_PREDICTOR_KWARGS
        }
        predictor = pretrained_mlip.get_predict_unit(
            predictor_name, **hosted_predictor_kwargs
        )

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


class NumericalForceCalculator:
    """Wrap an energy-only calculator so it exposes finite-difference forces.

    Some electronic-structure backends (ExaChem, PySCF correlated methods such
    as CCSD(T)) return only the energy. IQC's geometry optimization and
    vibrational analysis are force-driven, so those methods cannot be used for
    thermochemistry as-is. This wrapper adds forces via ASE's finite-difference
    machinery (``ase.calculators.fd.calculate_numerical_forces``) — the same
    numerical-force routine behind ``FiniteDifferenceCalculator`` — while
    delegating energy (and dipole, if the inner calculator provides it) to the
    wrapped calculator.

    Unlike ``ase.calculators.fd.FiniteDifferenceCalculator``, this wrapper does
    **not** attempt a numerical stress: that requires a periodic cell and a
    defined volume, which an isolated molecule does not have (``get_volume``
    would raise). Stress is only computed when the atoms actually have a full
    3D cell.

    IQC metadata (``_iqc_spin_charge_convention``, ``_iqc_calculator_family``,
    ``model_name``) and the inner ``parameters`` mapping are exposed on the
    wrapper so ``apply_spin_charge`` and result-labeling keep working: mutating
    ``wrapper.parameters`` mutates the inner calculator's parameters (same dict).
    """

    def __init__(self, calc, eps_disp: float = 0.01, force_consistent: bool = False):
        from ase.calculators.calculator import BaseCalculator

        self.calc = calc
        self.eps_disp = float(eps_disp)
        self.force_consistent = bool(force_consistent)
        self.results: dict = {}
        self.atoms = None
        # (geometry key, forces) of the last finite-difference evaluation.
        # ASE optimizers call get_forces() several times per step at the same
        # geometry; without this cache every call re-runs the full 6*N-single-
        # point stencil (measured ~3x the necessary single points per step).
        self._fd_forces_cache = None
        # Expose the inner calculator's IQC hints + parameter dict.
        self.parameters = getattr(calc, "parameters", {})
        self._iqc_calculator_family = getattr(calc, "_iqc_calculator_family", None)
        self._iqc_spin_charge_convention = getattr(
            calc, "_iqc_spin_charge_convention", ""
        )
        self.model_name = getattr(calc, "model_name", getattr(calc, "label", "calc"))
        base_props = list(getattr(calc, "implemented_properties", ["energy"]) or ["energy"])
        if "forces" not in base_props:
            base_props.append("forces")
        self.implemented_properties = base_props

    def __getattr__(self, item):
        # Delegate anything not defined here (e.g. get_dipole_moment helpers)
        # to the wrapped calculator.
        return getattr(self.__dict__["calc"], item)

    def get_potential_energy(self, atoms=None, force_consistent=False):
        target = atoms if atoms is not None else self.atoms
        return self.calc.get_potential_energy(target)

    @staticmethod
    def _geometry_key(atoms):
        return (
            atoms.get_positions().tobytes(),
            atoms.numbers.tobytes(),
            atoms.cell.array.tobytes(),
            atoms.pbc.tobytes(),
            atoms.get_initial_charges().tobytes(),
            atoms.get_initial_magnetic_moments().tobytes(),
        )

    def get_forces(self, atoms=None):
        from ase.calculators.fd import calculate_numerical_forces

        target = atoms if atoms is not None else self.atoms
        key = self._geometry_key(target)
        if self._fd_forces_cache is not None and self._fd_forces_cache[0] == key:
            return self._fd_forces_cache[1].copy()
        work = target.copy()
        work.calc = self.calc
        forces = calculate_numerical_forces(
            work, eps=self.eps_disp, force_consistent=self.force_consistent
        )
        self._fd_forces_cache = (key, forces)
        return forces.copy()

    def get_property(self, name, atoms=None, allow_calculation=True):
        target = atoms if atoms is not None else self.atoms
        if name == "forces":
            return self.get_forces(target)
        if name in ("energy", "free_energy"):
            return self.get_potential_energy(target)
        return self.calc.get_property(name, target, allow_calculation)

    def calculate(self, atoms=None, properties=("energy", "forces"), system_changes=None):
        target = atoms if atoms is not None else self.atoms
        self.atoms = target.copy()
        self.results = {}
        self.results["energy"] = self.calc.get_potential_energy(target)
        inner = getattr(self.calc, "results", {}) or {}
        if "free_energy" in inner:
            self.results["free_energy"] = inner["free_energy"]
        else:
            self.results["free_energy"] = self.results["energy"]
        if "forces" in properties:
            self.results["forces"] = self.get_forces(target)
        if "dipole" in inner:
            self.results["dipole"] = inner["dipole"]
        # Surface energy-component / timing metadata from the inner calculator
        # (e.g. scf/ccsd/(t) breakdown) so the orchestrator can persist it.
        for key, value in inner.items():
            if key.endswith("_eV") or key.endswith("_s"):
                self.results[key] = value

    def reset(self):
        self.results = {}
        self._fd_forces_cache = None
        if hasattr(self.calc, "reset"):
            self.calc.reset()

    def __repr__(self):
        return f"NumericalForceCalculator({self.calc!r}, eps_disp={self.eps_disp})"


def _wrap_with_numerical_forces(calc, eps_disp: float = None):
    """Return ``calc`` with finite-difference forces if it lacks native ones.

    Calculators that already implement forces (MACE, PySCF DFT/HF, VASP) are
    returned unchanged. Energy-only calculators (ExaChem, PySCF CCSD(T)) are
    wrapped so the force-driven optimization/vibration code can use them. The
    displacement defaults to the value stashed on the calculator
    (``_iqc_fd_eps``, set by get_calculator) or 0.01 Å.
    """
    if calc is None:
        return calc
    props = getattr(calc, "implemented_properties", []) or []
    if "forces" in props:
        return calc
    if eps_disp is None:
        eps_disp = float(getattr(calc, "_iqc_fd_eps", 0.01))
    logging.info(
        "Wrapping %s with finite-difference forces (eps_disp=%.3g Å).",
        type(calc).__name__,
        eps_disp,
    )
    return NumericalForceCalculator(calc, eps_disp=eps_disp)


def get_calculator(name="mace", **kwargs):
    """Initializes and returns the specified ASE calculator.

    Args:
        name (str): The name of the calculator ('mace', 'mace-polar', 'xtb',
                    'emt', 'uma', 'uma-s-omol', 'uma-s-omat', 'uma-s-odac',
                    'uma-m-omol', 'uma-m-omat', 'uma-m-odac', 'orca', or
                    'exachem').
        **kwargs: Additional keyword arguments passed to the calculator constructor.

    Returns:
        ase.calculators.calculator.Calculator: The initialized calculator instance.

    Raises:
        RuntimeError: if the requested calculator is unknown, its dependency is
            missing, or initialization fails. This function never silently
            substitutes a different calculator (e.g. EMT/MACE) — doing so would
            change the level of theory under the user. Callers that want a
            fallback must catch the error and choose one explicitly.
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
                # Retrying without dispersion changes the level of theory
                # (energies lose the D3 contribution), so it must never be
                # silent: skip the retry when dispersion was already off, and
                # record the downgrade in model_name so persisted rows are
                # distinguishable from dispersion-on results.
                if not mace_kwargs.get("dispersion"):
                    raise
                logging.warning(
                    f"Failed to initialize MACE with dispersion={mace_kwargs.get('dispersion')}: {str(e)}. Trying with dispersion=False."
                )
                mace_kwargs["dispersion"] = False
                calculator = mace_mp(**mace_kwargs)
                calculator.model_name = f"{mace_kwargs['model']}-no-dispersion"
                logging.warning(
                    "MACE initialized WITHOUT dispersion after the dispersion "
                    "setup failed; energies exclude the D3 correction and rows "
                    "are labeled model=%s.",
                    calculator.model_name,
                )
        except ImportError as e:
            raise RuntimeError(
                "MACE not found. Install with 'pip install mace' (or `iqc[mace]`)."
            ) from e
        except Exception as e:
            raise RuntimeError(f"MACE initialization failed: {e}") from e

    elif name == "mace-polar":
        try:
            calculator = _get_mace_polar_calculator(**kwargs)
        except ImportError as e:
            message = (
                f"MACE-Polar could not be imported: {e}. Electrostatic MACE "
                "requires MACE from the latest main branch and "
                "graph_electrostatics/graph_longrange."
            )
            logging.error(message)
            raise RuntimeError(message) from e
        except Exception as e:
            message = f"MACE-Polar initialization failed: {e}"
            logging.error(message)
            raise RuntimeError(message) from e

    elif name == "xtb":
        try:
            from xtb.ase.calculator import XTB

            xtb_kwargs = {"method": "GFN2-xTB", **kwargs}
            calculator = XTB(**xtb_kwargs)
            logging.info(f"Using XTB calculator with arguments: {xtb_kwargs}")
        except ImportError as e:
            raise RuntimeError(
                "XTB not found. Install with 'pip install xtb' or 'pip install iqc[xtb]'."
            ) from e
        except Exception as e:
            raise RuntimeError(f"XTB initialization failed: {e}") from e

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
        except CalculatorConfigurationError as e:
            logging.error(str(e))
            raise RuntimeError(str(e)) from e
        except ImportError as e:
            raise RuntimeError(
                f"FAIRChem UMA import failed: {e}. Install fairchem-core with "
                "compatible dependencies (see the MACE/UMA install workaround)."
            ) from e
        except Exception as e:
            raise RuntimeError(f"UMA initialization failed: {e}") from e

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
        except ImportError as e:
            raise RuntimeError(
                f"ASE ORCA calculator not available: {e}"
            ) from e
        except Exception as e:
            raise RuntimeError(f"ORCA initialization failed: {e}") from e

    elif name == "pyscf" or name.startswith("pyscf-"):
        # PySCF: DFT/HF (energy+forces+dipole) or MP2/CCSD/CCSD(T) (energy).
        # `pyscf-ccsd(t)` etc. select the method via the name suffix; a bare
        # `pyscf` takes its method from calculator_params (default PBE DFT).
        try:
            from iqc.pyscf_calc import PySCFCalculator

            pyscf_kwargs = dict(kwargs)
            fd_eps = float(pyscf_kwargs.pop("fd_eps", 0.01))
            if name.startswith("pyscf-"):
                pyscf_kwargs.setdefault("method", name.split("-", 1)[1])
            calculator = PySCFCalculator(**pyscf_kwargs)
            logging.info(
                "Using PySCF calculator (method=%s, xc=%s, basis=%s)",
                calculator.parameters.get("method"),
                calculator.parameters.get("xc"),
                calculator.parameters.get("basis"),
            )
            # Remember the FD step so the force-driven pipeline (opt/vibrations)
            # can add finite-difference forces for energy-only methods (CCSD(T)).
            calculator._iqc_fd_eps = fd_eps
        except ImportError as e:
            message = (
                f"PySCF calculator not importable: {e}. "
                "Install pyscf (pip install pyscf) or pick a different calculator."
            )
            logging.error(message)
            raise RuntimeError(message) from e
        except Exception as e:
            message = f"PySCF initialization failed: {e}"
            logging.error(message)
            raise RuntimeError(message) from e

    elif name == "vasp":
        # VASP configured to match the MACE-MP (MPtrj) training level: PBE,
        # ENCUT 520 eV, spin-polarized, standard PBE PAW potentials.
        try:
            from iqc.vasp_calc import get_vasp_calculator

            calculator = get_vasp_calculator(**kwargs)
            logging.info(
                "Using VASP calculator (%s)", getattr(calculator, "model_name", "vasp")
            )
        except ImportError as e:
            message = (
                f"VASP calculator not importable: {e}. Ensure ASE is installed "
                "and VASP is available on this system."
            )
            logging.error(message)
            raise RuntimeError(message) from e
        except Exception as e:
            message = f"VASP initialization failed: {e}"
            logging.error(message)
            raise RuntimeError(message) from e

    elif name == "exachem":
        # ExaChem is only ever requested explicitly (there is no implicit
        # exachem path in the codebase). Silently falling back to MACE would
        # change the level of theory under the user, so fail fast instead.
        try:
            from iqc.exachem import ExaChemCalculator

            exachem_kwargs = dict(kwargs)
            fd_eps = float(exachem_kwargs.pop("fd_eps", 0.01))
            calculator = ExaChemCalculator(**exachem_kwargs)
            logging.info(
                f"Using ExaChem calculator with method={calculator.parameters.get('method')}, "
                f"basis={calculator.parameters.get('basis')}, nproc={calculator.parameters.get('nproc')}"
            )
            # Remember the FD step; energy-only ExaChem gets finite-difference
            # forces from the force-driven pipeline (opt/vibrations) on demand.
            calculator._iqc_fd_eps = fd_eps
        except ImportError as e:
            message = (
                f"ExaChem calculator not importable: {e}. "
                "Install ExaChem and its Python bindings, or pick a different calculator."
            )
            logging.error(message)
            raise RuntimeError(message) from e
        except Exception as e:
            message = f"ExaChem initialization failed: {e}"
            logging.error(message)
            raise RuntimeError(message) from e

    else:
        raise RuntimeError(
            f"Unknown calculator {name!r}. Supported: mace, mace-polar, xtb, "
            f"emt, orca, exachem, pyscf, pyscf-<method>, vasp, uma, "
            f"uma-s-omol, uma-s-omat, uma-s-odac, "
            f"uma-m-omol, uma-m-omat, uma-m-odac."
        )

    if calculator is None:
        # All known-calculator branches now either set `calculator` or raise,
        # so reaching this should never happen. Keep as a defensive guard.
        raise RuntimeError(
            f"Calculator {name!r} returned None unexpectedly (no exception, "
            "but no instance either)."
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


def get_multiplicity(atoms, multiplicity=None, charge=0):
    """Resolve spin multiplicity (2S+1, integer >= 1).

    Defaults to 2 for an odd electron count (doublet), 1 for an even count
    (singlet), after applying the total molecular charge. Pass an explicit
    `multiplicity` to override (e.g. 3 for triplet O2).
    """
    if multiplicity is None:
        electron_count = get_total_electrons(atoms) - int(charge)
        return 2 if electron_count % 2 else 1
    multiplicity = int(multiplicity)
    if multiplicity < 1:
        raise ValueError(f"multiplicity must be >= 1, got {multiplicity}")
    return multiplicity


def get_spin(atoms, multiplicity=None, charge=0):
    """Return total spin S = (multiplicity - 1) / 2 (ASE thermo convention).

    If `multiplicity` is None, defaults from charged electron count parity
    (singlet for even, doublet for odd).

    Args:
        atoms: ASE Atoms object
        multiplicity (int, optional): Spin multiplicity 2S+1. When provided,
            overrides the parity-based default (e.g. for triplet O2 pass 3).
        charge (int): Total molecular charge used only for defaulting.

    Returns:
        float: Spin S (e.g. 0.0 for singlet, 0.5 for doublet, 1.0 for triplet).
    """
    return (get_multiplicity(atoms, multiplicity, charge=charge) - 1) / 2.0


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
      - MACE-Polar (`mace.calculators.mace_polar`): reads
        `atoms.info["charge"]` and `atoms.info["spin"]`; its spin input is
        the total spin S, so we set `atoms.info["spin"] = (multiplicity-1)/2`.
      - MACE (`MACECalculator` from mace_mp) and ASE EMT: no spin/charge
        support; a warning is logged if non-default values are requested.
      - ExaChem (`iqc.exachem.ExaChemCalculator`): writes charge and
        multiplicity into the calculator parameters which then surface in
        the SCF section of the generated JSON input.

    Returns:
        int: Resolved multiplicity that was applied (after defaulting).
    """
    charge = int(charge)
    multiplicity = get_multiplicity(atoms, multiplicity, charge=charge)
    unpaired = multiplicity - 1
    calc_class = type(calculator).__name__
    spin_charge_convention = getattr(calculator, "_iqc_spin_charge_convention", "")
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
    elif spin_charge_convention == "mace_polar":
        atoms.info["charge"] = charge
        atoms.info["spin"] = get_spin(atoms, multiplicity, charge=charge)
        atoms.info.setdefault("external_field", [0.0, 0.0, 0.0])
    elif calc_class == "ORCA":
        parameters = getattr(calculator, "parameters", None)
        if parameters is not None:
            parameters["charge"] = charge
            parameters["mult"] = multiplicity
    elif spin_charge_convention == "exachem":
        # ExaChem reads charge/multiplicity from the SCF section of the input
        # JSON it generates per call.
        parameters = getattr(calculator, "parameters", None)
        if parameters is not None:
            parameters["charge"] = charge
            parameters["multiplicity"] = multiplicity
            if parameters.get("scf_type") is None:
                parameters["scf_type"] = (
                    "restricted" if multiplicity == 1 else "unrestricted"
                )
    elif spin_charge_convention == "pyscf":
        # PySCF reads charge and spin (2S = unpaired electrons) from its
        # calculator parameters when it builds the gto.Mole per call.
        parameters = getattr(calculator, "parameters", None)
        if parameters is not None:
            parameters["charge"] = charge
            parameters["multiplicity"] = multiplicity
            parameters["spin"] = unpaired
    elif spin_charge_convention == "vasp":
        # VASP: spin via total magnetic moment (ISPIN=2 already set). Place all
        # unpaired electrons as an initial moment on atom 0; VASP relaxes it.
        # Non-zero charge requires an explicit NELECT (handled in the factory).
        magmoms = [0.0] * n
        if n and unpaired:
            magmoms[0] = float(unpaired)
        atoms.set_initial_magnetic_moments(magmoms)
        if charge != 0:
            logging.warning(
                "VASP: charge=%d requires an explicit NELECT via "
                "calculator_params; spin multiplicity=%d applied via magmoms.",
                charge,
                multiplicity,
            )
    elif calc_class in {"MACECalculator", "EMT"}:
        default_mult = get_multiplicity(atoms, charge=charge)
        if charge != 0 or multiplicity != default_mult:
            logging.warning(
                "%s does not support spin/charge; ignoring charge=%d, "
                "multiplicity=%d.",
                calc_class,
                charge,
                multiplicity,
            )
    else:
        default_mult = get_multiplicity(atoms, charge=charge)
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


def _to_serializable_array(value):
    """Convert tensor/array-like calculator results to JSON-friendly values."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_to_serializable_array(item) for item in value]
    return value


def _as_numpy_array(value):
    """Return a NumPy array view of tensor/array-like calculator results."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    try:
        return np.asarray(value)
    except Exception:
        return None


_EXACHEM_TOPLEVEL_FIELDS = (
    "scf_energy_eV",
    "mp2_correlation_eV",
    "ccsd_correlation_eV",
    "t_correction_eV",
    "total_energy_eV",
    "scf_time_s",
    "ccsd_time_s",
    "t_time_s",
    "basis",
    "scf_type",
    "method",
    "frozen_core",
)

# F4: artifact retention metadata. Always copied (when present in
# ``calc.results``) so the JSONL/SQLite schema stays stable across keep/no-keep
# runs — see ExaChemCalculator.EXACHEM_ARTIFACT_FIELDS for the contract.
_EXACHEM_ARTIFACT_FIELDS = (
    "run_dir",
    "artifact_manifest",
    "keep_artifacts",
    "artifact_archive",
)


def _store_calculator_observables(results, calc, prefix=""):
    """Persist dipoles and MACE-Polar density outputs when present."""

    calc_results = getattr(calc, "results", None)
    if not isinstance(calc_results, dict):
        return results

    # ExaChem-specific energy components and method metadata. These are
    # written by ``ExaChemCalculator._extract_components`` and surfaced here
    # as top-level keys so the orchestrator can persist them directly to
    # JSONL/SQLite (without burying them inside ``exachem_output``).
    if getattr(calc, "_iqc_calculator_family", None) == "exachem":
        for key in _EXACHEM_TOPLEVEL_FIELDS:
            if key in calc_results:
                # Use the prefix only when the caller is gathering a snapshot
                # of pre-task ("initial_") or post-opt ("opt_") state. For the
                # final single-point pass (prefix=""), expose the bare names.
                results[f"{prefix}{key}"] = calc_results[key]
        # F4: surface run_dir + per-artifact manifest so the orchestrator can
        # write absolute paths / sha256s into the per-row record. F9 fills in
        # artifact_archive after a successful tar.gz step; until then it stays
        # null. Behaviour for plain runs (keep_artifacts=False) is preserved
        # because the calculator writes empty list / null defaults itself.
        for key in _EXACHEM_ARTIFACT_FIELDS:
            if key in calc_results:
                results[f"{prefix}{key}"] = calc_results[key]

    if "dipole" in calc_results:
        results[f"{prefix}dipole"] = _to_serializable_array(calc_results["dipole"])

    density = _as_numpy_array(calc_results.get("density_coefficients"))
    if density is not None and density.ndim >= 2 and density.shape[-1] >= 4:
        results[f"{prefix}density_coefficients"] = _to_serializable_array(density)
        results[f"{prefix}partial_charges"] = _to_serializable_array(density[:, 0])
        results[f"{prefix}partial_dipoles"] = _to_serializable_array(
            density[:, [3, 1, 2]]
        )

    spin_density = _as_numpy_array(calc_results.get("spin_charge_density"))
    if (
        spin_density is not None
        and spin_density.ndim >= 3
        and spin_density.shape[1] >= 2
        and spin_density.shape[2] >= 1
    ):
        spin_up = spin_density[:, 0, 0]
        spin_down = spin_density[:, 1, 0]
        results[f"{prefix}spin_charge_density"] = _to_serializable_array(spin_density)
        results[f"{prefix}partial_spin_up_charges"] = _to_serializable_array(spin_up)
        results[f"{prefix}partial_spin_down_charges"] = _to_serializable_array(
            spin_down
        )
        results[f"{prefix}partial_spin_charges"] = _to_serializable_array(
            spin_up - spin_down
        )

    return results


class InvalidGeometryError(ValueError):
    """Raised when an input geometry fails a pre-flight sanity check."""


def _validate_geometry(
    atoms,
    unique_name="",
    min_distance=None,
    max_extent=None,
):
    """Reject obviously corrupt geometries before handing them to a calculator.

    Catches inputs that have historically triggered uncatchable GPU page
    faults or O(GiB) host-memory blowups inside MACE/matscipy (atoms on top
    of each other, NaN coordinates, molecules sprawling across hundreds of
    Angstroms). Thresholds can be overridden via environment variables so
    operators can tighten or loosen them per run.

    Args:
        atoms (ase.Atoms): geometry to validate.
        unique_name (str): label used in the error message.
        min_distance (float, optional): minimum interatomic distance in Å.
            Defaults to ``IQC_MIN_INTERATOMIC_DIST`` (env) or 0.3 Å.
        max_extent (float, optional): maximum bounding-box edge in Å.
            Defaults to ``IQC_MAX_GEOMETRY_EXTENT`` (env) or 100 Å.

    Raises:
        InvalidGeometryError: if the geometry fails any check.
    """
    if min_distance is None:
        min_distance = float(os.environ.get("IQC_MIN_INTERATOMIC_DIST", 0.3))
    if max_extent is None:
        max_extent = float(os.environ.get("IQC_MAX_GEOMETRY_EXTENT", 100.0))

    positions = np.asarray(atoms.get_positions(), dtype=float)
    if positions.size == 0:
        raise InvalidGeometryError(
            f"Geometry '{unique_name}' has zero atoms"
        )
    if not np.all(np.isfinite(positions)):
        raise InvalidGeometryError(
            f"Geometry '{unique_name}' contains non-finite coordinates (NaN/Inf)"
        )

    if len(positions) >= 2:
        diffs = positions[:, None, :] - positions[None, :, :]
        d2 = np.einsum("ijk,ijk->ij", diffs, diffs)
        np.fill_diagonal(d2, np.inf)
        dmin = float(np.sqrt(d2.min()))
        if dmin < min_distance:
            i, j = np.unravel_index(np.argmin(d2), d2.shape)
            raise InvalidGeometryError(
                f"Geometry '{unique_name}' has degenerate atom pair "
                f"({int(i)}, {int(j)}) separated by {dmin:.4g} Å < "
                f"{min_distance:g} Å (set IQC_MIN_INTERATOMIC_DIST to override)"
            )

    extent = positions.max(axis=0) - positions.min(axis=0)
    extent_max = float(extent.max())
    if extent_max > max_extent:
        raise InvalidGeometryError(
            f"Geometry '{unique_name}' bounding box {extent.round(2).tolist()} Å "
            f"exceeds limit {max_extent:g} Å (set IQC_MAX_GEOMETRY_EXTENT to override)"
        )


def validate_physical_results(results, atoms=None):
    """Flag nonphysical quantities in a result dict *without terminating*.

    Detects the failure modes seen in production (BFGS energy blow-ups to
    ~1e56 eV, ZPE/frequency overflow, negative entropy, NaN/Inf) so downstream
    aggregation can exclude them instead of silently averaging garbage. This
    never raises and never drops the row: on any violation it sets
    ``results['nonphysical'] = True`` and appends human-readable reasons to
    ``results['validation_messages']`` and ``results['warnings']``, and logs at
    ERROR level. It is idempotent — safe to call after each stage and again
    centrally — because messages are de-duplicated and the flag is recomputed.

    Thresholds are overridable via environment variables:
      ``IQC_MAX_ENERGY_PER_ATOM_EV`` (default 1e4),
      ``IQC_MAX_ZPE_PER_ATOM_EV`` (default 1.0),
      ``IQC_MAX_FREQ_CM`` (default 8000).

    Args:
        results (dict): result dict produced by a run_* function (mutated).
        atoms (ase.Atoms, optional): used only to recover the atom count when
            ``number_of_atoms`` is missing from ``results``.

    Returns:
        dict: the same ``results`` object, mutated in place.
    """
    n_atoms = results.get("number_of_atoms")
    if not n_atoms and atoms is not None:
        n_atoms = len(atoms)
    n_atoms = int(n_atoms) if n_atoms else 1

    max_e = float(os.environ.get("IQC_MAX_ENERGY_PER_ATOM_EV", 1e4)) * n_atoms
    max_zpe = float(os.environ.get("IQC_MAX_ZPE_PER_ATOM_EV", 1.0)) * n_atoms
    max_freq = float(os.environ.get("IQC_MAX_FREQ_CM", 8000.0))

    msgs = []

    def _num(key):
        value = results.get(key)
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # Energies: finite and within a generous per-atom magnitude bound.
    for key in ("initial_energy_eV", "opt_energy_eV", "energy_eV", "G_eV", "H_eV"):
        value = _num(key)
        if value is None:
            continue
        if not np.isfinite(value):
            msgs.append(f"{key} is non-finite ({results.get(key)!r})")
        elif abs(value) > max_e:
            msgs.append(
                f"{key}={value:.3e} eV exceeds |E| <= {max_e:.3e} eV "
                f"({n_atoms} atoms)"
            )

    # Zero-point energy: finite, non-negative, bounded.
    zpe = _num("E_ZPE_eV")
    if zpe is not None:
        if not np.isfinite(zpe):
            msgs.append(f"E_ZPE_eV is non-finite ({results.get('E_ZPE_eV')!r})")
        elif zpe < 0:
            msgs.append(f"E_ZPE_eV={zpe:.3e} eV is negative")
        elif zpe > max_zpe:
            msgs.append(f"E_ZPE_eV={zpe:.3e} eV exceeds {max_zpe:.3g} eV")

    # Entropy: finite and non-negative.
    entropy = _num("S_eV/K")
    if entropy is not None:
        if not np.isfinite(entropy):
            msgs.append(f"S_eV/K is non-finite ({results.get('S_eV/K')!r})")
        elif entropy < 0:
            msgs.append(f"S_eV/K={entropy:.3e} eV/K is negative")

    # Vibrational frequencies: finite and bounded in magnitude.
    freqs = results.get("vibrational_frequencies_cm^-1")
    if freqs is not None and len(freqs):
        arr = np.asarray(freqs, dtype=float)
        if not np.all(np.isfinite(arr)):
            msgs.append("vibrational_frequencies_cm^-1 contains non-finite values")
        elif np.max(np.abs(arr)) > max_freq:
            msgs.append(
                f"max|frequency|={np.max(np.abs(arr)):.3e} cm^-1 exceeds "
                f"{max_freq:g} cm^-1"
            )

    # Imaginary modes are informational (a real TS/under-optimized minimum),
    # not by themselves nonphysical; surface them as a flag for triage.
    try:
        if int(results.get("number_of_imaginary") or 0) > 0:
            results["has_imaginary"] = True
    except (TypeError, ValueError):
        pass

    existing = list(results.get("validation_messages", []))
    new_msgs = [m for m in msgs if m not in existing]
    if new_msgs:
        results["validation_messages"] = existing + new_msgs
        warnings = results.get("warnings")
        if isinstance(warnings, list):
            warnings.extend(new_msgs)
        else:
            results["warnings"] = list(new_msgs)
        label = results.get("unique_name") or results.get("_unique_name") or ""
        logging.error(
            "Nonphysical result(s) detected for %s: %s", label, "; ".join(new_msgs)
        )
    results["nonphysical"] = bool(results.get("validation_messages"))
    return results


def check_suspicious_no_op(results):
    """Warn if an optimization task looks like it never actually ran.

    This is the exact signature that let a silently-substituted calculator (the
    2026-06 UMA-on-XPU jobs that fell back to MACE) pass as a real result:
    ``opt_steps == 0``, sub-50 ms ``opt_time``, and ``opt_energy_eV`` identical
    to ``initial_energy_eV``. A genuinely pre-converged geometry is legal, so
    this only warns (sets ``results['suspicious_no_op'] = True``) and never
    fails the row. Non-optimization tasks (no ``opt_steps``) are ignored.
    """
    if "opt_steps" not in results:
        return results
    try:
        steps = int(results.get("opt_steps"))
        opt_time = float(results.get("opt_time"))
        initial_e = results.get("initial_energy_eV")
        opt_e = results.get("opt_energy_eV")
        if (
            steps == 0
            and opt_time < 0.05
            and initial_e is not None
            and opt_e is not None
            and float(initial_e) == float(opt_e)
        ):
            msg = (
                "suspicious no-op optimization: 0 steps, "
                f"opt_time={opt_time:.4g}s, opt_energy==initial_energy — the "
                "requested calculator may not have actually run"
            )
            results["suspicious_no_op"] = True
            warnings = results.get("warnings")
            if isinstance(warnings, list):
                if msg not in warnings:
                    warnings.append(msg)
            else:
                results["warnings"] = [msg]
            logging.error("%s (%s)", msg, results.get("unique_name", ""))
    except (TypeError, ValueError):
        pass
    return results


class _OptimizerDiverged(Exception):
    """Raised internally to stop an optimization whose energy has blown up."""


def _make_optimizer(name, atoms, trajectory=None, maxstep=None):
    """Instantiate an ASE optimizer by short name, capping the step if given.

    Supported: ``bfgs`` (default), ``lbfgs``, ``fire``. ``maxstep`` caps the
    per-step displacement (Å), which is the single most effective guard against
    the BFGS energy blow-ups seen with the mace-polar model.
    """
    name = (name or "bfgs").lower()
    kwargs = {}
    if maxstep is not None:
        kwargs["maxstep"] = maxstep
    if name == "bfgs":
        return BFGS(atoms, trajectory=trajectory, **kwargs)
    if name in ("lbfgs", "l-bfgs"):
        return LBFGS(atoms, trajectory=trajectory, **kwargs)
    if name == "fire":
        try:
            return FIRE(atoms, trajectory=trajectory, **kwargs)
        except TypeError:
            # Older ASE FIRE spells the cap differently; fall back to default.
            return FIRE(atoms, trajectory=trajectory)
    raise ValueError(f"Unknown optimizer {name!r}. Use 'bfgs', 'lbfgs', or 'fire'.")


def _run_staged_optimization(
    atoms,
    fmax,
    max_steps,
    trajectory,
    optimizer="bfgs",
    maxstep=None,
    recover=False,
    recover_optimizers=("lbfgs", "fire"),
    max_recovery_attempts=2,
):
    """Optimize ``atoms``, with an energy-divergence guard and staged restarts.

    Runs ``optimizer`` first. A per-step observer aborts the run the moment the
    energy becomes non-finite or exceeds a per-atom magnitude bound (default
    ``IQC_MAX_ENERGY_PER_ATOM_EV`` × natoms), so a diverging trajectory no
    longer grinds to ``max_steps`` and emits a 1e56 eV energy. If ``recover`` is
    set and the run does not converge to a finite geometry, it restarts from the
    current positions with the next optimizer in ``recover_optimizers`` (up to
    ``max_recovery_attempts`` extra tries).

    Returns ``(converged, total_steps, optimizer_used, recovery_used, attempts)``.
    """
    n_atoms = max(len(atoms), 1)
    energy_bound = float(os.environ.get("IQC_MAX_ENERGY_PER_ATOM_EV", 1e4)) * n_atoms

    def _attempt(opt_name):
        dyn = _make_optimizer(opt_name, atoms, trajectory=trajectory, maxstep=maxstep)

        def _guard():
            try:
                energy = atoms.get_potential_energy()
            except Exception:
                return
            if not np.isfinite(energy) or abs(energy) > energy_bound:
                raise _OptimizerDiverged(
                    f"energy {energy:.3e} eV diverged (|E| > {energy_bound:.3e})"
                )

        dyn.attach(_guard, interval=1)
        try:
            converged = bool(dyn.run(fmax=fmax, steps=max_steps))
        except _OptimizerDiverged as exc:
            logging.warning("Optimizer '%s' diverged and was aborted: %s", opt_name, exc)
            converged = False
        return converged, dyn.get_number_of_steps()

    sequence = [optimizer] + (list(recover_optimizers or ()) if recover else [])
    total_steps = 0
    attempts = 0
    used = optimizer
    converged = False
    for i, opt_name in enumerate(sequence):
        if i > 0:
            if attempts >= max_recovery_attempts:
                break
            attempts += 1
            logging.info(
                "Optimization recovery attempt %d: restarting with '%s'",
                attempts,
                opt_name,
            )
        converged, steps = _attempt(opt_name)
        total_steps += steps
        used = opt_name
        try:
            energy = atoms.get_potential_energy()
            finite = bool(np.isfinite(energy)) and abs(energy) <= energy_bound
        except Exception:
            finite = False
        if (converged and finite) or not recover:
            break
    return converged, total_steps, used, attempts > 0, attempts


def _imaginary_mode_vectors(frequencies, vib_modes, nrot, max_vib_imag):
    """Return the (N,3) displacement vectors for imaginary vibrational modes.

    ``frequencies`` is the complex frequency array from ASE; ``vib_modes`` are
    the corresponding modes (shape ``(3N, N, 3)``). The first ``3 + nrot``
    entries are translations/rotations and are skipped, mirroring the imaginary
    count elsewhere in this module.
    """
    vectors = []
    if vib_modes is None or len(vib_modes) == 0:
        return vectors
    body_freqs = frequencies[3 + nrot:]
    body_modes = vib_modes[3 + nrot:]
    for freq, mode in zip(body_freqs, body_modes):
        if abs(getattr(freq, "imag", 0.0)) > max_vib_imag:
            vectors.append(np.asarray(mode, dtype=float))
    return vectors


def _recover_imaginary(
    atoms, results, mode_vectors, recompute_fn, displacement=0.3,
    max_attempts=1, unique_name="",
):
    """Displace along imaginary modes, recompute, and keep the best geometry.

    ``recompute_fn(new_atoms) -> (new_atoms, new_results)`` re-optimizes and
    recomputes the Hessian (via run_vibrations or run_ir). A trial is accepted
    only if it errors out less and has strictly fewer imaginary modes; on
    acceptance ``atoms`` is moved to the improved geometry and the new result
    (tagged ``imag_recovery_used``/``imag_recovery_before``/``after``) is
    returned. Never raises — on any failure the original ``results`` is kept.
    """
    n_before = int(results.get("number_of_imaginary", 0) or 0)
    if n_before <= 0 or not mode_vectors:
        return results
    best = results
    for attempt in range(1, max_attempts + 1):
        disp = np.zeros((len(atoms), 3))
        for vec in mode_vectors:
            norm = np.linalg.norm(vec)
            if norm > 0:
                disp += vec / norm
        if not np.any(disp):
            break
        trial = atoms.copy()
        trial.set_positions(trial.get_positions() + displacement * disp)
        logging.info(
            "Imaginary-mode recovery attempt %d/%d for %s (currently %d imaginary)",
            attempt, max_attempts, unique_name,
            int(best.get("number_of_imaginary", 0) or 0),
        )
        try:
            trial, rec = recompute_fn(trial)
        except Exception as exc:  # never let recovery crash the parent task
            logging.warning("Imaginary-mode recovery attempt failed: %s", exc)
            break
        if not rec or rec.get("error"):
            break
        n_new = int(rec.get("number_of_imaginary", n_before) or 0)
        if n_new < int(best.get("number_of_imaginary", 0) or 0):
            rec["imag_recovery_used"] = True
            rec["imag_recovery_before"] = n_before
            rec["imag_recovery_after"] = n_new
            best = rec
            atoms.set_positions(trial.get_positions())
            if n_new == 0:
                break
        else:
            break
    return best


def _prepare_calculation(
    atoms,
    calculator=None,
    unique_name="",
    multiplicity=None,
    charge=0,
    apply_numerical_forces=True,
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

    _validate_geometry(atoms, unique_name=unique_name)

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

    # Energy-only calculators (ExaChem, PySCF CCSD(T)) get finite-difference
    # forces so the force-driven optimizer / Hessian can use them (no-op for
    # calculators that already implement forces). Gated by
    # apply_numerical_forces: force-needing tasks (opt/vib/ir/thermo) leave it
    # True (the default); `--task single` energy calculations pass False so
    # they do NOT pay for a full FD stencil (1 + 6*N_atoms extra single-point
    # invocations per molecule) they never use. This gating restores the
    # pre-2026-07-08 single-point behavior; without it, every ExaChem/PySCF
    # `--task single` silently ran a finite-difference forces stencil, making
    # sweep jobs ~10-100x slower and timing out (h=10 CCSD(T)/aug-cc-pVTZ:
    # ~30x exachem_run dirs per molecule, batch never finished).
    if apply_numerical_forces:
        calc = _wrap_with_numerical_forces(calc)

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
    _store_calculator_observables(results, calc, prefix="initial_")

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

    # Energy-only task: do NOT wrap energy-only calculators in finite-difference
    # forces. Single-point results discard forces (results["forces"] = [] below),
    # so an FD stencil would be pure waste (~6*N_atoms extra single points).
    calc, results = _prepare_calculation(
        atoms,
        calculator,
        unique_name,
        multiplicity=multiplicity,
        charge=charge,
        apply_numerical_forces=False,
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
        elif _is_orca_calculator(calc) and not _orca_requests_engrad(calc):
            # ORCA advertises "forces" in implemented_properties even when the
            # input has no ENGRAD, so atoms.get_forces() would re-run the
            # entire SCF a second time only to raise PropertyNotImplementedError.
            warning = (
                "Forces were not requested from ORCA (no ENGRAD in "
                "orcasimpleinput); single-point energy was saved."
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

    _store_calculator_observables(results, calc)

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
    optimizer="bfgs",
    maxstep=None,
    recover=False,
    recover_optimizers=("lbfgs", "fire"),
    max_recovery_attempts=2,
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
    # Coerce numeric params that may arrive as strings from YAML. YAML 1.1
    # parses unquoted scientific notation like "1e-3" as a *string*, not a
    # float; passing that straight to the optimizer otherwise surfaces as a
    # cryptic "ufunc 'less' ... Float64/StrDType" error at step 0.
    fmax = float(fmax)
    max_steps = int(max_steps)
    if maxstep is not None:
        maxstep = float(maxstep)

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
            "opt_optimizer": optimizer,
            "opt_recovery_used": False,
            "opt_recovery_attempts": 0,
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
        (
            converged,
            n_steps,
            optimizer_used,
            recovery_used,
            recovery_attempts,
        ) = _run_staged_optimization(
            atoms,
            fmax=fmax,
            max_steps=max_steps,
            trajectory=trajectory,
            optimizer=optimizer,
            maxstep=maxstep,
            recover=recover,
            recover_optimizers=recover_optimizers,
            max_recovery_attempts=max_recovery_attempts,
        )
        results["opt_time"] = time.time() - start_time
        results["opt_steps"] = n_steps
        results["opt_converged"] = converged
        results["opt_optimizer"] = optimizer_used
        results["opt_recovery_used"] = recovery_used
        results["opt_recovery_attempts"] = recovery_attempts
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
        _store_calculator_observables(results, calc, prefix="opt_")

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

    validate_physical_results(results, atoms=atoms)
    logging.info(f"Geometry optimization for {unique_name} completed")
    return atoms, results


def _optimization_extra_params(params):
    """Return only params accepted by run_optimization beyond common arguments.

    Includes the optimizer/recovery knobs so they propagate from
    ``optimization_params`` in the config through the vibration/IR/thermo paths
    (which forward ``**params``) down to ``run_optimization``.
    """

    return {
        key: params[key]
        for key in (
            "max_steps",
            "output_dir",
            "optimizer",
            "maxstep",
            "recover",
            "recover_optimizers",
            "max_recovery_attempts",
        )
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
    imag_recovery=False,
    imag_displacement=0.3,
    max_imag_attempts=1,
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

    # Energy-only calculators (ExaChem, PySCF CCSD(T)) get finite-difference
    # forces so the Hessian (and any non-optimizing path) can use them.
    calc = _wrap_with_numerical_forces(calc)

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

    imag_vectors = []
    try:
        start_time = time.time()
        vib_name = f"tmp_vib_{unique_name}"
        if vib_dir:
            os.makedirs(vib_dir, exist_ok=True)
            vib_name = os.path.join(vib_dir, vib_name)
        # Attach the resolved calculator: ASE's Vibrations captures atoms.calc,
        # which with optimize=False is otherwise None (crash) or a stale
        # calculator from an earlier run (silently wrong Hessian). run_ir does
        # the same before building its Infrared object.
        atoms.calc = calc
        vib = Vibrations(atoms, name=vib_name, indices=indices, delta=delta)
        # Drop any leftover displacement cache: ASE reuses same-named cache
        # files, so an interrupted earlier run would contribute forces from a
        # different geometry or calculator (run_ir already cleans first).
        vib.clean()
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
        imag_vectors = _imaginary_mode_vectors(
            frequencies, vib_modes, nrot, max_vib_imag
        )

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

    if (
        imag_recovery
        and not results.get("error")
        and int(results.get("number_of_imaginary", 0) or 0) > 0
        and imag_vectors
    ):
        def _recompute(new_atoms):
            return run_vibrations(
                new_atoms,
                calculator=calc,
                optimize=True,
                unique_name=f"{unique_name}_imagrec",
                vib_dir=vib_dir,
                indices=indices,
                fmax=fmax,
                delta=delta,
                max_trans_rot=max_trans_rot,
                max_vib_imag=max_vib_imag,
                multiplicity=multiplicity,
                charge=charge,
                imag_recovery=False,
                **params,
            )

        results = _recover_imaginary(
            atoms,
            results,
            imag_vectors,
            _recompute,
            displacement=imag_displacement,
            max_attempts=max_imag_attempts,
            unique_name=unique_name,
        )

    logging.info(f"Vibrational analysis for {unique_name} completed")
    return atoms, results


def _add_thermo_results_from_vibrations(
    atoms,
    results,
    ignore_imag_modes=True,
    multiplicity=None,
    potentialenergy=None,
):
    """Append IdealGasThermo properties using vibrational data in results.

    ``potentialenergy`` (eV) overrides the electronic energy that anchors the
    thermochemistry. This enables composite schemes where the geometry and
    Hessian come from a cheap force-capable calculator (e.g. MACE) while the
    electronic energy comes from a higher level of theory (e.g. CCSD(T)); if
    None, the energy of the calculator currently attached to ``atoms`` is used.
    """

    thermo = None
    try:
        vib_energies = results.get("vib_energies", None)
        if vib_energies is None:
            logging.error(
                "Vibrational energies not found in vibration results. Cannot calculate thermo properties."
            )
            results["error"] += "Missing vibrational energies for thermochemistry.\n"
            return None, results

        if not ignore_imag_modes:
            n_imag = results.get("number_of_imaginary", 0)
            if n_imag > 0:
                error = (
                    "Imaginary vibrational energies are present: "
                    f"({n_imag} imaginary modes).\n"
                )
                results["error"] += error
                logging.error(error)
                return None, results

        if potentialenergy is None:
            potentialenergy = atoms.get_potential_energy()
        # ASE's IdealGasThermo rejects periodic atoms (it is a gas-phase model).
        # Plane-wave backends (VASP) need a periodic cell, so the molecule was
        # wrapped in a box. Use a non-periodic copy for the thermo — the
        # positions/masses that set the moments of inertia are unchanged.
        thermo_atoms = atoms
        if bool(getattr(atoms, "pbc", None) is not None and atoms.pbc.any()):
            thermo_atoms = atoms.copy()
            thermo_atoms.pbc = False
            thermo_atoms.calc = None
        start_time = time.time()
        thermo = IdealGasThermo(
            vib_energies=vib_energies,
            geometry=get_geometry_type(thermo_atoms),
            atoms=thermo_atoms,
            potentialenergy=potentialenergy,
            spin=get_spin(thermo_atoms, results.get("multiplicity", multiplicity)),
            symmetrynumber=results.get("opt_sym_number", 1),
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

    validate_physical_results(results, atoms=atoms)
    return thermo, results


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
    max_trans_rot=100,
    max_vib_imag=50,
    multiplicity=None,
    charge=0,
    imag_recovery=False,
    imag_displacement=0.3,
    max_imag_attempts=1,
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
                "computed. Regular MACE and EMT do not support dipoles. Use "
                "a calculator that does (e.g. xtb, orca, or mace-polar), or "
                "pass an instance via the Python API.\n"
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

    imag_vectors = []
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

        vib_data = ir.get_vibrations()
        mode_frequencies = vib_data.get_frequencies()
        try:
            vib_energies, vib_modes = vib_data.get_energies_and_modes()
        except (AttributeError, TypeError, ValueError):
            vib_energies = (
                vib_data.get_energies() if hasattr(vib_data, "get_energies") else []
            )
            vib_modes = []
        results["frequencies_cm^-1"] = (
            mode_frequencies.tolist()
            if hasattr(mode_frequencies, "tolist")
            else mode_frequencies
        )
        results["vib_energies"] = (
            vib_energies.tolist() if hasattr(vib_energies, "tolist") else vib_energies
        )
        results["vib_modes"] = (
            vib_modes.tolist() if hasattr(vib_modes, "tolist") else vib_modes
        )
        nrot = 2 if is_linear_by_inertia(atoms) else 3
        if np.any(np.abs(mode_frequencies[: 3 + nrot]) > max_trans_rot):
            logging.warning(
                "Translational or rotational modes are too high: "
                f"{mode_frequencies[:3+nrot]}"
            )
            results["warnings"].append("Translational or rotational modes are too high")
        img_freqs = [
            f for f in mode_frequencies[3 + nrot :] if abs(f.imag) > max_vib_imag
        ]
        results["number_of_imaginary"] = len(img_freqs)
        results["vibrational_frequencies_cm^-1"] = [
            f.real for f in mode_frequencies[3 + nrot :]
        ]
        imag_vectors = _imaginary_mode_vectors(
            mode_frequencies, vib_modes, nrot, max_vib_imag
        )

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

    if (
        imag_recovery
        and not results.get("error")
        and int(results.get("number_of_imaginary", 0) or 0) > 0
        and imag_vectors
    ):
        def _recompute(new_atoms):
            return run_ir(
                new_atoms,
                calculator=calculator,
                optimization_calculator=optimization_calculator,
                vibration_calculator=vibration_calculator,
                dipole_calculator=dipole_calculator,
                optimize=True,
                unique_name=f"{unique_name}_imagrec",
                vib_dir=vib_dir,
                indices=indices,
                fmax=fmax,
                delta=delta,
                ir_spectrum_start=ir_spectrum_start,
                ir_spectrum_end=ir_spectrum_end,
                sparse_spectrum=sparse_spectrum,
                intensity_threshold=intensity_threshold,
                max_trans_rot=max_trans_rot,
                max_vib_imag=max_vib_imag,
                multiplicity=multiplicity,
                charge=charge,
                imag_recovery=False,
                **params,
            )

        results = _recover_imaginary(
            atoms,
            results,
            imag_vectors,
            _recompute,
            displacement=imag_displacement,
            max_attempts=max_imag_attempts,
            unique_name=unique_name,
        )

    logging.info(f"IR analysis for {unique_name} completed")
    return atoms, results


def run_ir_thermo(
    atoms,
    calculator=None,
    optimization_calculator=None,
    vibration_calculator=None,
    dipole_calculator=None,
    optimize=True,
    ignore_imag_modes=True,
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
    max_trans_rot=100,
    max_vib_imag=50,
    multiplicity=None,
    charge=0,
    **params,
):
    """Run IR and thermochemistry from one optimized Hessian calculation."""

    atoms, results = run_ir(
        atoms=atoms,
        calculator=calculator,
        optimization_calculator=optimization_calculator,
        vibration_calculator=vibration_calculator,
        dipole_calculator=dipole_calculator,
        optimize=optimize,
        unique_name=unique_name,
        vib_dir=vib_dir,
        indices=indices,
        fmax=fmax,
        delta=delta,
        trajectory=trajectory,
        save_geometry=save_geometry,
        ir_spectrum_start=ir_spectrum_start,
        ir_spectrum_end=ir_spectrum_end,
        sparse_spectrum=sparse_spectrum,
        intensity_threshold=intensity_threshold,
        max_trans_rot=max_trans_rot,
        max_vib_imag=max_vib_imag,
        multiplicity=multiplicity,
        charge=charge,
        **params,
    )
    if results["error"]:
        logging.error("IR analysis failed, cannot proceed with thermochemistry.")
        return atoms, results

    _, results = _add_thermo_results_from_vibrations(
        atoms,
        results,
        ignore_imag_modes=ignore_imag_modes,
        multiplicity=multiplicity,
    )
    if results["error"]:
        return atoms, results

    logging.info(f"IR thermochemistry calculation for {unique_name} completed")
    return atoms, results


# Energy-component keys copied from a composite thermo energy calculator
# (ExaChem / PySCF) into the top-level results so they are persisted directly.
_ENERGY_COMPONENT_FIELDS = (
    "scf_energy_eV",
    "mp2_correlation_eV",
    "ccsd_correlation_eV",
    "t_correction_eV",
    "total_energy_eV",
    "scf_time_s",
    "mp2_time_s",
    "ccsd_time_s",
    "t_time_s",
)


def run_thermo(
    atoms,
    calculator=None,
    ignore_imag_modes=True,
    unique_name="",
    trajectory=None,
    save_geometry=False,
    multiplicity=None,
    charge=0,
    optimization_calculator=None,
    vibration_calculator=None,
    energy_calculator=None,
    **params,
):
    """
    Run thermochemistry calculations for an ASE Atoms object.

    Supports two modes:

    * **Single-level** (default): one ``calculator`` does geometry
      optimization, the Hessian, and the electronic energy. Correct for
      force-capable methods (MACE, MACE-Polar, PySCF DFT/HF).

    * **Composite**: a force-capable ``vibration_calculator`` (falling back to
      ``optimization_calculator`` / ``calculator``) provides the optimized
      geometry and harmonic frequencies, while a separate ``energy_calculator``
      provides the electronic energy via a single point at the optimized
      geometry. This is how energy-only correlated methods (PySCF CCSD(T),
      ExaChem CCSD(T)) are turned into thermochemistry — e.g. CCSD(T) energy on
      a MACE geometry with MACE harmonic ZPE/thermal corrections.

    Args:
        atoms (ase.Atoms): ASE Atoms object
        calculator: Fallback calculator used for any role not set explicitly.
        ignore_imag_modes (bool): Whether to ignore imaginary vibrational modes
        unique_name (str): Unique name for the molecule
        trajectory (str): Path to save trajectory file during optimization
        save_geometry (bool): Whether to save the final optimized geometry
        optimization_calculator: Force-capable calc for the geometry step.
        vibration_calculator: Force-capable calc for the Hessian step.
        energy_calculator: Calc providing the final electronic energy.
        **opt_params: Additional keyword arguments passed to run_optimization.

    Returns:
        tuple: (thermo, results)
    """

    # Geometry + Hessian require forces; pick the first force-capable role.
    geom_calc = vibration_calculator or optimization_calculator or calculator

    atoms, results = run_vibrations(
        atoms,
        calculator=geom_calc,
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

    # Composite scheme: substitute the electronic energy from a higher level.
    potentialenergy = None
    if energy_calculator is not None and energy_calculator is not geom_calc:
        try:
            apply_spin_charge(
                atoms, energy_calculator, multiplicity=multiplicity, charge=charge
            )
            _assign_orca_work_directory(
                energy_calculator, unique_name, purpose="thermo_energy"
            )
            atoms.calc = energy_calculator
            potentialenergy = atoms.get_potential_energy()
            results["energy_calculator"] = str(energy_calculator)
            results["electronic_energy_eV"] = potentialenergy
            energy_results = getattr(energy_calculator, "results", {}) or {}
            for key in _ENERGY_COMPONENT_FIELDS:
                if key in energy_results:
                    results[key] = energy_results[key]
            _store_calculator_observables(results, energy_calculator, prefix="")
        except Exception as e:
            error = f"Energy calculator single-point failed: {e}\n"
            results["error"] += error
            logging.error(error)
            return None, results

    thermo, results = _add_thermo_results_from_vibrations(
        atoms,
        results,
        ignore_imag_modes=ignore_imag_modes,
        multiplicity=multiplicity,
        potentialenergy=potentialenergy,
    )
    if results["error"]:
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
    elif is_linear_by_inertia(atoms):
        return "linear"
    else:
        return "nonlinear"
