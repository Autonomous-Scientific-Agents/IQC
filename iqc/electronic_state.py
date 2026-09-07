"""Validate molecular charge and spin without importing calculator backends."""

import math
from collections.abc import Mapping


def integer_state(value, name):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer, got {value!r}") from None
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return int(number)


def validate_electronic_state(atoms, charge=0, multiplicity=None):
    charge = integer_state(charge, "charge")
    electrons = int(sum(atoms.get_atomic_numbers())) - charge
    if electrons < 0:
        raise ValueError(f"charge={charge} leaves a negative electron count")
    if multiplicity is None:
        multiplicity = 1 + electrons % 2
    multiplicity = integer_state(multiplicity, "multiplicity")
    unpaired = multiplicity - 1
    if unpaired < 0 or unpaired > electrons or (electrons - unpaired) % 2:
        raise ValueError(
            f"multiplicity={multiplicity} is incompatible with {electrons} "
            f"electrons at charge={charge}"
        )
    return charge, multiplicity


def set_electronic_state(atoms, multiplicity=None, charge=None, parameters=None):
    """Resolve overrides, input metadata, calculator defaults, then parity."""
    parameters = parameters if isinstance(parameters, Mapping) else {}
    if charge is None and "charge" not in atoms.info:
        charge = parameters.get("charge", 0)
    if multiplicity is None and "multiplicity" not in atoms.info:
        multiplicity = parameters.get("multiplicity", parameters.get("mult"))
        if multiplicity is None and parameters.get("spin") is not None:
            multiplicity = abs(integer_state(parameters["spin"], "spin")) + 1
    inferred = multiplicity is None and (
        "multiplicity" not in atoms.info
        or atoms.info.get("_iqc_multiplicity_inferred", False)
    )
    if charge is None:
        charge = atoms.info.get("charge", 0)
    if multiplicity is None and not inferred:
        multiplicity = atoms.info.get("multiplicity")
    charge, multiplicity = validate_electronic_state(atoms, charge, multiplicity)
    atoms.info.update(charge=charge, multiplicity=multiplicity)
    atoms.info["_iqc_multiplicity_inferred"] = inferred
    return charge, multiplicity
