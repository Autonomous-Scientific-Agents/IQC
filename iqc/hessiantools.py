"""Rigid-body (translation/rotation) projection of molecular Hessians.

ASE's ``Vibrations``/``Infrared`` diagonalise the full 3N x 3N mass-weighted
Hessian, so the six (five for linear molecules) rigid-body modes stay in the
frequency list, contaminated by whatever residual gradient, grid noise, or
finite-difference error the calculator produced. ASE's thermochemistry then
*guesses* which modes to drop by magnitude ordering. That guess fails exactly
where thermochemistry is most sensitive: a genuine soft torsion can be smaller
than a contaminated rotation, and any imaginary mode sorts below the
rigid-body block and is silently discarded.

This module implements the Eckart/Sayvetz projection instead: build an
orthonormal basis of the rigid-body motions in mass-weighted coordinates,
project it out of the Hessian, and diagonalise the remainder. The rigid-body
eigenvalues then vanish to machine precision and the 3N-6 (3N-5) vibrational
modes are unambiguous and decoupled.

The projection acts on the Hessian only, so it is independent of the
calculator that produced it: finite-difference Hessians from ASE (any
force-capable calculator, or energy-only calculators wrapped with numerical
forces), analytic Hessians read from an electronic-structure code, or a
``VibrationsData`` object restored from disk are all accepted.

Two caveats govern its use:

* The projection is exact only at a stationary point (forces ~ 0). At a
  non-stationary geometry the rotational vectors are not null vectors of the
  invariant Hessian and a gradient correction would be required.
* Projection removes contamination from the frequency list; it does not fix
  the Hessian. The report carries the rigid-body frequencies *before*
  projection so a loosely converged geometry or a poor finite-difference step
  is still visible (``max_trans_rot`` in the IQC pipeline still warns on
  them).

Typical use outside the IQC pipeline::

    from ase.vibrations import Vibrations
    from iqc.hessiantools import internal_mode_mask, project_vibrations_data

    vib = Vibrations(atoms); vib.run()
    projected, report = project_vibrations_data(vib.get_vibrations())
    energies = projected.get_energies()                       # 3N, ASE order
    internal = energies[internal_mode_mask(energies, report["n_rigid"])]
    # ``internal`` has exactly 3N-6 (3N-5) entries and keeps genuine imaginary
    # modes; a plain ``energies[n_rigid:]`` slice would drop them, because ASE
    # sorts by eigenvalue and imaginary modes precede the zero rigid block.
"""

from __future__ import annotations

import logging

import numpy as np
from ase import units
from ase.vibrations import VibrationsData

__all__ = [
    "canonicalize_vibrations_data",
    "rigid_body_basis",
    "project_hessian",
    "project_vibrations_data",
    "internal_mode_mask",
    "eigenvalues_to_frequencies_cm",
]

# sqrt(eV / (A^2 amu)) -> eV, the conversion used by ase.vibrations.VibrationsData
_EIGENVALUE_TO_EV = units._hbar * units.m / np.sqrt(units._e * units._amu)


def eigenvalues_to_frequencies_cm(eigenvalues):
    """Convert mass-weighted Hessian eigenvalues (eV/A^2/amu) to signed cm^-1.

    Negative eigenvalues (imaginary modes) are returned as negative
    frequencies, matching the ``vibrational_frequencies_cm^-1`` convention used
    in IQC result records.
    """
    ev = np.asarray(eigenvalues, dtype=float)
    return np.sign(ev) * np.sqrt(np.abs(ev)) * _EIGENVALUE_TO_EV / units.invcm


def internal_mode_mask(values, n_rigid):
    """Boolean mask selecting the internal (vibrational) modes.

    ``values`` are the mode energies or frequencies of a *projected* Hessian
    (real, complex, or signed-negative for imaginary modes). The ``n_rigid``
    entries of smallest magnitude are the projected rigid-body modes (zero by
    construction) and are masked out; everything else - including genuine
    imaginary modes, which ASE orders *before* the near-zero rigid block -
    is kept. Do not slice ``[n_rigid:]`` instead: at a saddle point that
    discards the imaginary vibration and keeps a rigid mode.
    """
    values = np.asarray(values)
    n_rigid = int(n_rigid)
    if not 0 <= n_rigid <= values.size:
        raise ValueError(f"n_rigid={n_rigid} out of range for {values.size} modes")
    mask = np.ones(values.size, dtype=bool)
    mask[np.argsort(np.abs(values), kind="stable")[:n_rigid]] = False
    return mask


def rigid_body_basis(atoms, n_rot=None, rank_tol=1e-6):
    """Orthonormal basis of translations + rotations in mass-weighted Cartesians.

    Args:
        atoms: ``ase.Atoms`` supplying masses and positions.
        n_rot: Number of rotational degrees of freedom to remove (3 for a
            nonlinear molecule, 2 for a linear one, 0 for an atom). ``None``
            detects it from the numerical rank of the rotation vectors.
        rank_tol: Relative singular-value threshold for rank detection, and
            the floor below which a *requested* rotation is judged
            numerically absent (falls back to the detected rank).

    Returns:
        tuple: ``(D, n_rigid)`` with ``D`` of shape ``(3N, n_rigid)`` and
        orthonormal columns; ``n_rigid = 3 + n_rot``.
    """
    n_atoms = len(atoms)
    if n_atoms == 0:
        raise ValueError("Cannot build a rigid-body basis for zero atoms")
    sqrt_m = np.sqrt(np.asarray(atoms.get_masses(), dtype=float))
    if not np.all(np.isfinite(sqrt_m)) or np.any(sqrt_m <= 0):
        raise ValueError("Atomic masses must be finite and positive")
    x = atoms.get_positions() - atoms.get_center_of_mass()

    columns = []
    for k in range(3):  # translations
        v = np.zeros((n_atoms, 3))
        v[:, k] = sqrt_m
        columns.append(v.ravel())
    rotations = []
    for k in range(3):  # infinitesimal rotations about the centre of mass
        axis = np.zeros(3)
        axis[k] = 1.0
        v = np.cross(axis[None, :], x) * sqrt_m[:, None]
        rotations.append(v.ravel())

    T = np.array(columns).T
    R = np.array(rotations).T
    # Translations are exactly orthogonal to rotations about the COM, so
    # orthonormalise the rotational block on its own and rank-detect it.
    T /= np.linalg.norm(T, axis=0)
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    scale = s[0] if s.size and s[0] > 0 else 1.0
    detected = int(np.sum(s > rank_tol * scale)) if scale > 0 else 0
    if n_rot is None:
        n_rot = detected
    else:
        n_rot = int(n_rot)
        if not 0 <= n_rot <= 3:
            raise ValueError(f"n_rot must be in 0..3, got {n_rot}")
        if n_rot > detected:
            logging.warning(
                "Requested %d rotational modes but only %d are numerically "
                "present; using %d.",
                n_rot,
                detected,
                detected,
            )
            n_rot = detected
    D = np.hstack([T, U[:, :n_rot]])
    return D, 3 + n_rot


def project_hessian(hessian, atoms, n_rot=None, rank_tol=1e-6):
    """Project translations and rotations out of a Cartesian Hessian.

    Args:
        hessian: Cartesian Hessian in eV/A^2, shape ``(3N, 3N)`` or
            ``(N, 3, N, 3)``. Symmetrised before use.
        atoms: ``ase.Atoms`` with the geometry the Hessian was evaluated at.
        n_rot: See :func:`rigid_body_basis`.
        rank_tol: See :func:`rigid_body_basis`.

    Returns:
        tuple: ``(projected_hessian, report)``. The projected Hessian is
        returned in *Cartesian* form (same units and shape convention as the
        input, always 2D) so it can be fed straight back into
        ``ase.vibrations.VibrationsData.from_2d``. ``report`` is a dict with:

        * ``n_rigid`` - number of projected rigid-body modes (6/5/3),
        * ``geometry`` - ``"nonlinear"``, ``"linear"`` or ``"monatomic"``,
        * ``rigid_body_frequencies_cm`` - signed cm^-1 of the rigid-body block
          before projection (contamination diagnostic),
        * ``rigid_body_coupling_norm`` - Frobenius norm of the block coupling
          rigid-body and internal motions (vanishes at a stationary point of
          an invariant PES),
        * ``frequencies_unprojected_cm`` / ``frequencies_projected_cm`` -
          sorted signed cm^-1 of the full Hessian before and after projection.
    """
    n_atoms = len(atoms)
    H = np.asarray(hessian, dtype=float)
    if H.shape == (n_atoms, 3, n_atoms, 3):
        H = H.reshape(3 * n_atoms, 3 * n_atoms)
    if H.shape != (3 * n_atoms, 3 * n_atoms):
        raise ValueError(
            f"Hessian shape {H.shape} does not match {n_atoms} atoms; "
            "rigid-body projection needs the complete molecular Hessian"
        )
    if not np.all(np.isfinite(H)):
        raise ValueError("Hessian contains NaN/Inf")
    H = 0.5 * (H + H.T)

    inv_sqrt_m = 1.0 / np.sqrt(np.repeat(np.asarray(atoms.get_masses(), float), 3))
    H_mw = H * inv_sqrt_m[:, None] * inv_sqrt_m[None, :]

    D, n_rigid = rigid_body_basis(atoms, n_rot=n_rot, rank_tol=rank_tol)
    P = np.eye(3 * n_atoms) - D @ D.T
    H_proj = P @ H_mw @ P
    H_proj = 0.5 * (H_proj + H_proj.T)

    rigid_block = D.T @ H_mw @ D
    coupling = P @ H_mw @ D
    report = {
        "n_rigid": int(n_rigid),
        "geometry": {6: "nonlinear", 5: "linear", 3: "monatomic"}[n_rigid],
        "rigid_body_frequencies_cm": eigenvalues_to_frequencies_cm(
            np.linalg.eigvalsh(rigid_block)
        ),
        "rigid_body_coupling_norm": float(np.linalg.norm(coupling)),
        "frequencies_unprojected_cm": eigenvalues_to_frequencies_cm(
            np.linalg.eigvalsh(H_mw)
        ),
        "frequencies_projected_cm": eigenvalues_to_frequencies_cm(
            np.linalg.eigvalsh(H_proj)
        ),
    }
    sqrt_m = 1.0 / inv_sqrt_m
    H_cart = H_proj * sqrt_m[:, None] * sqrt_m[None, :]
    return H_cart, report


def project_vibrations_data(vib_data, n_rot=None, rank_tol=1e-6):
    """Return a rigid-body-projected copy of an ASE ``VibrationsData``.

    Accepts a ``VibrationsData`` or any object exposing ``get_vibrations()``
    (``ase.vibrations.Vibrations``, ``Infrared``, and their subclasses), so
    the result of any ASE finite-difference run can be projected regardless
    of calculator.

    The input is first passed through :func:`canonicalize_vibrations_data`,
    so an unsorted ``indices`` list covering all atoms is projected in atom
    order; the returned object is in canonical atom order (``indices=None``).
    Raises ``ValueError`` for partial Hessians (``indices`` restricting the
    displaced atoms): there is no complete rigid-body subspace to remove.

    Returns:
        tuple: ``(VibrationsData, report)`` - see :func:`project_hessian`.
    """
    if not isinstance(vib_data, VibrationsData) and hasattr(vib_data, "get_vibrations"):
        vib_data = vib_data.get_vibrations()
    if not isinstance(vib_data, VibrationsData):
        raise TypeError(
            "project_vibrations_data expects an ase.vibrations.VibrationsData "
            f"(or an object with get_vibrations()), got {type(vib_data).__name__}"
        )
    vib_data = canonicalize_vibrations_data(vib_data)
    atoms = vib_data.get_atoms()
    indices = vib_data.get_indices()
    n_atoms = len(atoms)
    if indices is not None and len(indices) != n_atoms:
        raise ValueError(
            "Rigid-body projection requires a complete Hessian; "
            f"only {len(indices)} of {n_atoms} atoms were displaced"
        )
    projected, report = project_hessian(
        vib_data.get_hessian_2d(), atoms, n_rot=n_rot, rank_tol=rank_tol
    )
    return VibrationsData.from_2d(atoms, projected), report


def canonicalize_vibrations_data(vib_data):
    """Return ``vib_data`` with its Hessian blocks in ascending atom order.

    ``ase.vibrations.Vibrations``/``Infrared`` fill the Hessian in the order
    of the ``indices`` they were given, whereas ``VibrationsData``
    mass-weights the Hessian and reports modes in ascending-index (mask)
    order. For an unsorted ``indices`` list that permutes atoms of unequal
    mass, ASE's own frequencies are therefore inconsistent (EMT water with
    ``indices=[2, 1, 0]``: 2670 vs 2406 cm^-1), and any consumer that lines
    the blocks up with ``atoms`` - such as the rigid-body projector - sees the
    wrong atoms. Sorting the blocks removes the ambiguity; partial Hessians
    are handled the same way.

    Returns the input object unchanged when ``indices`` is ``None`` or
    already sorted.
    """
    indices = vib_data.get_indices()
    if indices is None:
        return vib_data
    indices = np.asarray(indices, dtype=int)
    order = np.argsort(indices, kind="stable")
    if np.array_equal(order, np.arange(len(indices))):
        return vib_data
    # New block j describes atom sorted(indices)[j] = indices[order[j]].
    perm = (3 * order[:, None] + np.arange(3)[None, :]).ravel()
    hessian = vib_data.get_hessian_2d()[np.ix_(perm, perm)]
    return VibrationsData.from_2d(vib_data.get_atoms(), hessian, indices=indices[order])
