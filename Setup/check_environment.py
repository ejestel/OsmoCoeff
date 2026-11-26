#!/usr/bin/env python3
"""
Check that an OsmoCoeff environment is safe to simulate with.

    python Setup/check_environment.py                # login node / no GPU needed
    python Setup/check_environment.py --require-gpu  # on a GPU node

Exits with status 1 if any check fails.
"""
import argparse
import sys
from importlib.metadata import PackageNotFoundError, version

Q_O, Q_H = -0.834, 0.417      # TIP3P partial charges (e), Jorgensen et al. 1983
TOL = 1e-4

failures = []


def check(ok, label, detail=""):
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f"  ({detail})" if detail else ""))
    if not ok:
        failures.append(label)


def show_versions():
    for pkg in ("polymerist", "openff-toolkit", "openff-interchange", "openff-nagl",
                "openff-forcefields", "openmm", "numpy", "torch", "MDAnalysis"):
        try:
            print(f"       {pkg:20s} {version(pkg)}")
        except PackageNotFoundError:
            print(f"       {pkg:20s} not found in package metadata")


def check_water_molecule():
    """The molecule the dispatcher passes to charge_from_molecules."""
    from polymerist.mdtools.openfftools.solvation.solvents import water_TIP3P

    charges = water_TIP3P.partial_charges.m_as("elementary_charge")
    by_element = {(a.symbol, round(float(q), 4)) for a, q in zip(water_TIP3P.atoms, charges)}
    check(by_element == {("O", Q_O), ("H", Q_H)}, "polymerist water_TIP3P charges",
          str(sorted(by_element)))
    return water_TIP3P


def check_final_system(water):
    """Charges in the OpenMM System, i.e. what would actually be simulated."""
    import openmm
    from openff.toolkit import ForceField, Molecule, Topology
    from openff.units import unit
    from polymerist.mdtools.openfftools import GTR

    smiles = ["[Na+]", "[Cl-]", "[NH4+]", "[O-][N+](=O)[O-]"]
    ions = [Molecule.from_smiles(s) for s in smiles]
    n_water = 10
    top = Topology.from_molecules([water] * n_water + ions)
    top.box_vectors = [[4, 0, 0], [0, 4, 0], [0, 0, 4]] * unit.nanometer

    ff = ForceField("tip3p.offxml", "openff-2.3.0.offxml")
    system = ff.create_interchange(
        topology=top, toolkit_registry=GTR, charge_from_molecules=[water]
    ).to_openmm_system()

    nonbonded = next(f for f in system.getForces() if isinstance(f, openmm.NonbondedForce))
    q = [nonbonded.getParticleParameters(i)[0].value_in_unit(openmm.unit.elementary_charge)
         for i in range(nonbonded.getNumParticles())]

    waters = [q[3 * i:3 * i + 3] for i in range(n_water)]   # atom order O, H, H
    check(all(abs(w[0] - Q_O) < TOL and abs(w[1] - Q_H) < TOL and abs(w[2] - Q_H) < TOL
              for w in waters), "water charges in the OpenMM System", f"first water: {waters[0]}")
    check(all(abs(sum(w)) < TOL for w in waters), "every water is neutral")

    start = 3 * n_water
    for smi, mol in zip(smiles, ions):
        total = sum(q[start:start + mol.n_atoms])
        formal = mol.total_charge.m_as("elementary_charge")
        check(abs(total - formal) < TOL, f"net charge of {smi}", f"{total:+.4f}, expected {formal:+.0f}")
        start += mol.n_atoms
    check(abs(sum(q)) < TOL, "whole system is neutral", f"{sum(q):+.5f} e")


def check_gpu():
    """Actually create a CUDA context; listing the platform alone proves nothing."""
    import openmm

    system = openmm.System()
    system.addParticle(1.0)
    context = openmm.Context(system, openmm.VerletIntegrator(0.001),
                             openmm.Platform.getPlatformByName("CUDA"))
    context.setPositions([openmm.Vec3(0, 0, 0)])
    context.getState(getEnergy=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--require-gpu", action="store_true", help="fail if CUDA does not work")
    args = parser.parse_args()

    print(f"python: {sys.executable}")
    show_versions()

    try:
        check_final_system(check_water_molecule())
    except Exception as error:   # report any failure as a failed check, with its cause
        check(False, "build a test system", f"{type(error).__name__}: {error}")

    try:
        check_gpu()
        check(True, "OpenMM CUDA context")
    except Exception as error:
        if args.require_gpu:
            check(False, "OpenMM CUDA context", str(error))
        else:
            print(f"[SKIP] OpenMM CUDA context unavailable: {error}")

    print(f"\n{len(failures)} check(s) failed" if failures else "\nAll required checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
