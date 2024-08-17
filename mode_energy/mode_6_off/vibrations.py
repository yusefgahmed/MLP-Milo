#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sample vibrational velocity for molecule groups in program state."""

import numpy as np

from milo import containers
from milo import energy_calculator
from milo import enumerations as en
from milo import scientific_constants as sc
from milo.exceptions import InputError


def sample(ps):
    """Calculate vibrational velocities."""

    velocities = containers.Velocities()
    for _ in range(ps.number_atoms):
        velocities.append(0, 0, 0, en.VelocityUnit.METER_PER_SEC)

    print("### Vibrational Velocity Sampling --------------------------------")

    if not ps.vibrations:
        print("  No vibrational velocity added.")
        print()
        return velocities

    for group_index in ps.vibrations:
        molecule_group = ps.molecule_groups[group_index]
        if not molecule_group:
            continue

        # Pull correct frequency data into numpy arrays
        if group_index == 0 and ps.frequencies:
            original_frequencies = ps.frequencies.as_recip_cm()
            frequencies = np.array(original_frequencies)
            reduced_masses = np.array(ps.reduced_masses.as_kilogram())
            mode_displacements = np.array(ps.mode_displacements)
        elif (group_index in ps.molecule_group_names
                and ps.molecule_group_names[group_index]
                in ps.solvent_frequencies):
            name = ps.molecule_group_names[group_index]
            original_frequencies = ps.solvent_frequencies[name].as_recip_cm()
            frequencies = np.array(original_frequencies)
            reduced_masses = \
                np.array(ps.solvent_reduced_masses[name].as_kilogram())
            mode_displacements = _get_aligned_displacements(ps, group_index)
        else:
            raise InputError(f"Couldn't find frequency data needed for "
                             f"vibrational sampling of molecule group "
                             f"{group_index + 1}.")

        # Set imaginary frequencies to 2 cm^-1 (treat as translation)
        has_imaginary_mode = frequencies[0] < 0
        frequencies[frequencies < 0] = 2

        # If using classical sampling, set all frequencies to 2 cm^-1
        if ps.vibrational_sampling is en.VibrationalSampling.CLASSICAL:
            frequencies.fill(sc.CLASSICAL_SPACING)

        # Sample vibrational quantum numbers and directions
        quantum_numbers = np.array(
            [ps.random.vibrational_quantum_number(f, ps.temperature)
             if ps.temperature > 0 else 0 for f in frequencies])
        directions = np.array(
            [ps.random.one_or_neg_one() for _ in frequencies])

        # Take care of phase and fixed quanta/direction (only solute)
        if group_index == 0:
            # Phase
            if (has_imaginary_mode and ps.phase_direction
                    is not en.PhaseDirection.RANDOM):
                # Check if mode pushes apart
                atom0 = np.array(ps.structures[-1].as_angstrom(ps.phase[0]))
                atom1 = np.array(ps.structures[-1].as_angstrom(ps.phase[1]))
                before_distance = np.linalg.norm(atom0 - atom1)
                atom0 += mode_displacements[0][ps.phase[0]]
                atom1 += mode_displacements[0][ps.phase[1]]
                after_distance = np.linalg.norm(atom0 - atom1)
                pushes_apart = after_distance > before_distance
                directions[0] = 1 if pushes_apart else -1
            if ps.phase_direction is en.PhaseDirection.BRING_TOGETHER:
                directions *= -1

            # These needs to rewrite the random calls from above so that the
            # rng will produce the same results for everything else.
            for mode in ps.fixed_vibrational_quanta:
                quantum_numbers[mode] = ps.fixed_vibrational_quanta[mode]
            for mode in ps.fixed_mode_directions:
                directions[mode] = ps.fixed_mode_directions[mode]

        # Calculate mode energies
        zero_point_energies = 0.5 * frequencies * sc.RECIP_CM_TO_JOULE
        zero_point_energies[frequencies <= 10] = 0
        if ps.vibrational_sampling is en.VibrationalSampling.CLASSICAL:
            zero_point_energies.fill(0)
        mode_energies = zero_point_energies + \
            (quantum_numbers * frequencies * sc.RECIP_CM_TO_JOULE)

        # Set the energy of the Xth vibrational mode to 0
        mode_index = 5  # 0-based index for the 6th mode
        mode_energies[mode_index] = 0

        # Calculate mode velocities
        mode_velocities = directions * np.sqrt(2 * mode_energies
                                               / reduced_masses)

        # Set the velocity of the Xth vibrational mode to 0
        mode_velocities[mode_index] = 0

        # Calculate atomic velocities
        atomic_velocities = np.sum(
            mode_displacements * mode_velocities[:, None, None], axis=0)

        # Save in container
        for i, (x, y, z) in zip(molecule_group, atomic_velocities):
            velocities.alter(i, x, y, z, en.VelocityUnit.METER_PER_SEC)

        # Print summary from sampling
        print(f"  Molecule Group {group_index + 1}:")
        print("    Mode  Wavenumber  Wavenumber Used  Quantum No.  Direction")
        print("    ---------------------------------------------------------")
        for i, (orig_freq, freq, quantum_n, direction) in enumerate(zip(
                original_frequencies, frequencies, quantum_numbers,
                directions), start=1):
            print(f"    {i:>4}  {orig_freq:10.3f}  {freq:15.3f}  "
                  f"{quantum_n:>11}  {direction:>9}")
        print("    Mode    Energy (kcal/mol)   Velocity (m/s)   Includes ZPE")
        print("    ---------------------------------------------------------")
        for i, (energy, velocity, has_ZPE) in enumerate(zip(
                mode_energies, mode_velocities, zero_point_energies != 0),
                start=1):
            print_ZPE = "Yes" if has_ZPE else "No"
            if i == 6:
                print(f"    {i:>4}    {energy * sc.JOULE_TO_KCAL_PER_MOLE:17.6f}"
                      f"   {velocity:14.6e}   {print_ZPE:>12}"
                      f"   (Energy and velocity set to 0)")
            else:
                print(f"    {i:>4}    {energy * sc.JOULE_TO_KCAL_PER_MOLE:17.6f}"
                      f"   {velocity:14.6e}   {print_ZPE:>12}")

    # Scale vibrational velocities, if requested
    if ps.vibrational_energy_scalar != 1:
        scalar = np.sqrt(ps.vibrational_energy_scalar)
        print(f"  Scaling vibrational velocities by {scalar:.6f}, so the")
        print(f"  vibrational energy will be scaled by "
              f"{ps.vibrational_energy_scalar}.")
        velocities *= scalar

    print("  Vibrational Velocities (meter/second):")
    for atom, (x, y, z) in zip(ps.atoms, velocities.as_meter_per_sec()):
        print(f"    {atom.symbol:2} {x:15.6e} {y:15.6e} {z:15.6e}")

    print("  Vibrational Kinetic Energy (kcal/mol):")
    energy = energy_calculator.kinetic_energy(velocities, ps.atoms)
    print(f"    {energy.as_kcal_per_mole():.6f}")

    print()

    return velocities


def _get_aligned_displacements(ps, group_index):
    """
    Aligns displacement modes to new positions using Kabsch algorithm.

    Reference: https://en.wikipedia.org/wiki/Kabsch_algorithm
    """
    name = ps.molecule_group_names[group_index]
    if name not in ps.solvent_structure_inputs:
        raise InputError(f"Couldn't find original structure needed for "
                         f"vibrational sampling of molecule group "
                         f"{group_index + 1}. Put original structure from "
                         f" frequency calculation in $solvent_molecule.")

    old_coords = ps.solvent_structure_inputs[name].structure.as_angstrom()
    new_coords = np.array(
        [ps.structures[-1].as_angstrom(i) for i in
         ps.molecule_groups[group_index]])

    # Translation (center coordinates on centroid)
    old_centered = old_coords - np.average(old_coords, axis=0)
    new_centered = new_coords - np.average(new_coords, axis=0)

    # Computation of the covariance matrix
    H = old_centered.T @ new_centered

    # Computation of the optimal rotation matrix
    U, S, Vh = np.linalg.svd(H)
    V = Vh.T
    d = np.sign(np.linalg.det(V @ U.T))  # Make sure it's right-handed
    R = V @ np.diag([1, 1, d]) @ U.T

    # Use rotation matrix to align mode displacements
    old_displacements = \
        np.array(ps.solvent_mode_displacements[name])
    return (R @ old_displacements.transpose((0, 2, 1))).transpose((0, 2, 1))


def _check_if_mode_pushes_apart(ps):
    """Return if first mode increases distance between the atoms."""