#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Handle calling ESPs and parsing output."""

import os
import shutil
import struct
import subprocess
from sys import platform
from time import perf_counter
import logging



import numpy as np
import mlptrain as mlt
from ase.io import read, write
from ase.data import atomic_numbers as ase_atomic_numbers


from scipy.special import factorial2
from scipy.spatial import distance_matrix

from milo import containers
from milo import enumerations as en
from milo import exceptions
from milo.program_state import ProgramState
from milo.scientific_constants import (
    ELECTRON_VOLT_TO_JOULE,
    JOULE_TO_HARTREE,
    AMU_TO_KG,
    WAVENUMBER_TO_HARTREE,
    ELECTRON_VOLT_TO_HARTREE,
    ELECTRON_VOLT_ANGSTROM_TO_HARTREE_BOHR,
)

from milo.scientific_constants import h_JOULE_SEC as PLANCKS_CONSTANT

from AaronTools.fileIO import FileReader
from AaronTools.geometry import Geometry
from AaronTools.theory import ForceJob, TDDFTJob
try:
    import unixmilo.qm.cioverlap.wf_overlap as wf_overlap
    import unixmilo.mqc.el_propagator.el_run as el_run
    WITH_UNIXMD = True
except ModuleNotFoundError:
    WITH_UNIXMD = False



def get_program_handler(program_state, nonadiabatic=False, spinorbit=False):
    """Return the configured electronic structure program handler."""
    if program_state.program_id is en.ProgramID.MLP:
        if not nonadiabatic:
            return MLPHandler(program_state.executable)
        raise NotImplementedError("nonadiabatic dynamics not currently supported with MLP")

    raise NotImplementedError(f'Unknown electronic structure program '
                     f'"{program_state.program_id}"')


class ProgramHandler:
    def __init__(self, executable):
        self.executable = executable

    def generate_forces(self, ps: ProgramState, state=None, *args, **kwargs):
        """overwrite this method to return forces"""
        raise NotImplementedError

    @staticmethod
    def remove_files(*args):
        for f in args:
            try:
                os.remove(f)
            except FileNotFoundError:
                pass


class MLPHandler(ProgramHandler):
    """for running MLP jobs"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = None
        self.system = None

    def generate_forces(self, ps: ProgramState, state=None, energy_only=False):
        """Preform computation and append forces to list in program state."""
        if state is not None:
            job_name = f"mlp_{ps.current_step}_{state}"
        else:
            job_name = f"mlp_{ps.current_step}"
        mlp_file = self.call_mlp(
            job_name, ps, state=state,
        )
        self.parse_forces(mlp_file, ps, state=state)
        if ps.remove_files:
            self.remove_files(f"{job_name}.xyz", f"{job_name}.mlp")
    
    def call_mlp(self, job_name, ps: ProgramState, state=None):
        """Call MLP and return a string with the name of the mlp output file."""
    
        # Define file names
        job_xyz_file = f"{job_name}.xyz"
        job_mlp_file = f"{job_name}.mlp"

        # Prepare the XYZ file
        self.prepare_xyz_file(job_xyz_file, ps, state=state)

        # Load the MLP potential only once
        if self.mlp is None:
            with open(job_xyz_file, 'r') as f:
                atoms = read(f)
            self.system = mlt.System(mlt.Molecule(job_xyz_file, charge=1, mult=1), box=None)
            self.mlp = mlt.potentials.MACE('/home/yusefa/bin/sandwich_DSDPBEP86D3BJ_ccpvtz', system=self.system)
            mlt.Config.mace_params = {
                'calc_device': 'cuda',
            }

        # Update the atoms with the new XYZ file
        with open(job_xyz_file, 'r') as f:
            atoms = read(f)

        # Set the calculator for the atoms
        atoms.set_calculator(self.mlp.ase_calculator)

        # Calculate energy and forces
        energy_ev = atoms.get_potential_energy()
        forces_ev_angstrom = atoms.get_forces()
        energy_hartree = energy_ev * ELECTRON_VOLT_TO_HARTREE
        forces_hartree_bohr = forces_ev_angstrom * ELECTRON_VOLT_ANGSTROM_TO_HARTREE_BOHR
        
        with open(job_mlp_file, "w") as outfile:
            outfile.write("-------\n")
            outfile.write("MILO\n")
            outfile.write("-------\n")
            outfile.write("MLP\n")
            outfile.write("Input (angstrom):\n")
            
            for atom in atoms:
                symbol = atom.symbol
                pos = atom.position
                outfile.write(f"{symbol:<2}    {pos[0]:13.9f}   {pos[1]:13.9f}   {pos[2]:13.9f}\n")


            outfile.write("\n -------------------------------------------------------------------\n")
            outfile.write("Forces (Hartrees/Bohr)\n")

            
            for i, force in enumerate(forces_hartree_bohr):
                symbol = atoms[i].symbol
                atomic_number = ase_atomic_numbers[symbol]
                outfile.write(f"{symbol:<2}    {force[0]:13.9f}   {force[1]:13.9f}   {force[2]:13.9f}\n")

            outfile.write(f"Energy = {energy_hartree:.9f} Hartrees\n")

            outfile.write("\nNormal termination\n")
        
        return job_mlp_file
    

    @staticmethod
    def prepare_xyz_file(
        file_name,
        ps: ProgramState,
        state=None,
    ):
        """Prepare a .xyz file for a MLP run."""
        ps.geometry.coords = ps.structures[-1].as_angstrom()
        kwargs = dict()
        
        ps.geometry.write(outfile=file_name, **kwargs)
    def parse_forces(self, mlp_file_name, ps: ProgramState, state=None):
        """Parse forces into program state from the given mlp file."""
        fr = FileReader(mlp_file_name, just_geom=False)
        if not fr["finished"]:
            raise ElectronicStructureProgramError(
                "MLP force calculation log file was not valid. MLP "
                "returned an error or could not be called correctly.")

        forces = containers.Forces()
        for i, fv in enumerate(fr["forces"]):
            forces.append(*fv, en.ForceUnit.HARTREE_PER_BOHR)
            if any(i in ps.molecule_groups[j] for j in ps.frozen):
                continue
            force_mag = np.linalg.norm(fv)
            if force_mag > 1:
                raise RuntimeError(
                    f"extreme force detected ({force_mag} Eh/a0)\n" \
                    "this is likely the result of either an erroneous\n" \
                    "MLP issue, or a bad structure\n"
                    "If you believe this is a MLP issue, simply restart\n" \
                    "the MD run" \
                )
        if state is not None:
            ps.state_energies[-1][state] = containers.Energy(fr["energy"], en.EnergyUnit.HARTREE)
            ps.state_forces[-1][state] = forces
        else:
            ps.potential_energies.append(
                containers.Energy(fr["energy"], en.EnergyUnit.HARTREE)
            )
            ps.forces.append(forces)

        for key in ps.print_info:
            try:
                ps.other_info[key]
                ps.other_info[key] = [ps.other_info[key]]
            except KeyError:
                pass

            try:
                ps.other_info[key] = fr[key]
            except (KeyError, AttributeError):
                print("data not found:", key)

        del fr