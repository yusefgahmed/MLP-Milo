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
    
