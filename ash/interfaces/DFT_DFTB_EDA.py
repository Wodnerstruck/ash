from dataclasses import dataclass
from ash import Fragment, XEDATheory, OpenMMTheory, QMMMTheory, XEDA_EB, QMMMTheory_EDA, \
    ashexit, Singlepoint, Energy_decomposition
from ash import xTBTheory
import openmm.app as app
import numpy as np
import copy
import time
import ash.modules.module_coords
import ash.constants
from typing import List, Optional
from ash.functions.functions_general import ashexit, BC, blankline, listdiff, print_time_rel, printdebug, print_line_with_mainheader, writelisttofile, print_if_level
import ash.settings_ash
from ash.modules.module_QMMM import fullindex_to_qmindex, linkatom_force_fix


@dataclass
class DMEBConfig:
    numcores: int
    functional: str
    basis: str
    high_level_atoms: List[int]
    charge: int
    mult: int
    eda_atm: List[int]
    eda_charge: List[int]
    eda_mult: List[int]
    scf_type: str = 'RHF'
    xyzfile: Optional[str] = None


@dataclass
class DMEBEDAterms:
    ele: float = 0.0
    ex: float = 0.0
    rep: float = 0.0
    pol: float = 0.0
    ec: float = 0.0
    tot: float = 0.0


class DFT_DFTB():
    def __init__(self, config: DMEBConfig, printlevel=2):
        self.config = config
        self.printlevel = printlevel
        raise NotImplementedError()
    raise NotImplementedError()
