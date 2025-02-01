from dataclasses import dataclass
from ash import Fragment, XEDATheory, OpenMMTheory, QMMMTheory, XEDA_EB, QMMMTheory_EDA, \
    ashexit, Singlepoint, Energy_decomposition
#from ash import xTBTheory
#import openmm.app as app
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
    hl_atoms: List[int]
    charge: int
    mult: int
    eda_atm: List[int]
    eda_charge: List[int]
    eda_mult: List[int]
    gfn: int = 1
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


class DFTxTB_EDA():
    def __init__(self, config: DMEBConfig, printlevel=2):
        self.config = config
        self.printlevel = printlevel
        print_line_with_mainheader("DFT-DFTB_EDA initialization")
        self.validate_config()
        self.initialize_fragment()
        self.qm_theory_full = XEDATheory(
            numcores=self.config.numcores,
            scf_type=self.config.scf_type,
            basis=self.config.basis,
            functional=self.config.functional
        )
        self.dftdftb_eda_results: DMEBEDAterms = DMEBEDAterms()

    def validate_config(self):
        if self.config.charge != sum(self.config.eda_charge):
            raise ValueError(
                "Sum of charge of monomers do not equal to total charge.")

    def initialize_fragment(self):
        if self.config.xyzfile is not None:
            self.full_fragment = Fragment(
                xyzfile=self.config.xyzfile, charge=self.config.charge, mult=self.config.mult)

    def split_fragment(self, xyzfile: str, split_indices: List[int]):
        coord = []
        elem = []
        with open(xyzfile, 'r') as f:
            lines = f.readlines()
            try:
                num_atoms = int(lines[0])
            except ValueError:
                print(f"Error: XYZ-file {xyzfile} does not have a valid number of atoms in the first line.")
                print(f"Line: {lines[0]}")
                ashexit()
            if sum(split_indices) != num_atoms:
                print("Sum of NumAtoms of monomers do not equal to total NumAtoms.")
                ashexit()
            
            for line in lines[2:]:
                if len(line) > 3:
                    line_split = line.split()
                    elem.append(line_split[0])
                    coord.append([float(line_split[1]), float(line_split[2]), float(line_split[3])])
            if len(elem) != num_atoms:
                print("Number of atoms in header not equal to number of coordinate-lines. Check XYZ file!")
                ashexit()

        with open('fragment1.xyz', 'w') as f1:
            f1.write(str(split_indices[0]) + "\n\n")
            for i in range(split_indices[0]):
                f1.write(elem[i] + " " + str(coord[i][0]) + " " + str(coord[i][1]) + " " + str(coord[i][2]) + "\n")
        
        with open('fragment2.xyz', 'w') as f2:
            f2.write(str(split_indices[1]) + "\n\n")
            for i in range(split_indices[1]):
                f2.write(elem[i + split_indices[0]] + " " + str(coord[i + split_indices[0]][0]) + " " +
                         str(coord[i + split_indices[0]][1]) + " " + str(coord[i + split_indices[0]][2]) + "\n")

    def split_hlatms(self):
        split_point = self.config.eda_atm[0]
        self.hl_atoms1 = list(filter(lambda x: x < split_point, self.config.hl_atoms))
        self.hl_atoms2 = list(map(lambda x: x - split_point, filter(lambda x: x >= split_point, self.config.hl_atoms))) 
        
    