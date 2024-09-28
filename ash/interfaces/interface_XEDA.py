import subprocess as sp
import os
import shutil
import time
import numpy as np
import glob
import copy
import re

import ash.modules.module_coords
from ash.functions.functions_general import ashexit, insert_line_into_file, BC, print_time_rel, print_line_with_mainheader, pygrep2, pygrep, search_list_of_lists_for_index, print_if_level
from ash.modules.module_singlepoint import Singlepoint
from ash.modules.module_coords import check_charge_mult
# from ash.functions.functions_elstructure import xdm_run, calc_cm5
import ash.functions.functions_elstructure
import ash.constants
import ash.settings_ash
import ash.functions.functions_parallel
# XEDA Thoery class


class XEDATheory:
    def __init__(self, numcores=1, printlevel=2, label='xeda',
                 scf_type=None, basis=None, basis_file=None, ecp=None, functional=None, filename='xeda',
                 scf_maxiter=128, eda=False, ct=False, blw=False, eda_atm=None, eda_charge=None, eda_mult=None,
                 bc=None):
        # self.analytic_gradient = False
        # self.analytic_hessian = False
        self.theorynamelabel = "XEDA"
        self.theorytype = "QM"
        self.printlevel = printlevel
        print_line_with_mainheader("XEDATheory initialization")
        if scf_type is None:
            print(
                "Error: You must select an scf_type, e.g. 'RHF', 'UHF','ROHF', 'RKS', 'UKS', 'ROKS'")
            ashexit()
        if basis is None:
            print(
                "Error: You must provide basis or basis_file keyword . Basis set can a name (string)")
            ashexit()
        if functional is not None:
            if self.printlevel >= 1:
                print(f"Functional keyword: {functional} chosen. DFT is on!")
            if scf_type == 'RHF':
                print("Changing RHF to RKS")
                scf_type = 'RKS'
            if scf_type == 'UHF':
                print("Changing UHF to UKS")
                scf_type = 'UKS'
        else:
            if scf_type == 'RKS' or scf_type == 'UKS':
                print("Error: RKS/UKS chosen but no functional. Exiting")
                ashexit()
        self.properties = {}
        self.label = label
        self.filename = filename

        # SCF
        self.scf_type = scf_type
        self.basis = basis
        self.functional = functional
        self.basis_file = basis_file
        self.scf_maxiter = scf_maxiter
        self.ecp = ecp
        # background charge
        self.bc = bc
        # EDA
        self.eda = eda
        self.blw = blw
        self.ct = ct
        if not eda and (eda_atm is not None or eda_charge is not None or eda_mult is not None):
            raise ValueError(
                "You have set eda to False but provided eda_atm, eda_charge or eda_mult. Please set eda to True or remove these keywords.")
        if eda:
            if eda_atm is None or eda_charge is None or eda_mult is None:
                raise ValueError(
                    "You have set eda to True but did not provide eda_atm, eda_charge or eda_mult. Please set eda to False or provide these keywords.")
            if not len(eda_atm) == len(eda_charge) == len(eda_mult):
                raise ValueError(
                    "You have provided eda_atm, eda_charge and eda_mult but they are not the same length. Please check.")

            self.eda_atm, self.eda_charge, self.eda_mult = eda_atm, eda_charge, eda_mult

        self.numcores = numcores
        # print the options
        if self.printlevel >= 1:
            print("SCF-type:", self.scf_type)
            print("SCF Max Iter:", self.scf_maxiter)
            print("Basis:", self.basis)
            print("Basis-file:", self.basis_file)
            print("Functional:", self.functional)
            print("DM-EDA:", self.eda)
            if self.ct == True:
                print("Charge-transfer:", self.ct)
            print("BLW-EDA:", self.blw)

    def set_numcores(self, numcores):
        self.numcores = numcores
        print("Setting XEDA numcores to: ", self.numcores)
        # import py_api.mole as mole
        # mole.xscf_world.set_thread_num(numcores)

    def set_DFT_options(self):
        import py_api.scf as scf
        self.hf = scf.scf_info(self.mol)
        self.hf.init_hf()
        self.hf.init_guess_sad()
        if self.blw is not True:
            if self.functional is not None:
                self.hf.load_dft(self.functional)

    def set_embedding_options(self, PC=False, MM_coords=None, MMcharges=None):
        if PC is True:
            import py_api.builder as builder
            # QM/MM pointcharge embedding
            print("PC True. Adding pointcharges")
            MMcharges = np.array(MMcharges)
            MM_coords = np.array(MM_coords)
            self.bc = builder.charge_info(self.mol, [np.ravel(np.column_stack(
                (MMcharges[:, np.newaxis], MM_coords * ash.constants.ang2bohr)))])
            self.hf.load_ham(self.bc, 1.0, 0.0)


    def create_mol(self, qm_elems, current_coords, charge, mult):
        if self.printlevel >= 1:
            print("Creating mol object")
        import py_api.mole as mole
        if not self.eda:
            coords_string = ash.modules.module_coords.create_coords_string_xscf(
                qm_elems, current_coords)
            self.mol = mole.mol_info(
                f'{charge} {mult};\n' + coords_string, bas_str=self.basis)
        else:
            coords_string = ash.modules.module_coords.create_coords_string_xeda(
                qm_elems, current_coords, self.eda_atm, self.eda_charge, self.eda_mult)
            self.mol = mole.mol_info(coords_string, bas_str=self.basis)

    def run_SCF(self):
        if self.printlevel >= 1:
            print("\nrun_SCF")
        module_init_time = time.time()
        if self.scf_type == 'RHF' or self.scf_type == 'RKS':
            self.hf.do_scf()
        elif self.scf_type == 'UHF' or self.scf_type == 'UKS':
            self.hf.do_scf("u")
        elif self.scf_type == 'ROHF' or self.scf_type == 'ROKS':
            raise NotImplementedError(
                "ROHF/ROKS functionality is not yet implemented.")
        print_time_rel(module_init_time, modulename='XEDA run_SCF',
                       moduleindex=2, currprintlevel=self.printlevel, currthreshold=2)
        return self.hf.tol_energy[0]

    def run_EDA(self):
        if self.printlevel >= 1:
            print("\nrun_DM-EDA")
        module_init_time = time.time()
        import py_api.eda as eda
        if self.functional is not None:
            self.eda_obj = eda.eda_info(self.mol, dft=self.functional)
        else:
            self.eda_obj = eda.eda_info(self.mol)

        self.eda_obj.do_eda(self.hf.tol_energy, self.hf.d_matrix)
        # self.eda_obj.do_eda_atomic(self.hf.d_matrix, None)
        self.eda_obj.show()
        print_time_rel(module_init_time, modulename='XEDA run_EDA',
                       moduleindex=2, currprintlevel=self.printlevel, currthreshold=2)
        if self.ct is not True:
            return (self.eda_obj.ES, self.eda_obj.EX, self.eda_obj.REP, self.eda_obj.POL, self.eda_obj.EC, self.eda_obj.TOL)
        else:
            raise NotImplementedError(
                "Charge-transfer functionality is not yet implemented.")

    def run_BLW(self):
        if self.printlevel >= 1:
            print("\nrun_BLW")
        module_init_time = time.time()
        self.blw_obj.do_eda()
        print_time_rel(module_init_time, modulename='XEDA run_BLW',
                       moduleindex=2, currprintlevel=self.printlevel, currthreshold=2)
        raise NotImplementedError("BLW functionality is not yet implemented.")

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None):
        self.prepare_run(current_coords=current_coords, elems=elems, charge=charge, mult=mult,
                         current_MM_coords=current_MM_coords,
                         MMcharges=MMcharges, qm_elems=qm_elems, Grad=Grad, PC=PC,
                         numcores=numcores, pe=pe, potfile=potfile, restart=restart, label=label)
        # Actual run
        return self.actualrun(current_coords=current_coords, current_MM_coords=current_MM_coords, MMcharges=MMcharges, qm_elems=qm_elems,
                              elems=elems, Grad=Grad, PC=PC, numcores=numcores, pe=pe, potfile=potfile, restart=restart, label=label,
                              charge=charge, mult=mult)

    def prepare_run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
                    elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
                    charge=None, mult=None):

        module_init_time = time.time()
        if self.printlevel > 0:
            print(BC.OKBLUE, BC.BOLD,
                  "------------PREPARING XEDA INTERFACE-------------", BC.END)
            print("Object-label:", self.label)
            print("Run-label:", label)

            import py_api.mole as mole
            mole.xscf_world.set_thread_num(self.numcores)

            if self.printlevel > 1:
                print("Number of XEDA  threads is:", self.numcores)

                # Checking if charge and mult has been provided
        if self.eda is False and (charge == None or mult == None):
            print(
                BC.FAIL, "Error. charge and mult has not been defined for XEDATheory.run method", BC.END)
            ashexit()
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()

        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems
        #####################
        # CREATE MOL OBJECT
        #####################
        self.create_mol(qm_elems, current_coords, charge, mult)
        #####################
        # DFT
        #####################
        self.set_DFT_options()
        ##############################
        # EMBEDDING OPTIONS
        ##############################
        self.set_embedding_options(
            PC=PC, MM_coords=current_MM_coords, MMcharges=MMcharges)

        if self.printlevel > 1:
            print_time_rel(module_init_time,
                           modulename='XEDA prepare', moduleindex=2)

    def actualrun(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
                  elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
                  charge=None, mult=None):

        module_init_time = time.time()
        #############################################################
        # RUNNING
        #############################################################

        #####################
        # SCF STEP
        #####################
        if self.eda is not True and self.blw is not True:
            if self.printlevel > 1:
                print(f"Running SCF (SCF-type: {self.scf_type})")
            self.energy = self.run_SCF()
            if self.bc is not None:
                self.bc.cal_energy(self.hf.d_matrix)
            if self.printlevel >= 0:
                print("Single-point XEDA energy:", self.energy)
                if self.bc is not None:
                    print("QM_charge interaction energy:", self.bc.energy[0])
                print_time_rel(module_init_time,
                               modulename='XEDA actualrun', moduleindex=2)

            return self.energy + self.bc.energy[0] if self.bc is not None else self.energy
        elif self.eda is True:
            if self.printlevel > 1:
                print(f"Running SCF (SCF-type: {self.scf_type})")
            e_tol = self.run_SCF()
            eda_results = self.run_EDA()
            return eda_results
        elif self.blw is True:
            blw_results = self.run_BLW()
            return blw_results

        if self.printlevel >= 1:
            print()
            print(BC.OKBLUE, BC.BOLD,
                  "------------ENDING XEDA INTERFACE-------------", BC.END)
