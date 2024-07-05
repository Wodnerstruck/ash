import subprocess as sp
import os
import shutil
import time
import multiprocessing as mp
import numpy as np
import glob
import copy
import re

import ash.modules.module_coords
from ash.functions.functions_general import ashexit,insert_line_into_file,BC,print_time_rel, print_line_with_mainheader, pygrep2, pygrep, search_list_of_lists_for_index,print_if_level
from ash.modules.module_singlepoint import Singlepoint
from ash.modules.module_coords import check_charge_mult
#from ash.functions.functions_elstructure import xdm_run, calc_cm5
import ash.functions.functions_elstructure
import ash.constants
import ash.settings_ash
import ash.functions.functions_parallel

#XEDA Thoery class
class XEDATheory:
    def __init__(self, xedadir=None, filename='XEDA', label='XEDA', xedainput=None, method='hf', numcores=1):
        self.theorynamelabel = 'XEDA'
        self.label = label
        self.theorytype = 'QM'
        #self.analytic_gradient = False
        #self.analytic_hessian = False
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        if xedainput is not None:
            print(f"{self.theorynamelabel}Theory requires a xedainput keyword")
            ashexit()
        #Finding XEDA
        if xedadir is None:
            print(BC.WARNING, f"No xedadir argument passed to {self.theorynamelabel}Theory. Attempting to find xedadir variable inside settings_ash", BC.END)
            try:
                self.xedadir=ash.settings_ash.settings_dict["xedadir"]
            except:
                print(BC.WARNING,"Found no xedadir variable in settings_ash module either.",BC.END)
                try:
                    self.xedadir = os.path.dirname(shutil.which('xeda'))
                    print(BC.OKGREEN,"Found xeda in PATH. Setting xeda to:", self.xedadir, BC.END)
                except:
                    print(BC.FAIL,"Found no xeda executable in PATH. Exiting... ", BC.END)
                    ashexit()
        else:
            self.xedaidr = xedadir

        self.filename=filename
        self.xedainput=xedainput
        self.method=method
        self.numcores=numcores

    def set_numcores(self, numcores):
        self.numcores=numcores

    def cleanup(self):
        print(f"{self.theorynamelabel} cleanup not yet implemented.")


    #Run function
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
        elems=None, Grad=False, PC=False, numcores=None, restart=False, label=None,
        charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores        
        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        if charge == None or mult == None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()
        print("Job label:", label)
        print(f"Creating inputfile: {self.filename}.inp")
        print(f"{self.theorynamelabel} input:")
        print(self.xedainput)


        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()
        #What elemlist to use. If qm_elems provided then QM/MM job, otherwise use elems list
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems
        #Write PC to disk
        if PC is True:
            print("pc true")
            create_XEDA_pcfile_general(current_MM_coords,MMcharges, filename=self.filename)
            pcfile=self.filename+'.dat'
        else:
            pcfile=None

        #Write inputfile 
        write_XEDA_input(self.xedainput,charge,mult,qm_elems,current_coords, method=self.method,
                Grad=Grad, PCfile=pcfile, filename=self.filename)
        
        #Run XEDA
        run_XEDA(self.xedaidr, self.filename, numcores=numcores)

        #Grab energy
        self.energy = grab_XEDA_energy(self.filename+'.out')
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)

        if Grad is True:
            self.gradient = grab_XEDA_gradient(self.filename+'.out', len(current_coords))
            #Grab PC gradient
            if PC is True:
                self.pcgradient = grab_pcgradient_XEDA(self.filename+'.out', len(current_MM_coords))
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient, self.pcgradient
            else:
                print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
                return self.energy, self.gradient
        else:
            print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
            return self.energy
    


################################
# Independent XEDA functions
################################

def run_XEDA(xedadir, filename):
    raise NotImplementedError('run XEDA is not implemented yet')

def grab_XEDA_energy(filename):
    grabline = '********'

    with open(filename, 'r') as f:
        for line in f:
            if grabline in line:
                energy = float(re.split(r'\s+', line)[-1]) 
        return energy
    raise NotImplementedError('grab XEDA energy is not implemented yet')
def grab_XEDA_terms(filename):
    energy_read = False
    E_ele = 0.0
    E_ex = 0.0
    E_rep = 0.0
    E_corr = 0.0
    E_disp = 0.0
    tot_E_disp = 0.0
    E_tot = 0.0
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'ALL BASIS SET' in line:
                energy_read = True
                continue
            if energy_read and 'ELECTROSTATIC ENERGY' in line:
                E_ele += float(line.strip().split()[-1])
            if energy_read and 'EXCHANGE ENERGY' in line:
                E_ex += float(line.strip().split()[-1])
            if energy_read and 'REPULSION ENERGY' in line:
                E_rep += float(line.strip().split()[-1])  
            #if energy_read and 'GRIMME DISP CORRECTION' in line:
                # E_disp += float(line.strip().split()[-1])
            if energy_read and 'ELECTRON CORRELATION' in line:
                E_corr += float(line.strip().split()[-1])
            if energy_read and 'TOTAL INTERACTION ENERGY' in line:
                E_tot += float(line.strip().split()[-1])
            if energy_read and 'AB_DISP:' in line:
                E_disp += float(line.strip().split()[-1])
            if energy_read and 'TOT_DISP:' in line:
                tot_E_disp += float(line.strip().split()[-1])
            if energy_read and 'ele_AD' in line or 'ele_BC' in line or 'ele_CD' in line:
                E_ele += float(line.strip().split()[-2])
    return E_ele, E_ex, E_rep, E_corr, E_tot, tot_E_disp# kcal/mol
    raise NotImplementedError('grab XEDA terms is not implemented yet')

def grab_XEDA_gradient(filename, numatoms):
    raise NotImplementedError('grab XEDA gradient is not implemented yet')

def grab_pcgradient_XEDA(filename, numatoms):
    raise NotImplementedError('grab pcgradient XEDA is not implemented yet')

def write_XEDA_input(xedainput,charge,mult,elems,coords, filename='xeda',
    PCfile=None, Grad=False, method='hf'):
    raise NotImplementedError('write XEDAinput is not implemented yet')
    
def create_XEDA_pcfile_general(coords,pchargelist,filename='charge'):
    raise NotImplementedError('create XEDAPCfile is not implemented yet')