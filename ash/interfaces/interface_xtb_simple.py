import os
import shutil
import numpy as np
import subprocess as sp
import time
import ash.constants
import ash.settings_solvation
import ash.settings_ash
from ash.functions.functions_general import ashexit, blankline,reverse_lines, print_time_rel,BC, print_line_with_mainheader,print_if_level
import ash.modules.module_coords
from ash.modules.module_coords import elemstonuccharges, check_multiplicity, check_charge_mult


class xTBTheory_simple:
    
    def __init__(self, xtbdir=None, xtbmethod='GFN1', runmode='inputfile', numcores=1, printlevel=2, filename='xtb_', maxiter=500, eletronic_temp=300, label=None, accuracy=0.1, hardness_PC=99, solvent=None):
        self.theorynamelabel="xTB"
        self.theorytype = "QM"
        
        self.hardness=hardness_PC    
        self.accuracy=accuracy
        self.printlevel=printlevel
        
        self.label=label #label to distinguish different xtb objects
        self.filename=filename
        self.numcores=numcores
        self.xtbmethod=xtbmethod
        self.maxiter=maxiter
        self.runmode=runmode
        self.electronic_temp=eletronic_temp
        
        print_line_with_mainheader("xTB INTERFACE ")
        #print("Runmode", self.runmode)
        print("Method", self.xtbmethod)
        print("xTB object numcores", self.numcores)
        os.environ['OMP_NUM_THREADS'] = str(self.numcores)  
        
        if xtbdir is None:
            print(BC.WARNING, "No xtbdir argument passed to xTBTheory. Attempting to find xtbdir ", BC.END)
            
            try:
                self.xtbdir = os.path.dirname(shutil.which('xtb'))
                print(BC.OKGREEN,"Found xtb in path. Setting xtbdir to:", self.xtbdir, BC.END)
            except:
                print("Found no xtb executable in path. Exiting... ")
                ashexit()
        else:
            self.xtbdir = xtbdir
            
        if solvent != None:
                self.solvent_line="--alpb {}".format(solvent)
        else:
            self.solvent_line=""
    
    def set_numcores(self, numcores):
        self.numcores=numcores
        
    def cleanup(self):
        if self.printlevel >= 2:
            print("Cleaning up old xTB files")
        files=[self.filename + '.xyz',self.filename + '.out','xtbopt.xyz','xtbopt.log','xtbrestart','molden.input','charges','pcgrad','wbo','xtbinput','pcharge','xtbtopo.mol']

        for file in files:
            try:
                os.remove(file)
            except:
                pass
    
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None, printlevel=None,
                elems=None, Grad=False, PC=False, numcores=None, label=None, charge=None, mult=None):
        module_init_time=time.time()
        if MMcharges is None:
            MMcharges=[]

        if numcores is None:
            numcores=self.numcores

        if self.printlevel >= 2:
            print("------------STARTING XTB INTERFACE-------------")
        if current_coords is not None:
            pass
        else:
            print("no current_coords")
            ashexit()
            
        if charge == None or mult == None:
            print(BC.FAIL, "Error. charge and mult has not been defined for xTBTheory.", BC.END)
            ashexit()
            
        if qm_elems is None:
            if elems is None:
                print("No elems provided")
                ashexit()
            else:
                qm_elems = elems
        
        check_multiplicity(qm_elems,charge,mult)
        if self.runmode=='inputfile':
            if self.printlevel >=2:
                print("Using inputfile-based xTB interface")
            #TODO: Add restart function so that xtbrestart is not always deleted
            #Create XYZfile with generic name for xTB to run
            #inputfilename="xtb-inpfile"
            if self.printlevel >= 2:
                print("Creating inputfile:", self.filename+'.xyz')
            num_qmatoms=len(current_coords)
            num_mmatoms=len(MMcharges)
            
            ash.modules.module_coords.write_xyzfile(qm_elems, current_coords, self.filename,printlevel=self.printlevel)
            
            # Run inputfile
            if self.printlevel >= 2:
                print("------------Running xTB-------------")
                print("Running xTB using {} cores".format(self.numcores))
                print("...")
            
            if PC:
                create_xtb_pcfile_general(current_MM_coords, MMcharges, hardness=self.hardness)
                run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter, printlevel=self.printlevel,
                                      electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line, numcores=numcores)
            else:
                run_xtb_SP_serial(self.xtbdir, self.xtbmethod, self.filename + '.xyz', charge, mult, maxiter=self.maxiter, printlevel=self.printlevel,
                                      electronic_temp=self.electronic_temp, accuracy=self.accuracy, solvent=self.solvent_line, numcores=numcores)
            if self.printlevel >= 2:
                print("------------xTB calculation done-----")
            outfile=self.filename+'.out'
            self.energy=xtbfinalenergygrab(outfile)
            if self.printlevel >= 2:
                print("xtb energy :", self.energy)
                print("------------ENDING XTB-INTERFACE-------------")
            print_time_rel(module_init_time, modulename='xTB run', moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.energy
            
def create_xtb_pcfile_general(coords, pchargelist,hardness=99,elems=None):
    with open('pccharge','w') as pcfile:
        pcfile.write(str(len(pchargelist)) + '\n')
        for p, c in zip(pchargelist,coords):
            line = "{} {} {} {}".format(p, c[0], c[1], c[2], hardness)#待修改  
    
def run_xtb_SP_serial(xtbdir, xtbmethod, xyzfile, charge, mult, Grad=False, Opt=False, Hessian=False, maxiter=500, electronic_temp=300, accuracy=0.1, solvent=None, printlevel=2, numcores=1):
    
    if solvent is None:
        solvent_line = ""
    else:
        solvent_line = solvent
    
    basename = xyzfile.split('.')[0]
    uhf = mult - 1
    with open('xtbinput', 'w') as xfile:
        xfile.write('$embedding\n')
        xfile.write('interface=orca\n')
        xfile.write('end\n')

    if 'GFN2' in xtbmethod.upper():
        xtbflag = 2
    elif 'GFN1' in xtbmethod.upper():
        xtbflag = 1
    elif 'GFN0' in xtbmethod.upper():
        xtbflag = 0
    else:
        print("Unknown xtbmethod chosen. Exiting...")
        ashexit()
    command_list=[xtbdir + '/xtb', basename + '.xyz', str(solvent_line), '--gfn', str(xtbflag), '--chrg', str(charge), '--uhf', str(uhf), '--iterations', str(maxiter),
                      '--etemp', str(electronic_temp),  '--acc', str(accuracy), '--parallel', str(numcores), '--input', 'xtbinput']
    if printlevel >= 1:
        print("The running command:", command_list) 
    try:
        with open(basename+'.out', 'w') as ofile:
            process = sp.run(command_list, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
            if process.returncode == 0:
                print_if_level(f"xTB job succeeded.",printlevel,2)
                return
    except sp.CalledProcessError:
        print("xTB subprocess gave error.")
        if Hessian == True:
            if os.path.exists("hessian"):
                print("Hessian file was still created, ignoring error and continuing.")
                return
            else:
                print("Hessian file was not created. Check xtb output for error")
                ashexit()
        else:
            #Some other error. Restarting without xtbrestart (in case a bad one) and trying again.
            print("Something went wrong with xTB. ")
            #TODO: Check for SCF convergence?
            print("Removing xtbrestart MO-file and trying to run again")
            try:
                os.remove("xtbrestart")
            except FileNotFoundError:
                print("Nof xtbrestart file present")
            shutil.copyfile(basename+'.out', basename+'_firstrun.out')
            try:
                with open(basename+'.out', 'w') as ofile:
                    process = sp.run(command_list, check=True, stdout=ofile, stderr=ofile, universal_newlines=True)
                if process.returncode == 0:
                    return
            except:
                print("Still an xtb problem. Exiting. Check xtb outputfile")
                ashexit()
    else:
        print("some other error")
        print("process:", process)
        print("process returncode", process.returncode)
        ashexit()
def xtbfinalenergygrab(file):
    energy=None
    with open(file) as f:
        for line in f:
            if 'TOTAL ENERGY' in line:
                energy=float(line.split()[-3])
    return energy
    


