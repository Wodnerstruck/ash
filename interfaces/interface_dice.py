import subprocess as sp
import time
import numpy as np
import os
import sys
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to Dice: QMC and NEVPT2
class DiceTheory:
    def __init__(self, dicedir=None, pyscftheoryobject=None, filename='input.dat', printlevel=2,
                numcores=1, NEVPT2=False, AFQMC=False, trialWF=None, frozencore=True,
                SHCI_numdets=1000,
                dt=0.005, nsteps=50, nblocks=1000, nwalkers_per_proc=5):

        self.theorynamelabel="Dice"
        self.theorytype="QM"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")
        
        if dicedir == None:
            print(BC.WARNING, f"No dicedir argument passed to {self.theorynamelabel}Theory. Attempting to find dicedir variable inside settings_ash", BC.END)
            try:
                print("settings_ash.settings_dict:", ash.settings_ash.settings_dict)
                self.dicedir=ash.settings_ash.settings_dict["dicedir"]
            except:
                print(BC.WARNING,"Found no dicedir variable in settings_ash module either.",BC.END)
                print("Exiting")
                ashexit()
        else:
            self.dicedir = dicedir

        #Check for PySCFTheory object 
        if pyscftheoryobject is None:
            print("Error:No pyscftheoryobject was provided. This is required")
            ashexit()
        
        #Path to Dice binary
        self.dice_binary=self.dicedir+"/bin/Dice"
        #Put Dice script dir in path
        sys.path.insert(0, dicedir+"/scripts")
        #Import various functionality
        try:
            import QMCUtils
            self.QMCUtils=QMCUtils
        except:
            print("Problem import QMCUtils. Dice directory is probably incorrect")
            ashexit()
        #SHCI pyscf plugin
        try:
            from pyscf.shciscf import shci
            self.shci=shci
        except:
            print("Problem importing pyscf.sciscf")
            ashexit()

        if numcores > 1:
            try:
                print(f"MPI-parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
                check_OpenMPI()
            except:
                print("Problem with mpirun")
                ashexit()
        
        #Printlevel
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        self.pyscftheoryobject=pyscftheoryobject
        self.NEVPT2=NEVPT2
        self.AFQMC=AFQMC
        self.trialWF=trialWF
        self.SHCI_numdets=SHCI_numdets
        self.frozencore=frozencore
        self.dt=dt
        self.nsteps=nsteps
        self.nblocks=nblocks
        self.nwalkers_per_proc=nwalkers_per_proc

        #Print stuff
        print("Printlevel:", self.printlevel)
        print("Num cores:", self.numcores)
        print("PySCF object:", self.pyscftheoryobject)
        print("NEVPT2:", self.NEVPT2)
        print("AFQMC:", self.AFQMC)
        print("Frozencore:", self.frozencore)
        if self.AFQMC is True:
            print("trialWF:", self.trialWF)
            if self.trialWF is 'SHCI':
                print("SHCI_numdets:", self.SHCI_numdets)
            print("QMC settings:")
            print("dt:", self.dt)
            print("Number of steps per block (nsteps):", self.nsteps)
            print("Number of blocks (nblocks):", self.nblocks)
            print("Number of walkers per proc:", self.nwalkers_per_proc)
    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("not ready")
        #print(f"Deleting old checkpoint file: {self.checkpointfilename}")
        #files=[self.checkpointfilename]
        #for file in files:
        #    try:
        #        os.remove(file)
        #    except:
        #        pass
    def determine_frozen_core(self,elems):
        print("Determining frozen core based on system list of elements")
        #Main elements 
        FC_elems={'H':0,'He':0,'Li':0,'Be':0,'B':2,'C':2,'N':2,'O':2,'F':2,'Ne':2,
        'Na':2,'Mg':2,'Al':10,'Si':10,'P':10,'S':10,'Cl':10,'Ar':10,
        'K':10,'Ca':10,'Sc':10,'Ti':10,'V':10,'Cr':10,'Mn':10,'Fe':10,'Co':10,'Ni':10,'Cu':10,'Zn':10,
        'Ga':18,'Ge':18,'As':18,'Se':18, 'Br':18, 'Kr':18}
        #NOTE: To be updated for 4d TM row etc
        num_el=0
        for el in elems:
            num_el+=FC_elems[el]
        self.frozen_core_el=num_el
        self.frozen_core_orbs=int(num_el/2)
        print("Total frozen electrons in system:", self.frozen_core_el)
        print("Total frozen orbitals in system:", self.frozen_core_orbs)

    #Write dets.bin file. Requires running SHCI once more to get determinants
    def run_and_write_dets(self,numdets):
        print("Calling run_and_write_dets")
        #Run once more 
        self.shci.dryrun(self.pyscftheoryobject.mch)
        self.shci.writeSHCIConfFile(self.pyscftheoryobject.mch.fcisolver, self.pyscftheoryobject.mch.nelecas, False)
        with open(self.pyscftheoryobject.mch.fcisolver.configFile, 'a') as f:
            f.write(f'writebestdeterminants {numdets}\n\n')
        self.run_shci_directly()

    def run_shci_directly(self):
        print("Calling SHCI PySCF interface")
        #Running Dice via SHCI-PySCF interface
        self.shci.executeSHCI(self.pyscftheoryobject.mch.fcisolver)
    # run_dice_directly: In case we need to. Currently unused
    def run_dice_directly(self):
        print("Calling Dice executable directly")
        #For calling Dice directly when needed
        print(f"Running Dice with ({self.numcores} MPI processes)")
        with open('output.dat', "w") as outfile:
            sp.call(['mpirun', '-np', str(self.numcores), self.dice_binary, self.filename], stdout=outfile)

    # Run function. Takes coords, elems etc. arguments and computes E or E+G.
    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
            elems=None, Grad=False, Hessian=False, PC=False, numcores=None, restart=False, label=None,
            charge=None, mult=None):
        module_init_time=time.time()
        if numcores == None:
            numcores = self.numcores

        print(BC.OKBLUE, BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        #Checking if charge and mult has been provided
        if charge == None or mult == None:
            print(BC.FAIL, f"Error. charge and mult has not been defined for {self.theorynamelabel}Theory.run method", BC.END)
            ashexit()

        print("Job label:", label)

        #Coords provided to run
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

        #Cleanup before run.
        self.cleanup()

        #Run PySCF to get integrals

        self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)
        print("pyscftheoryobject:", self.pyscftheoryobject.__dict__)

        #Get frozen-core
        if self.frozencore is True:
            self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0


        #NEVPT2 or AFQMC
        if self.NEVPT2 is True:
            print("Running Dice NEVPT2 calculation on multiconfigurational WF")
            print("Not ready")
            mc=self.pyscftheoryobject.mch
            self.QMCUtils.run_nevpt2(mc, nelecAct=None, numAct=None, norbFrozen=self.frozen_core_orbs,
               integrals="FCIDUMP.h5", nproc=numcores, seed=None,
               fname="nevpt2.json", foutname='nevpt2.out', nroot=0,
               spatialRDMfile=None, spinRDMfile=None, stochasticIterNorms=1000,
               nIterFindInitDets=100, numSCSamples=10000, stochasticIterEachSC=100,
               fixedResTimeNEVPT_Ene=False, epsilon=1.0e-8, efficientNEVPT_2=True,
               determCCVV=True, SCEnergiesBurnIn=50, SCNormsBurnIn=50,
               diceoutfile="dice.out")

            #TODO: Grab energy from function call
            self.energy=0.0

        elif self.AFQMC is True:
            print("Running Dice AFQMC")
            #QMCUtils.run_afqmc(mc, ndets = 100, norb_frozen = norb_frozen)
            if self.trialWF == 'SHCI':
                print("Using multiconfigurational WF via SHCI")
                mc=self.pyscftheoryobject.mch
                
                #Get dets.bin file
                print("Running SHCI (via PySCFTheory object) once again to write dets.bin")
                self.run_and_write_dets(self.SHCI_numdets)

                #Phaseless AFQMC with hci trial
                e_afqmc, err_afqmc = self.QMCUtils.run_afqmc_mc(mc, mpi_prefix=f"mpirun -np {numcores}",
                                norb_frozen=self.frozen_core_orbs, chol_cut=1e-5,
                                ndets=self.SHCI_numdets, nroot=0, seed=None,
                                dt=self.dt, steps_per_block=self.nsteps, nwalk_per_proc=self.nwalkers_per_proc,
                                nblocks=self.nblocks, ortho_steps=20, burn_in=50,
                                cholesky_threshold=1.0e-3, weight_cap=None, write_one_rdm=False,
                                use_eri=False, dry_run=False)
                e_afqmc=e_afqmc[0] 
                err_afqmc=err_afqmc[0]
            else:
                print("Using single-determinant WF from PySCF object")
                mf=self.pyscftheoryobject.mf
                #Phaseless AFQMC with simple mf trial
                e_afqmc, err_afqmc = self.QMCUtils.run_afqmc_mf(mf, mpi_prefix=f"mpirun -np {numcores}",
                    norb_frozen=self.frozen_core_orbs, chol_cut=1e-5, seed=None, dt=self.dt,
                    steps_per_block=self.nsteps, nwalk_per_proc=self.nwalkers_per_proc, nblocks=self.nblocks,
                    ortho_steps=20, burn_in=50, cholesky_threshold=1.0e-3,
                    weight_cap=None, write_one_rdm=False, dry_run=False)
            self.energy=e_afqmc
            self.error=err_afqmc
            ##Analysis
            print("Final Dice AFQMC energy:", self.energy)
            print("error:", self.error)
            print(f"Error: {self.error} Eh ({self.error*harkcal} kcal/mol)")
        else:
            print("Unknown Dice run option")
            ashexit()

        print("Dice is finished")

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


