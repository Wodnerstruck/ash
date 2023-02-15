import subprocess as sp
import shutil
import time
import numpy as np
import os
import sys
import glob
from ash.modules.module_coords import elematomnumbers, check_charge_mult
from ash.constants import ang2bohr, harkcal
from ash.functions.functions_general import ashexit, BC, print_time_rel,print_line_with_mainheader,pygrep,pygrep2,find_program
from ash.functions.functions_parallel import check_OpenMPI
import ash.settings_ash

#Interface to Block: Block2 primarily via PySCF and also directly via FCIdump
# Possibly later both Block 1.5 and Stackblock via PySCF

#Block 2 docs: https://block2.readthedocs.io
#BLock 1.5 docs: https://pyscf.org/Block/with-pyscf.html


class BlockTheory:
    def __init__(self, blockdir=None, pyscftheoryobject=None, blockversion='Block2', filename='input.dat', printlevel=2, numcores=1, 
                moreadfile=None, initial_orbitals='MP2', memory=20000, frozencore=True, fcidumpfile=None, 
                active_space=None, active_space_range=None, cas_nmin=None, cas_nmax=None, macroiter=0,
                Block_direct=False, maxM=1000, tol=1e-10, scratchdir=None):

        self.theorynamelabel="Block"
        self.theorytype="QM"
        self.blockversion=blockversion
        exename="block2main"
        print_line_with_mainheader(f"{self.theorynamelabel}Theory initialization")

        #Check for PySCFTheory object 
        if pyscftheoryobject is None:
            print("Error: No pyscftheoryobject was provided. This is required")
            ashexit()

        #MAKING SURE WE HAVE BLOCK
        #TODO: Look for regular block binary also here??
        print("Using Blockversion:", self.blockversion)        
        #Look up directory based on blockdir variable (takes precedence), "blockdir" in ash.settings or finally exename
        self.blockdir = find_program(blockdir,"blockdir",exename,self.theorynamelabel)
        print("self.blockdir:", self.blockdir)
        #Path to Block binary
        self.block_binary=self.blockdir+"/bin/"+exename
        #TODO: Check that binary is executable. Otherwise bin might be wrong
        print("self.block_binary:", self.block_binary)

        #Now import pyscf and dmrgscf (interface between pyscf and various DMRG programs)
        self.load_pyscf()
        self.load_dmrgscf()
        #self.load_block_15()
        #Telling dmrgscf interface where the block binary is
        #self.dmrgscf.settings.BLOCKEXE = os.popen("which block2main").read().strip()
        self.dmrgscf.settings.BLOCKEXE = self.block_binary
        print("self.dmrgscf.settings.BLOCKEXE:", self.dmrgscf.settings.BLOCKEXE)
        if self.blockversion == 'Block2':
            #Using Block2 internal NEVPT2 approach
            self.dmrgscf.settings.BLOCKEXE_COMPRESS_NEVPT = self.block_binary

        if numcores > 1:
            try:
                print(f"MPI-parallel job requested with numcores: {numcores} . Make sure that the correct OpenMPI version is available in your environment")
                check_OpenMPI()
            except:
                print("Problem with mpirun")
                ashexit()
        if scratchdir == None:
            print("Scratchdir not set")
            print("Setting scratchdir to current dir (hopefully a local scratch drive)")
            scratchdir='.'
        #Printlevel
        self.scratchdir=scratchdir
        self.printlevel=printlevel
        self.filename=filename
        self.numcores=numcores
        #SETTING NUMCORES by setting prefix
        self.dmrgscf.settings.MPIPREFIX = f'mpirun -n {self.numcores} --bind-to none'
        self.dmrgscf.settings.BLOCKSCRATCHDIR = self.scratchdir
        self.pyscftheoryobject=pyscftheoryobject


        self.moreadfile=moreadfile
        self.macroiter=macroiter
        self.Block_direct=Block_direct
        self.maxM=maxM
        self.tol=tol

        self.fcidumpfile=fcidumpfile
        self.active_space=active_space
        self.active_space_range=active_space_range
        self.cas_nmin=cas_nmin
        self.cas_nmax=cas_nmax
        self.frozencore=frozencore
        self.memory=memory #Memory in MB (total) assigned to PySCF mcscf object
        self.initial_orbitals=initial_orbitals #Initial orbitals to be used (unless moreadfile option)
        #Print stuff
        print("Printlevel:", self.printlevel)
        print("Memory (MB):", self.memory)
        print("Num cores:", self.numcores)
        print("PySCF object:", self.pyscftheoryobject)

        print("Frozencore:", self.frozencore)
        print("moreadfile:", self.moreadfile)
        print("Initial orbitals:", self.initial_orbitals)
        if self.Block_direct is True:
            print("FCIDUMP file:", self.fcidumpfile)
    
    
    def load_pyscf(self):
        try:
            import pyscf
        except:
            print(BC.FAIL, "Problem importing pyscf. Make sure pyscf has been installed: pip install pyscf", BC.END)
            ashexit(code=9)
        self.pyscf=pyscf
        print("\nPySCF version:", self.pyscf.__version__)
    def load_dmrgscf(self):
        #DMRGSCF pyscf plugin
        try:
            from pyscf import dmrgscf
            self.dmrgscf=dmrgscf
        except ModuleNotFoundError:
            print("Problem importing dmrgscf (PySCF interface module to Block)")
            print("See: https://github.com/pyscf/dmrgscf/ on how to install dmrgscf module for pyscf")
            print("Most likely: pip install git+https://github.com/pyscf/dmrgscf")
            ashexit()
        except ImportError as err:
            print("Problem initializing dmrgscf interface")
            print("See: https://github.com/pyscf/dmrgscf/ on how to install dmrgscf module for pyscf")
            print("Most likely: pip install git+https://github.com/pyscf/dmrgscf")
            print("Errormessage:", err)
            print("Once dmrgscf is installed you have to create settings.py file (see path in pyscf import error message, probably above) and add this:")
            m=f"""
BLOCKEXE = "{self.blockdir}/bin/Block"
BLOCKEXE_COMPRESS_NEVPT = "" #path to block.spin_adapted. Can be left blank for now
BLOCKSCRATCHDIR = "." #path to scratch dir. Can be kept as . (but make sure you execute program in scratchdir)
MPIPREFIX = "" # mpi-prefix. Best to leave blank
            """
            print(m)
            ashexit()

    #Set numcores method
    def set_numcores(self,numcores):
        self.numcores=numcores
    def cleanup(self):
        print("Cleaning up Block temporary files")
        print("not ready")
        #bkpfiles=glob.glob('*.bkp')
        #for bkpfile in bkpfiles:
        #    os.remove(bkpfile)
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

    #Grabbing Dice variational energy
    def grab_var_energy(self):
        grab=False
        with open("output.dat") as f:
            for line in f:
                if len(line.split()) < 3:
                    grab=False
                if grab is True:
                    if len(line.split()) == 3:
                        energy=float(line.split()[1])
                if 'Root             Energy' in line:
                    grab=True
        return energy

    def grab_important_dets(self):
        grab=False
        det_strings=[]
        with open("output.dat") as f:
            for line in f:
                if len(line.split()) < 2:
                    grab=False
                if grab is True:
                    if len(line.split()) > 10:
                        det_strings.append(line)
                if ' Det     weight  Determinant string' in line:
                    grab=True
        return det_strings
    def grab_num_dets(self):
        grab=False
        numdet=0
        with open("output.dat") as f:
            for line in f:
                if 'Performing final tigh' in line:
                    grab=False
                if grab is True:
                    if len(line.split()) == 7:
                        numdet=int(line.split()[3])
                if 'Iter Root       Eps1   #Var. Det.               Ener' in line:
                    grab=True
        return numdet

    # call_block_directly : Call block directly if inputfile and FCIDUMP file already exists
    def call_block_directly(self):
        module_init_time=time.time()
        print("Calling Block executable directly")
        print(f"Running Block with ({self.numcores} MPI processes)")
        with open('output.dat', "w") as outfile:
            sp.call(['mpirun', '-np', str(self.numcores), self.block_binary, self.filename], stdout=outfile)
        print_time_rel(module_init_time, modulename='Block-direct-run', moduleindex=2)
    
    #Set up initial orbitals
    #This returns a set of MO-coeffs and occupations either from checkpointfile or from MP2/CC job
    def setup_initial_orbitals(self, elems):
        module_init_time=time.time()
        print("\n INITIAL ORBITAL OPTION")
        if len(self.pyscftheoryobject.mf.mo_occ) == 2:
            totnumborb=len(self.pyscftheoryobject.mf.mo_occ[0])
        else:
            totnumborb=len(self.pyscftheoryobject.mf.mo_occ)
        print(f"There are {totnumborb} orbitals in the system")
        #READ ORBITALS OR DO natural orbitals with MP2/CCSD/CCSD(T)
        if self.moreadfile == None:
            print("No checkpoint file given (moreadfile option).")
            print(f"Will calculate PySCF {self.initial_orbitals} natural orbitals to use as input in Dice CAS job")
            if self.initial_orbitals not in ['canMP2','MP2','DFMP2', 'DFMP2relax', 'CCSD','CCSD(T)', 'DMRG', 'AVAS-CASSCF', 'DMET-CASSCF','CASSCF']:
                print("Error: Unknown initial_orbitals choice. Exiting.")
                ashexit()
            print("Options are: MP2, CCSD, CCSD(T), DMRG, AVAS-CASSCF, DMET-CASSCF")
            #Option to do small-M DMRG-CASCI/CASSCF step 
            if self.initial_orbitals == 'DMRG':
                print("DMRG initial orbital option")
                print("First calculating MP2 natural orbitals, then doing DMRG-job")
                print("This is not ready")
                ashexit()
                #Call pyscftheory method for MP2,CCSD and CCSD(T)
                MP2nat_occupations, MP2nat_mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method='MP2', elems=elems)
                self.setup_active_space(occupations=MP2nat_occupations)
                self.setup_DMRG_job(verbose=5, rdmoption=True) #Creates the self.mch CAS-CI/CASSCF object with RDM True
                #self.SHCI_object_set_mos(mo_coeffs=MP2nat_mo_coefficients) #Sets the MO coeffs of mch object              
                self.DMRG_object_run(MP2nat_mo_coefficients) #Runs the self.mch object

                #????
                print("Done with DMRG-run for initial orbital step")
                print("Now making natural orbitals from DMRG WF")
                occupations, mo_coefficients = self.pyscf.mcscf.addons.make_natural_orbitals(self.mch)
                print("DMRG natural orbital occupations:", occupations)
            elif self.initial_orbitals == 'AVAS-CASSCF' or self.initial_orbitals == 'DMET-CASSCF':
                print("Calling calculate_natural_orbitals using AVAS/DMET method")

                if self.CAS_AO_labels == None:
                    print("Error: CAS_AO_labels are messing, provide keyword to DiceTheory!")
                    ashexit()

                occupations, mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method=self.initial_orbitals, 
                                                                CAS_AO_labels=self.CAS_AO_labels, elems=elems)
            else:
                print("Calling nat-orb option in pyscftheory")
                #Call pyscftheory method for MP2,CCSD and CCSD(T)
                occupations, mo_coefficients = self.pyscftheoryobject.calculate_natural_orbitals(self.pyscftheoryobject.mol,
                                                                self.pyscftheoryobject.mf, method=self.initial_orbitals,
                                                                elems=elems)

        else:
            print("Will read MOs from checkpoint file:", self.moreadfile)
            if '.chk' not in self.moreadfile:
                print("Error: not a PySCF chkfile")
                ashexit()
            mo_coefficients = self.pyscf.lib.chkfile.load(self.moreadfile, 'mcscf/mo_coeff')
            occupations = self.pyscf.lib.chkfile.load(self.moreadfile, 'mcscf/mo_occ')
            print("Chk-file occupations:", occupations)
            print("Length of occupations array:", len(occupations))
            if len(occupations) != totnumborb:
                print("Occupations array length does NOT match length of MO coefficients in PySCF object")
                print("Is basis different? Exiting")
                ashexit()

        #Check if occupations are sensible (may be nonsense if CCSD/CAS calc failed)
        if True in [i > 2.00001 for i in occupations]:
            print("Warning! Occupation array contains occupations larger than 2.0. Something possibly wrong (bad convergence?)")
            print("Continuing but these orbitals may be bad")
            #ashexit()
        if True in [i < 0.0 for i in occupations]:
            print("Warning!. Occupation array contains negative occupations. Something possibly wrong (bad convergence?)")
            print("Continuing but these orbitals may be bad")
        #    ashexit()
        print("Initial orbital step complete")
        print("----------------------------------")
        print()
        print_time_rel(module_init_time, modulename='Dice-Initial-orbital-step', moduleindex=2)
        return mo_coefficients, occupations

    #Determine active space based on either natural occupations of initial orbitals or 
    def setup_active_space(self, occupations=None):
        print("Inside setup_active_space")
        with np.printoptions(precision=3, suppress=True):
            print("Occupations:", occupations)

        # If active_space defined then this takes priority over natural occupations
        if self.active_space != None:
            print("Active space given as user-input: active_space=", self.active_space)
            # Number of orbital and electrons from active_space keyword!
            self.nelec=self.active_space[0]
            self.norb=self.active_space[1]
        elif self.active_space_range != None:
            #Convenvient when we have the orbitals we want but we can't define active_space because the electron-number changes (IEs)
            #TODO: Problem: Occupations depend on the system CHK file we read in. Electrons not matching anymore
            print("Active space range:", self.active_space_range)
            self.norb = len(occupations[self.active_space_range[0]:self.active_space_range[1]])
            self.nelec = round(sum(occupations[self.active_space_range[0]:self.active_space_range[1]]))
            print(f"Selected active space from range: CAS({self.nelec},{self.norb})")      
        else:
            print(f"Active space determined from {self.initial_orbitals} NO threshold parameters: cas_nmin={self.cas_nmin} and cas_nmax={self.cas_nmax}")

            if self.cas_nmin == None or self.cas_nmax == None:
                print("You need to set cas_nmin and cas_nmax parameters")
                ashexit()
            print("Note: Use active_space keyword if you want to select active space manually instead")
            # Determing active space from natorb thresholds
            nat_occs_for_thresholds=[i for i in occupations if i < self.cas_nmin and i > self.cas_nmax]
            indices_for_thresholds=[i for i,j in enumerate(occupations) if j < self.cas_nmin and j > self.cas_nmax]
            firstMO_index=indices_for_thresholds[0]
            lastMO_index=indices_for_thresholds[-1]
            self.norb = len(nat_occs_for_thresholds)
            self.nelec = round(sum(nat_occs_for_thresholds))
            print(f"To get this same active space in another calculation you can also do: \nactive_space=[{self.nelec},{self.norb}]")
            print(f"or: \nactive_space_range=[{firstMO_index},{lastMO_index}]")
        #Check if active space still not defined and exit if so
        if self.norb == None:
            print("No active space has been defined!")
            print("You probably need to provide active_space keyword!")
            ashexit()

    #Setup a DMRG CAS-CI or CASSCF job using the DMRGSCF interface
    def setup_DMRG_job(self,verbose=5, rdmoption=None):
        print(f"\nDoing DMRG-CAS calculation with {self.nelec} electrons in {self.norb} orbitals!")
        print("macroiter:", self.macroiter)

        #Turn RDM creation on or off
        #if rdmoption is False:
        #    #Pick user-selected (default: False)
        #    dordm=self.SHCI_DoRDM
        #else:
        #    #Probably from initial-orbitals call (RDM True)
        #    dordm=rdmoption
        #    print("RDM option:", dordm)

        if self.macroiter == 0:
            print("This is single-iteration CAS-CI via pyscf and DMRG")

            #self.mch = self.pyscf.mcscf.CASCI(self.pyscftheoryobject.mf,self.norb, self.nelec)
            self.mch = self.dmrgscf.DMRGSCF(self.pyscftheoryobject.mf, self.norb, self.nelec, maxM=self.maxM, tol=self.tol)
            print("Turning off canonicalization step in mcscf object")
            self.mch.canonicalization = False
        else:
            print("This is CASSCF via pyscf and DMRG (orbital optimization)")
            self.mch = self.pyscf.mcscf.CASSCF(self.pyscftheoryobject.mf,self.norb, self.nelec)
            self.mch.canonicalization = False
            self.mch.natorb = True
        #Settings
        #self.mch.fcisolver = self.shci.SHCI(self.pyscftheoryobject.mol)
        self.mch.fcisolver.mpiprefix = f'mpirun -np {self.numcores}'

        self.mch.verbose=verbose
        #Setting memory
        self.mch.max_memory= int(self.memory / 1000) # mem in GB
        print("Memory in pyscf object set to:", self.mch.max_memory)
        #CASSCF iterations
        self.mch.max_cycle_macro = self.macroiter        

    #Run the defined pyscf mch object
    def DMRG_run(self,mos):
        module_init_time=time.time()
        #Run DMRGSCF object created above
        print("Running DMRG via DMRGSCF interface in pyscf")
        result = self.mch.kernel(mos)

        print("result:", results)
        #Get final energy
        self.energy = result.energy

        print_time_rel(module_init_time, modulename=f'{self.blockversion}-run', moduleindex=2)
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

        #Run PySCF to get integrals and MOs. This would probably only be an SCF
        if self.Block_direct != True:
            self.pyscftheoryobject.run(current_coords=current_coords, elems=qm_elems, charge=charge, mult=mult)

        #Get frozen-core
        if self.frozencore is True:
            if self.Block_direct == None:
                self.determine_frozen_core(qm_elems)
        else:
            self.frozen_core_orbs=0

        # NOW RUNNING
        #TODO: Distinguish between Block2, Stackblock, Block1.5. 
        #TODO: Distinguish between direct run using FCIDUMP also
        if self.blockversion =='Block2':
            mo_coeffs, occupations = self.setup_initial_orbitals(elems) #Returns mo-coeffs and occupations of initial orbitals
            self.setup_active_space(occupations=occupations) #This will define self.norb and self.nelec active space
            self.setup_DMRG_job(verbose=5, rdmoption=None)
            self.DMRG_run(mo_coeffs)
        else:
            print("Unknown blockversion")
            ashexit()

        print("Block is finished")
        #Cleanup Block scratch stuff (big files)
        self.cleanup()

        print(BC.OKBLUE, BC.BOLD, f"------------ENDING {self.theorynamelabel} INTERFACE-------------", BC.END)
        print(f"Single-point {self.theorynamelabel} energy:", self.energy)
        print_time_rel(module_init_time, modulename=f'{self.theorynamelabel} run', moduleindex=2)
        return self.energy


