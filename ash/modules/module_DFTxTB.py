import time
import numpy as np
import os
import copy
from collections import defaultdict
import math

import ash
from ash.functions.functions_general import BC, ashexit, print_time_rel, print_line_with_mainheader, listdiff, printdebug
from ash.modules.module_theory import Theory
from ash.interfaces.interface_ORCA import grabatomcharges_ORCA
from ash.interfaces.interface_xtb import grabatomcharges_xTB, grabatomcharges_xTB_output
from ash.modules.module_QMMM import linkatom_force_contribution, linkatom_force_chainrule


class DFTxTBTheory(Theory):
    def __init__(self, theories_N=None, regions_N=None, regions_chargemult=None,
                 embedding="mechanical", full_pointcharges=None, chargemodel="CM5", dipole_correction=False,
                 fullregion_charge=None, fullregion_mult=None, fragment=None, label=None,
                 chargeboundary_method="chargeshift", excludeboundaryatomlist=None,
                 linkatom_method='ratio', linkatom_simple_distance=None, linkatom_forceproj_method="adv",
                 linkatom_ratio=0.723, linkatom_type='H', printlevel=2, numcores=1):
        super().__init__()
        self.theorytype = "DFTxTB"
        self.printlevel = printlevel
        self.label = label
        self.filename = ""
        self.theorynamelabel = "DFTxTBTheory"
        print_line_with_mainheader("DFTxTB Theory")
        if fragment is None:
            print("Error: fragment= keyword has not been defined . Exiting")
            ashexit()
        if fullregion_charge is None or fullregion_mult is None:
            print("Error: Full-region charge and multiplicity must be provided (fullregion_charge, fullregion_mult keywords)")
            ashexit()

        if not isinstance(theories_N, list):
            print("Error: theories_N should be defined and be a list")
            ashexit()
        print(f"{len(theories_N)} theories provided. This is a {len(theories_N)}-layer ONIOM.")
        if regions_N is None:
            print("Error: regions_N must be provided for N-layer ONIOM")
            ashexit()
        if regions_chargemult is None:
            print("Error: regions_chargemult must be provided for N-layer ONIOM (list of lists of charge,mult for each region)")
            ashexit()
        if len(theories_N) != len(regions_N):
            print("Error: Number of theories and regions must match")
            ashexit()
        if len(theories_N) != len(regions_chargemult):
            print("Error: Number of theories and regions_chargemult must match")
            ashexit()

        self.fragment = fragment
        self.allatoms = self.fragment.allatoms
        self.theories_N = theories_N
        self.regions_N = regions_N
        self.regions_chargemult = regions_chargemult

        self.linkatoms = False
        # Linkatom method strategy to determine linkatom position or QM-L distance
        self.linkatom_type = linkatom_type  # Usually 'H'
        self.linkatom_method = linkatom_method  # Options: 'simple' or 'ratio'
        self.linkatom_simple_distance = linkatom_simple_distance  # For method simple, Default 1.09 Angstrom
        # For method ratio. see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9314059/
        self.linkatom_ratio = linkatom_ratio
        # Linkatom projection method Options: 'adv', 'lever', 'chain', 'none'
        self.linkatom_forceproj_method = linkatom_forceproj_method
        if self.linkatom_forceproj_method is None:
            self.linkatom_forceproj_method = "none"
       # Embedding
        # Note: by default no embedding, meaning LL theory for everything
        self.embedding = embedding
        self.chargemodel = chargemodel  # TODO: check this function
        self.chargeboundary_method = chargeboundary_method

        # Defining pointcharges for full system
        self.full_pointcharges = full_pointcharges
        # Dipole correction for charge-shifting
        self.dipole_correction = dipole_correction

        # N-layer ONIOM
        self.fullregion_charge = fullregion_charge
        self.fullregion_mult = fullregion_mult

        # Defining charge/mult here as well
        self.charge = self.fullregion_charge
        self.mult = self.fullregion_mult

        print("Embedding:", self.embedding)
        print("Theories:")
        for i, t in enumerate(self.theories_N):
            print(f"Theory {i+1}: {t.theorynamelabel} . Numcores: {t.numcores}")

        if numcores != 1:
            print(f"Numcores keyword (numcores={numcores}) was set for ONIOMTheory.")
            self.numcores = numcores
            print(
                f"Warning: ONIOM will use {numcores} cores in general for all Theories (overriding numcores in Theory objects)")
            for i, t in enumerate(self.theories_N):
                t.numcores = numcores
                print(f"Warning: Setting numcores={numcores} for Theory {i+1}: {t.theorynamelabel}")
        else:
            print("Warning: numcores attribute was not set for ONIOMTheory.")
            print("This is fine, but check if numcores settings above are appropriate for each Theory object")
            self.numcores = 1

        print("\nRegions provided:")
        #
        for i, r in enumerate(self.regions_N):
            print(f"Region {i+1} ({len(r)} atoms):", r)
        print("Allatoms:", self.allatoms)
        print("\nRegion-chargemult info provided:")
        #
        for i, r in enumerate(self.regions_chargemult):
            print(f"Region {i+1} Charge:{r[0]} Mult:{r[1]}")

        # REGIONS AND BOUNDARY
        conn_scale = 1.0
        conn_tolerance = 0.2
        if len(self.theories_N) == 2:
            self.theorylabels = ["HL", "LL"]
            # Atom labels
            atomlabels = ["HL" if b in self.regions_N[0] else "LL" for b in self.fragment.allatoms]
            # If HL-LL covalent boundary issue and ASH exits then printing QM-coordinates is useful
            print("\nFull-system coordinates (before any linkatoms):")
            ash.modules.module_coords.print_coords_for_atoms(
                self.fragment.coords, self.fragment.elems, self.fragment.allatoms, labels=atomlabels)
            print()
            self.boundaryatoms = ash.modules.module_coords.get_boundary_atoms(
                self.regions_N[0],
                self.fragment.coords,
                self.fragment.elems,
                conn_scale,
                conn_tolerance,
                excludeboundaryatomlist=excludeboundaryatomlist,
                unusualboundary=None)
        # Checking for covalent boundary
        if len(self.theories_N) == 2 and len(self.boundaryatoms) > 0:
            print("Found covalent boundary.")
            print("Boundaryatoms (HL:LL pairs):", self.boundaryatoms)
            print(
                "Note: used connectivity settings, scale={} and tol={} to determine boundary.".format(
                    conn_scale, conn_tolerance))
            self.linkatoms = True
            # Get MM boundary information. Stored as self.MMboundarydict
            self.get_MMboundary(self.boundaryatoms, conn_scale, conn_tolerance)

        else:
            print("No covalent boundary.")
            self.linkatoms = False
            self.dipole_correction = False
            self.MMboundary_indices = []

    def create_linkatoms(self, current_coords, region_atoms, elems):
        checkpoint = time.time()
        # Get linkatom coordinates
        # NOTE: Option to change linkatom_distance, now 1.08736
        self.linkatoms_dict = ash.modules.module_coords.get_linkatom_positions(
            self.boundaryatoms,
            region_atoms,
            current_coords,
            elems,
            linkatom_type=self.linkatom_type,
            linkatom_method=self.linkatom_method,
            linkatom_simple_distance=self.linkatom_simple_distance,
            linkatom_ratio=self.linkatom_ratio)
        print("linkatoms_dict:", self.linkatoms_dict)
        if self.printlevel > 1:
            print("Adding linkatom positions to region coords")
        self.linkatom_indices = [len(region_atoms) + i for i in range(0, len(self.linkatoms_dict))]
        self.num_linkatoms = len(self.linkatom_indices)
        linkatoms_coords = [self.linkatoms_dict[pair] for pair in sorted(self.linkatoms_dict.keys())]

        print_time_rel(
            checkpoint,
            modulename='create_linkatoms',
            moduleindex=3,
            currprintlevel=self.printlevel,
            currthreshold=2)
        return linkatoms_coords

    def ZeroQMCharges(self, atoms):
        print("Setting Region charges to Zero")
        # Looping over charges and setting region atoms to zero
        # 1. Copy charges to charges_qmregionzeroed
        charges_qmregionzeroed = copy.copy(self.charges)
        # 2. change charge for QM-atom
        for i, c in enumerate(charges_qmregionzeroed):
            # Setting QMatom charge to 0
            if i in atoms:
                charges_qmregionzeroed[i] = 0.0
        # 3. Flag that this has been done
        self.ChargesZeroed = True

        return charges_qmregionzeroed

    def ShiftMMCharges(self, charges_qmregionzeroed):
        if self.printlevel > 1:
            print("Shifting MM charges at ONIOM boundary.")
        # Convert lists to NumPy arrays for faster computations
        pointcharges = np.array(charges_qmregionzeroed)
        self.charges = np.array(self.charges)

        # Extract charges for MM boundary atoms
        MM1_charges = self.charges[self.MMboundary_indices]
        # Set charges of MM boundary atoms to 0
        pointcharges[self.MMboundary_indices] = 0.0

        # Calculate charge fractions to distribute
        MM1charge_fract = MM1_charges / self.MMboundary_counts

        # Charge-shifting method
        # Distribute charge fractions to neighboring MM atoms
        for indices, fract in zip(self.MMboundarydict.values(), MM1charge_fract):
            pointcharges[[indices]] += fract

        self.chargeshifting_done=True
        return pointcharges
    # From QM1:MM1 boundary dict, get MM1:MMx boundary dict (atoms connected to MM1)
    def get_MMboundary(self, boundaryatoms, scale, tol):
        timeA = time.time()
        # if boundarydict is not empty we need to zero MM1 charge and distribute charge from MM1 atom to MM2,MM3,MM4
        # Creating dictionary for each MM1 atom and its connected atoms: MM2-4
        self.MMboundarydict = {}
        for (QM1atom, MM1atom) in boundaryatoms.items():
            connatoms = ash.modules.module_coords.get_connected_atoms(
                self.fragment.coords, self.fragment.elems, scale, tol, MM1atom)
            # Deleting QM-atom from connatoms list
            connatoms.remove(QM1atom)
            self.MMboundarydict[MM1atom] = connatoms

        # Used by ShiftMMCharges
        self.MMboundary_indices = list(self.MMboundarydict.keys())
        self.MMboundary_counts = np.array([len(self.MMboundarydict[i]) for i in self.MMboundary_indices])

        print("")
        print("MM boundary (MM1:MMx pairs):", self.MMboundarydict)
        print_time_rel(timeA, modulename="get_MMboundary")
        
    def run(self, current_coords=None, Grad=False, elems=None, charge=None, mult=None, label=None, numcores=None):

        print(BC.OKBLUE,BC.BOLD, f"------------RUNNING {self.theorynamelabel} INTERFACE-------------", BC.END)
        # Full coordinates
        full_coords=current_coords
        full_elems=elems
        # Dicts to keep energy and gradient for each theory-region combo
        E_dict={} # (theory,region) -> energy
        G_dict={} # (theory,region) -> gradient
        num_theories = len(self.theories_N)
        # First doing LowLevel (LL) theory on Full region
        ll_theory = self.theories_N[-1]
        print(f"Running Theory LL ({ll_theory.theorynamelabel}) on Full-region ({len(full_elems)} atoms)")

        # Derive pointcharges unless full_pointcharges were already provided
        if self.full_pointcharges is None and self.embedding.lower() == "elstat":
            print("Electrostatic embedding but no full-system pointcharges provided yet")
            print("This means that we must derive pointcharges for full system")
            # TODO: How to do this in general
            # Check if the low-level theory is compatible with some charge model
            # Should probably do this in init instead though
            if isinstance(ll_theory, ash.ORCATheory):
                print(f"Theory is ORCATheory. Using {self.chargemodel} charge model")
                if self.chargemodel == "CM5" or self.chargemodel.lower() == "hirshfeld":
                    ll_theory.extraline+="\n! hirshfeld "
                else:
                    print("Unknown charge model")
                    ashexit()
            elif isinstance(ll_theory, ash.xTBTheory_simple):
                 print(f"Theory is xTBTheory. Using default xtb charge model")
            else:
                print("Problem: Theory-level not compatible with pointcharge-creation")
                ashexit()
        ###############################################
        # RUN FULL REGION
        ###############################################
        # Copying theory object for full-region to avoid interference with other regions
        ll_theory_full = copy.deepcopy(ll_theory)
        label = "LL_full"
        ll_theory_full.filename = f"{label}"
        # RUN FULL
        res_full = ll_theory_full.run(current_coords=full_coords,
                                    elems=full_elems, Grad=Grad, numcores=ll_theory.numcores,
                                    label=label, charge=self.fullregion_charge, mult=self.fullregion_mult)
        if Grad:
            e_LL_full,g_LL_full = res_full
        else:
            e_LL_full = res_full

        # Grabbing atom charges from ORCA output
        # TODO: Remove theory-specific code
        if self.embedding.lower() == "elstat" and self.full_pointcharges is None:
            print("Grabbing atom charges for whole system")
            print("Chargemodel:", self.chargemodel)
            if isinstance(ll_theory_full, ash.ORCATheory):
                self.full_pointcharges = grabatomcharges_ORCA(self.chargemodel,f"{ll_theory_full.filename}.out")
                # Remove ll_theory.extraline from ORCATheory for next LL calculations
                ll_theory.extraline=""
            elif isinstance(ll_theory_full, ash.xTBTheory_simple):
                print(f"{ll_theory_full.filename}.out")
                # Note: format issue
                # self.full_pointcharges = grabatomcharges_xTB_output(ll_theory.filename+'.out', chargemodel=self.chargemodel)
                self.full_pointcharges = grabatomcharges_xTB()
            print("self.full_pointcharges:", self.full_pointcharges)
            print(len(self.full_pointcharges))

        if self.embedding.lower() == "elstat":
            # Defining charges for full system
            self.charges=self.full_pointcharges
            print("Num full system charges:", len(self.charges))
            print("Full system charges:", self.charges)

        E_dict[(num_theories-1,-1)] = e_LL_full
        if Grad:
            G_dict[(num_theories-1,-1)] = g_LL_full