from dataclasses import dataclass
from ash import Fragment, XEDATheory, OpenMMTheory, QMMMTheory, ashexit, Singlepoint, Energy_decomposition
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
class QMMMConfig:
    numcores: int
    functional: str
    basis: str
    scf_tpye: str = 'RHF'
    # qm_theory_full: XEDATheory
    qm_atoms: List[int]
    charge: int
    mult: int
    eda_atm: List[int]
    eda_charge: List[int]
    eda_mult: List[int]
    pdbfile: Optional[str] = None
    xyzfile: Optional[str] = None
    forcefield: Optional[str] = None
    xmlfiles: Optional[List[str]] = None


class QMMMTheory_EDA(QMMMTheory):

    def __init__(self, qm_theory=None, qmatoms=None, fragment=None, mm_theory=None, charges=None,
                 embedding="elstat", printlevel=2, numcores=1, actatoms=None, frozenatoms=None,
                 excludeboundaryatomlist=None, unusualboundary=False, openmm_externalforce=False, TruncatedPC=False,
                 TruncPCRadius=55, TruncatedPC_recalc_iter=50, qm_charge=None, qm_mult=None,
                 dipole_correction=True, do_qm=True):
        super().__init__(qm_theory=qm_theory, qmatoms=qmatoms, fragment=fragment, mm_theory=mm_theory, charges=charges,
                         embedding=embedding, printlevel=printlevel, numcores=numcores, actatoms=actatoms,
                         frozenatoms=frozenatoms, excludeboundaryatomlist=excludeboundaryatomlist,
                         unusualboundary=unusualboundary, openmm_externalforce=openmm_externalforce,
                         TruncatedPC=TruncatedPC, TruncPCRadius=TruncPCRadius,
                         TruncatedPC_recalc_iter=TruncatedPC_recalc_iter,
                         qm_charge=qm_charge, qm_mult=qm_mult, dipole_correction=dipole_correction)
        self.do_qm = do_qm

    def elstat_run(self, current_coords=None, elems=None, Grad=False, numcores=1,
                   exit_after_customexternalforce_update=False, label=None, charge=None, mult=None):
        module_init_time = time.time()
        CheckpointTime = time.time()

        if self.printlevel >= 2:
            print("Embedding: Electrostatic")

        #############################################
        # If this is first run then do QM/MM runprep
        # Only do once to avoid cost in each step
        #############################################
        if self.runcalls == 0:
            print("First QMMMTheory run. Running runprep")
            self.runprep(current_coords)
            # This creates self.pointcharges, self.current_qmelems, self.mm_elems_for_qmprogram
            # self.linkatoms_dict, self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords

        # Updating runcalls
        self.runcalls += 1

        #########################################################################################
        # General QM-code energy+gradient call.
        #########################################################################################

        # Split current_coords into MM-part and QM-part efficiently.
        used_mmcoords, used_qmcoords = current_coords[~self.xatom_mask], current_coords[self.xatom_mask]

        if self.linkatoms is True:
            # Update linkatom coordinates. Sets: self.linkatoms_dict,
            # self.linkatom_indices, self.num_linkatoms, self.linkatoms_coords
            linkatoms_coords = self.create_linkatoms(current_coords)
            # Add linkatom coordinates to QM-coordinates
            used_qmcoords = np.append(
                used_qmcoords, np.array(linkatoms_coords), axis=0)

        # store used_qmcoords
        self.qm_coords = used_qmcoords  # for eda input with link_atoms
        # Update self.pointchargecoords based on new current_coords
        # print("self.dipole_correction:", self.dipole_correction)
        if self.dipole_correction:
            self.SetDipoleCharges(current_coords)  # Note: running again
            self.pointchargecoords = np.append(
                used_mmcoords, np.array(self.dipole_coords), axis=0)
        else:
            self.pointchargecoords = used_mmcoords

        # TRUNCATED PC Option: Speeding up QM/MM jobs of large systems by passing only a truncated PC field to the QM-code most of the time
        # Speeds up QM-pointcharge gradient that otherwise dominates
        # TODO: TruncatedPC is inactive
        if self.TruncatedPC is True:
            self.TruncatedPCfunction(used_qmcoords)

            # Modifies self.pointcharges and self.pointchargecoords
            # print("Number of charges after truncation :", len(self.pointcharges))
            # print("Number of charge coordinates after truncation :", len(self.pointchargecoords))

        # If numcores was set when calling QMMMTheory.run then using, otherwise use self.numcores
        if numcores == 1:
            numcores = self.numcores

        if self.printlevel > 1:
            print("Number of pointcharges (to QM program):",
                  len(self.pointcharges))
            print("Number of charge coordinates:", len(self.pointchargecoords))
        if self.printlevel >= 2:
            print("Running QM/MM object with {} cores available".format(numcores))
        ################
        # QMTheory.run
        ################
        print_time_rel(module_init_time, modulename='before-QMstep',
                       moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()
        if self.qm_theory_name == "None" or self.qm_theory_name == "ZeroTheory":
            print("No QMtheory. Skipping QM calc")
            QMenergy = 0.0
            self.linkatoms = False
            PCgradient = np.array([0.0, 0.0, 0.0])
            QMgradient = np.array([0.0, 0.0, 0.0])
        else:

            # TODO: Add check whether QM-code supports both pointcharges and pointcharge-gradient?

            # Calling QM theory, providing current QM and MM coordinates.
            if Grad is True:
                if self.PC is True:
                    QMenergy, QMgradient, PCgradient = self.qm_theory.run(current_coords=used_qmcoords,
                                                                          current_MM_coords=self.pointchargecoords,
                                                                          MMcharges=self.pointcharges,
                                                                          qm_elems=self.current_qmelems, mm_elems=self.mm_elems_for_qmprogram,
                                                                          charge=charge, mult=mult,
                                                                          Grad=True, PC=True, numcores=numcores)
                else:
                    QMenergy, QMgradient = self.qm_theory.run(current_coords=used_qmcoords,
                                                              current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                              qm_elems=self.current_qmelems, Grad=True, PC=False, numcores=numcores, charge=charge, mult=mult)
            else:
                if self.do_qm:
                    QMenergy = self.qm_theory.run(current_coords=used_qmcoords,
                                                  current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges,
                                                  mm_elems=self.mm_elems_for_qmprogram, qm_elems=self.current_qmelems,
                                                  Grad=False, PC=self.PC, numcores=numcores, charge=charge, mult=mult)
                else:
                    QMenergy = 0.0
                    print("Skipping QM step")

        print_time_rel(CheckpointTime, modulename='QM step', moduleindex=2,
                       currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        # Final QM/MM gradient. Combine QM gradient, MM gradient, PC-gradient (elstat MM gradient from QM code).
        # Do linkatom force projections in the end
        # UPDATE: Do MM step in the end now so that we have options for OpenMM extern force
        if Grad:
            Grad_prep_CheckpointTime = time.time()
            # assert len(self.allatoms) == len(self.MMgradient)
            # Defining QMgradient without linkatoms if present
            if self.linkatoms:
                self.QMgradient = QMgradient
                # remove linkatoms
                QMgradient_wo_linkatoms = QMgradient[0:-self.num_linkatoms]
            else:
                self.QMgradient = QMgradient
                QMgradient_wo_linkatoms = QMgradient

            # if self.printlevel >= 2:
            #    ash.modules.module_coords.write_coords_all(self.QMgradient_wo_linkatoms, self.qmelems, indices=self.allatoms, file="QMgradient_wo_linkatoms", description="QM+ gradient withoutlinkatoms (au/Bohr):")

            # TRUNCATED PC Option:
            if self.TruncatedPC is True:
                # DONE ONCE: CALCULATE FULL PC GRADIENT TO DETERMINE CORRECTION
                if self.TruncatedPC_recalc_flag is True:
                    CheckpointTime = time.time()
                    truncfullCheckpointTime = time.time()

                    # We have calculated truncated QM and PC gradient
                    QMgradient_trunc = QMgradient
                    PCgradient_trunc = PCgradient

                    print("Now calculating full QM and PC gradient")
                    print("Number of PCs provided to QM-program:",
                          len(self.pointcharges_full))
                    QMenergy_full, QMgradient_full, PCgradient_full = self.qm_theory.run(
                        current_coords=used_qmcoords,
                        current_MM_coords=self.pointchargecoords_full,
                        MMcharges=self.pointcharges_full,
                        qm_elems=self.current_qmelems, charge=charge, mult=mult,
                        Grad=True, PC=True, numcores=numcores)
                    print_time_rel(
                        CheckpointTime, modulename='trunc-pc full calculation', moduleindex=3)
                    CheckpointTime = time.time()

                    # TruncPC correction to QM energy
                    self.truncPC_E_correction = QMenergy_full - QMenergy
                    print(
                        f"Truncated PC energy correction: {self.truncPC_E_correction} Eh")
                    self.QMenergy = QMenergy + self.truncPC_E_correction
                    # Now determine the correction once and for all
                    CheckpointTime = time.time()
                    self.calculate_truncPC_gradient_correction(
                        QMgradient_full, PCgradient_full, QMgradient_trunc, PCgradient_trunc)
                    print_time_rel(
                        CheckpointTime, modulename='calculate_truncPC_gradient_correction', moduleindex=3)
                    CheckpointTime = time.time()

                    # Now defining final QMgradient and PCgradient
                    self.QMgradient_wo_linkatoms, self.PCgradient = self.TruncatedPCgradientupdate(
                        QMgradient_wo_linkatoms, PCgradient)
                    print_time_rel(
                        CheckpointTime, modulename='truncPC_gradient update ', moduleindex=3)
                    print_time_rel(
                        truncfullCheckpointTime, modulename='trunc-full-step pcgrad update', moduleindex=3)

                else:
                    CheckpointTime = time.time()
                    # TruncPC correction to QM energy
                    self.QMenergy = QMenergy + self.truncPC_E_correction
                    self.QMgradient_wo_linkatoms, self.PCgradient = self.TruncatedPCgradientupdate(
                        QMgradient_wo_linkatoms, PCgradient)
                    print_time_rel(
                        CheckpointTime, modulename='trunc pcgrad update', moduleindex=3)
            else:
                self.QMenergy = QMenergy
                # No TruncPC approximation active. No change to original QM and PCgradient from QMcode
                self.QMgradient_wo_linkatoms = QMgradient_wo_linkatoms
                if self.embedding.lower() == "elstat":
                    self.PCgradient = PCgradient

            # Populate QM_PC gradient (has full system size)
            CheckpointTime = time.time()
            self.make_QM_PC_gradient()  # populates self.QM_PC_gradient
            print_time_rel(CheckpointTime, modulename='QMpcgrad prepare',
                           moduleindex=3, currprintlevel=self.printlevel, currthreshold=2)
            # self.QM_PC_gradient = np.zeros((len(self.allatoms), 3))
            # qmcount=0;pccount=0
            # for i in self.allatoms:
            #    if i in self.qmatoms:
            #        #QM-gradient. Linkatom gradients are skipped
            #        self.QM_PC_gradient[i]=self.QMgradient_wo_linkatoms[qmcount]
            #        qmcount+=1
            #    else:
            #        #Pointcharge-gradient. Dipole-charge gradients are skipped (never reached)
            #        self.QM_PC_gradient[i] = self.PCgradient[pccount]
            #        pccount += 1
            # assert qmcount == len(self.qmatoms)
            # assert pccount == len(self.mmatoms)
            # assert qmcount+pccount == len(self.allatoms)

            # print(" qmcount+pccount:", qmcount+pccount)
            # print("len(self.allatoms):", len(self.allatoms))
            # print("len self.QM_PC_gradient", len(self.QM_PC_gradient))
            # ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM_PC_gradient", description="QM_PC_gradient (au/Bohr):")

            # if self.printlevel >= 2:
            #    ash.modules.module_coords.write_coords_all(self.PCgradient, self.mmatoms, indices=self.allatoms, file="PCgradient", description="PC gradient (au/Bohr):")

            # if self.printlevel >= 2:
            #    ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient_before_linkatomproj", description="QM+PC gradient before linkatomproj (au/Bohr):")

            # LINKATOM FORCE PROJECTION
            # Add contribution to QM1 and MM1 contribution???
            if self.linkatoms is True:
                CheckpointTime = time.time()

                for pair in sorted(self.linkatoms_dict.keys()):
                    printdebug("pair: ", pair)
                    # Grabbing linkatom data
                    linkatomindex = self.linkatom_indices.pop(0)
                    printdebug("linkatomindex:", linkatomindex)
                    Lgrad = self.QMgradient[linkatomindex]
                    printdebug("Lgrad:", Lgrad)
                    Lcoord = self.linkatoms_dict[pair]
                    printdebug("Lcoord:", Lcoord)
                    # Grabbing QMatom info
                    fullatomindex_qm = pair[0]
                    printdebug("fullatomindex_qm:", fullatomindex_qm)
                    printdebug("self.qmatoms:", self.qmatoms)
                    qmatomindex = fullindex_to_qmindex(
                        fullatomindex_qm, self.qmatoms)
                    printdebug("qmatomindex:", qmatomindex)
                    Qcoord = used_qmcoords[qmatomindex]
                    printdebug("Qcoords: ", Qcoord)

                    Qgrad = self.QM_PC_gradient[fullatomindex_qm]
                    printdebug("Qgrad (full QM/MM grad)s:", Qgrad)

                    # Grabbing MMatom info
                    fullatomindex_mm = pair[1]
                    printdebug("fullatomindex_mm:", fullatomindex_mm)
                    Mcoord = current_coords[fullatomindex_mm]
                    printdebug("Mcoord:", Mcoord)

                    Mgrad = self.QM_PC_gradient[fullatomindex_mm]
                    printdebug("Mgrad (full QM/MM grad): ", Mgrad)

                    # Now grabbed all components, calculating new projected gradient on QM atom and MM atom
                    newQgrad, newMgrad = linkatom_force_fix(
                        Qcoord, Mcoord, Lcoord, Qgrad, Mgrad, Lgrad)
                    printdebug("newQgrad: ", newQgrad)
                    printdebug("newMgrad: ", newMgrad)

                    # Updating full QM_PC_gradient (used to be QM/MM gradient)
                    # self.QM_MM_gradient[fullatomindex_qm] = newQgrad
                    # self.QM_MM_gradient[fullatomindex_mm] = newMgrad
                    self.QM_PC_gradient[fullatomindex_qm] = newQgrad
                    self.QM_PC_gradient[fullatomindex_mm] = newMgrad

            print_time_rel(CheckpointTime, modulename='linkatomgrad prepare',
                           moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            print_time_rel(Grad_prep_CheckpointTime, modulename='QM/MM gradient prepare',
                           moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            CheckpointTime = time.time()
        else:
            # No Grad
            self.QMenergy = QMenergy

        # if self.printlevel >= 2:
        #    ash.modules.module_coords.write_coords_all(self.QM_PC_gradient, self.elems, indices=self.allatoms, file="QM+PCgradient", description="QM+PC gradient (au/Bohr):")

        # MM THEORY
        if self.mm_theory_name == "NonBondedTheory":
            if self.printlevel >= 2:
                print("Running MM theory as part of QM/MM.")
                print(
                    "Using MM on full system. Charges for QM region  have to be set to zero ")
                # printdebug("Charges for full system is: ", self.charges)
                print(
                    "Passing QM atoms to MMtheory run so that QM-QM pairs are skipped in pairlist")
                print(
                    "Passing active atoms to MMtheory run so that frozen pairs are skipped in pairlist")
            assert len(current_coords) == len(self.charges_qmregionzeroed)

            # NOTE: charges_qmregionzeroed for full system but with QM-charges zeroed (no other modifications)
            # NOTE: Using original system coords here (not with linkatoms, dipole etc.). Also not with deleted zero-charge coordinates.
            # charges list for full system, can be zeroed but we still want the LJ interaction

            self.MMenergy, self.MMgradient = self.mm_theory.run(current_coords=current_coords,
                                                                charges=self.charges_qmregionzeroed, connectivity=self.connectivity,
                                                                qmatoms=self.qmatoms, actatoms=self.actatoms)

        elif self.mm_theory_name == "OpenMMTheory":
            if self.printlevel >= 2:
                print("Using OpenMM theory as part of QM/MM.")
            if self.QMChargesZeroed:
                if self.printlevel >= 2:
                    print("Using MM on full system. Charges for QM region {} have been set to zero ".format(
                        self.qmatoms))
            else:
                print("QMCharges have not been zeroed")
                ashexit()
            # printdebug("Charges for full system is: ", self.charges)
            # Todo: Need to make sure OpenMM skips QM-QM Lj interaction => Exclude
            # Todo: Need to have OpenMM skip frozen region interaction for speed  => => Exclude
            if Grad:
                CheckpointTime = time.time()
                # print("QM/MM Grad is True")
                # Provide QM_PC_gradient to OpenMMTheory
                if self.openmm_externalforce:
                    print_if_level(
                        f"OpenMM externalforce is True", self.printlevel, 2)
                    # Calculate energy associated with external force so that we can subtract it later
                    # self.extforce_energy = 3 * np.mean(np.sum(self.QM_PC_gradient * current_coords * 1.88972612546, axis=0))
                    scaled_current_coords = current_coords * 1.88972612546
                    self.extforce_energy = 3 * \
                        np.mean(np.sum(self.QM_PC_gradient *
                                scaled_current_coords, axis=0))
                    print_if_level(
                        f"Extforce energy: {self.extforce_energy}", self.printlevel, 2)
                    print_time_rel(CheckpointTime, modulename='extforce prepare',
                                   moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
                    # NOTE: Now moved mm_theory.update_custom_external_force call to MD simulation instead
                    # as we don't have access to simulation object here anymore. Uses self.QM_PC_gradient
                    if exit_after_customexternalforce_update:
                        print_if_level(
                            f"OpenMM custom external force updated. Exit requested", self.printlevel, 2)
                        # This is used if OpenMM MD is handling forces and dynamics
                        return

                self.MMenergy, self.MMgradient = self.mm_theory.run(
                    current_coords=current_coords, qmatoms=self.qmatoms, Grad=True)
            else:
                print("QM/MM Grad is false")
                self.MMenergy = self.mm_theory.run(
                    current_coords=current_coords, qmatoms=self.qmatoms)
        else:
            self.MMenergy = 0
        print_time_rel(CheckpointTime, modulename='MM step', moduleindex=2,
                       currprintlevel=self.printlevel, currthreshold=1)
        CheckpointTime = time.time()

        # Final QM/MM Energy. Possible correction for OpenMM external force term
        self.QM_MM_energy = self.QMenergy + self.MMenergy - self.extforce_energy
        if self.printlevel >= 2:
            blankline()
            if self.embedding.lower() == "elstat":
                print(
                    "Note: You are using electrostatic embedding. This means that the QM-energy is actually the polarized QM-energy")
                print(
                    "Note: MM energy also contains the QM-MM Lennard-Jones interaction\n")
            energywarning = ""
            if self.TruncatedPC is True:
                # if self.TruncatedPCflag is True:
                print(
                    "Warning: Truncated PC approximation is active. This means that QM and QM/MM energies are approximate.")
                energywarning = "(approximate)"

            print("{:<20} {:>20.12f} {}".format(
                "QM energy: ", self.QMenergy, energywarning))
            print("{:<20} {:>20.12f}".format("MM energy: ", self.MMenergy))
            print("{:<20} {:>20.12f} {}".format(
                "QM/MM energy: ", self.QM_MM_energy, energywarning))
            blankline()

        # FINAL QM/MM GRADIENT ASSEMBLY
        if Grad is True:
            # If OpenMM external force method then QM/MM gradient is already complete
            # NOTE: Not possible anymore
            if self.openmm_externalforce is True:
                pass
            #    self.QM_MM_gradient = self.MMgradient
            # Otherwise combine
            else:
                # Now assemble full QM/MM gradient
                # print("len(self.QM_PC_gradient):", len(self.QM_PC_gradient))
                # print("len(self.MMgradient):", len(self.MMgradient))
                assert len(self.QM_PC_gradient) == len(self.MMgradient)
                self.QM_MM_gradient = self.QM_PC_gradient + self.MMgradient

            if self.printlevel >= 3:
                print("Printlevel >=3: Printing all gradients to disk")
                # print("QM gradient (au/Bohr):")
                # module_coords.print_coords_all(self.QMgradient, self.qmelems, self.qmatoms)
                ash.modules.module_coords.write_coords_all(
                    self.QMgradient_wo_linkatoms,
                    self.qmelems,
                    indices=self.qmatoms,
                    file="QMgradient-without-linkatoms_{}".format(label),
                    description="QM gradient w/o linkatoms {} (au/Bohr):".format(label))

                # Writing QM+Linkatoms gradient
                ash.modules.module_coords.write_coords_all(self.QMgradient,
                                                           self.qmelems + ['L' for i in range(self.num_linkatoms)],
                                                           indices=self.qmatoms + [0 for i in range(self.num_linkatoms)],
                                                           file="QMgradient-with-linkatoms_{}".format(label),
                                                           description="QM gradient with linkatoms {} (au/Bohr):".format(label))

                # blankline()
                # print("PC gradient (au/Bohr):")
                # module_coords.print_coords_all(self.PCgradient, self.mmelems, self.mmatoms)
                ash.modules.module_coords.write_coords_all(
                    self.PCgradient,
                    self.mmelems,
                    indices=self.mmatoms,
                    file="PCgradient_{}".format(label),
                    description="PC gradient {} (au/Bohr):".format(label))
                # blankline()
                # print("QM+PC gradient (au/Bohr):")
                # module_coords.print_coords_all(self.QM_PC_gradient, self.elems, self.allatoms)
                ash.modules.module_coords.write_coords_all(
                    self.QM_PC_gradient,
                    self.elems,
                    indices=self.allatoms,
                    file="QM+PCgradient_{}".format(label),
                    description="QM+PC gradient {} (au/Bohr):".format(label))
                # blankline()
                # print("MM gradient (au/Bohr):")
                # module_coords.print_coords_all(self.MMgradient, self.elems, self.allatoms)
                ash.modules.module_coords.write_coords_all(
                    self.MMgradient,
                    self.elems,
                    indices=self.allatoms,
                    file="MMgradient_{}".format(label),
                    description="MM gradient {} (au/Bohr):".format(label))
                # blankline()
                # print("Total QM/MM gradient (au/Bohr):")
                # print("")
                # module_coords.print_coords_all(self.QM_MM_gradient, self.elems,self.allatoms)
                ash.modules.module_coords.write_coords_all(
                    self.QM_MM_gradient,
                    self.elems,
                    indices=self.allatoms,
                    file="QM_MMgradient_{}".format(label),
                    description="QM/MM gradient {} (au/Bohr):".format(label))
            if self.printlevel >= 2:
                print(BC.WARNING, BC.BOLD,
                      "------------ENDING QM/MM MODULE-------------", BC.END)
            print_time_rel(module_init_time, modulename='QM/MM run',
                           moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_MM_energy, self.QM_MM_gradient
        else:
            print_time_rel(module_init_time, modulename='QM/MM run',
                           moduleindex=2, currprintlevel=self.printlevel, currthreshold=1)
            return self.QM_MM_energy


class XEDA_EB(XEDATheory):
    def __init__(self, numcores=1, printlevel=2, label='xeda',
                 scf_type=None, basis=None, basis_file=None, ecp=None, functional=None,
                 scf_maxiter=128, eda=False, ct=False, blw=False, eda_atm=None, eda_charge=None, eda_mult=None,
                 bc=None, bc_list=None):
        super().__init__(
            self,
            numcores=numcores,
            printlevel=printlevel,
            label=label,
            scf_type=scf_type,
            basis=basis,
            basis_file=basis_file,
            ecp=ecp,
            functional=functional,
            scf_maxiter=scf_maxiter,
            eda=eda,
            ct=ct,
            blw=blw,
            eda_atm=eda_atm,
            eda_charge=eda_charge,
            eda_mult=eda_mult,
            bc=bc)
        self.bc_list = bc_list

    def set_embedding_options(self, PC=False, MM_coords_l=None, MMcharges_l=None):
        if PC is True:
            import pyxm.builder as builder
            # QM/MM pointcharge embedding
            print("PC True. Adding pointcharges")
            MMcharges_nd = [np.array(l) if isinstance(l, (list, np.ndarray)) else l for l in MMcharges_l]
            MM_coords_nd = [np.array(l) if isinstance(l, (list, np.ndarray)) else l for l in MM_coords_l]
            chgs = []
            for mm_charges, mm_coords in zip(MMcharges_nd, MM_coords_nd):
                if mm_charges is not None and mm_coords is not None:
                    chgs.append(np.ravel(np.column_stack(
                    (mm_charges[:, np.newaxis], mm_coords * ash.constants.ang2bohr))))
                else:
                    chgs.append([])
            self.bc = builder.charge_info(self.mol, chgs)
            self.hf.load_ham(self.bc, 1.0, 0.0)

    def run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None, mm_elems=None,
            elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
            charge=None, mult=None):
        self.prepare_run(current_coords=current_coords, elems=elems, charge=charge, mult=mult,
                         current_MM_coords=current_MM_coords,
                         MMcharges=MMcharges, qm_elems=qm_elems, Grad=Grad, PC=PC,
                         numcores=numcores, pe=pe, potfile=potfile, restart=restart, label=label)
        # Actual run
        return self.actualrun(
            current_coords=current_coords,
            current_MM_coords=current_MM_coords,
            MMcharges=MMcharges,
            qm_elems=qm_elems,
            elems=elems,
            Grad=Grad,
            PC=PC,
            numcores=numcores,
            pe=pe,
            potfile=potfile,
            restart=restart,
            label=label,
            charge=charge,
            mult=mult)

    def prepare_run(self, current_coords=None, current_MM_coords=None, MMcharges=None, qm_elems=None,
                    elems=None, Grad=False, PC=False, numcores=None, pe=False, potfile=None, restart=False, label=None,
                    charge=None, mult=None):

        module_init_time = time.time()
        if self.printlevel > 0:
            print(BC.OKBLUE, BC.BOLD,
                  "------------PREPARING XEDA INTERFACE-------------", BC.END)
            print("Object-label:", self.label)
            print("Run-label:", label)

            import pyxm.mole as mole
            mole.xscf_world.set_thread_num(self.numcores)

            if self.printlevel > 1:
                print("Number of XEDA  threads is:", self.numcores)

                # Checking if charge and mult has been provided
        if self.eda is False and (charge is None or mult is None):
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
            PC=PC, MM_coords_l=current_MM_coords, MMcharges_l=MMcharges)

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


class QMMM_EDA:

    def __init__(self, config: QMMMConfig):
        self.config = config
        self.validate_config()
        self.initialize_fragment()
        self.qm_theory_full = XEDATheory(
            numcores=self.config.numcores,
            scf_type=self.config.scf_tpye,
            basis=self.config.basis,
            functional=self.config.functional
        )
        self.qmmm_theory1  = None
        self.qmmm_theory2  = None

    def validate_config(self):
        if self.config.charge != sum(self.config.eda_charge):
            raise ValueError(
                "Sum of charge of monomers do not equal to total charge.")
        if self.config.forcefield is None and self.config.xmlfiles is None:
            print("No forcefield or xmlfile provided.")
            ashexit()

    def initialize_fragment(self):
        if self.config.pdbfile is not None:
            self.full_fragment = Fragment(
                pdbfile=self.config.pdbfile, charge=self.config.charge, mult=self.config.mult)

    def split_fragment(self, pdbfile, split_indices: List[int]):
        pdb = app.PDBFile(pdbfile)
        topology = pdb.topology
        positions = pdb.positions
        if sum(split_indices) != topology.getNumAtoms():
            raise ValueError(
                "Sum of NumAtoms of monomers do not equal to total NumAtoms.")
        modeller1 = app.Modeller(topology, positions)
        modeller2 = app.Modeller(topology, positions)
        modeller1.delete([atom for atom in pdb.topology.atoms()
                         if atom.index >= split_indices[0]])
        modeller2.delete([atom for atom in pdb.topology.atoms()
                         if atom.index < split_indices[0]])

        with open('fragment1.pdb', 'w') as f:
            app.PDBFile.writeFile(modeller1.topology, modeller1.positions, f)

        with open('fragment2.pdb', 'w') as f:
            app.PDBFile.writeFile(modeller2.topology, modeller2.positions, f)
        self.pdbfile1 = 'fragment1.pdb'
        self.pdbfile2 = 'fragment2.pdb'
        self.fragment1 = Fragment(
            pdbfile=self.pdbfile1, charge=self.config.eda_charge[0], mult=self.config.eda_mult[0])
        self.fragment2 = Fragment(
            pdbfile=self.pdbfile2, charge=self.config.eda_charge[1], mult=self.config.eda_mult[1])

    def split_qmatms(self):
        split_point = self.config.eda_atm[0]
        self.qm_atoms1 = list(filter(lambda x: x < split_point, self.config.qm_atoms))
        self.qm_atoms2 = list(map(lambda x: x - split_point, filter(lambda x: x >= split_point, self.config.qm_atoms)))

    def _prepare_theories(self):
        self._prepare_qm_theories()
        self._prepare_mm_theories()
        self._prepare_qmmm_theories()

    def _prepare_qm_theories(self):
        self.qm_theory1 = copy.deepcopy(self.qm_theory_full)
        self.qm_theory2 = copy.deepcopy(self.qm_theory_full)

    def _prepare_mm_theories(self):
        self.split_fragment(self.config.pdbfile, self.config.eda_atm)
        self.split_qmatms()

        self.is_fragment1_mm = False
        self.is_fragment2_mm = False
        if len(self.qm_atoms1) == self.fragment1.numatoms:
            self.is_fragment1_mm = False
        if len(self.qm_atoms2) == self.fragment2.numatoms:
            self.is_fragment2_mm = False
        if self.config.forcefield is not None:
            self.mm_theory_full = OpenMMTheory(
                numcores=self.config.numcores, pdbfile=self.config.pdbfile, forcefield=self.config.forcefield,
                autoconstraints=None, rigidwater=False)
            if self.is_fragment1_mm:
                self.mm_theory1 = OpenMMTheory(
                    numcores=self.config.numcores, pdbfile=self.pdbfile1, forcefield=self.config.forcefield,
                    autoconstraints=None, rigidwater=False)
            else:
                self.mm_theory1 = None
            if self.is_fragment2_mm:
                self.mm_theory2 = OpenMMTheory(
                    numcores=self.config.numcores, pdbfile=self.pdbfile2, forcefield=self.config.forcefield,
                    autoconstraints=None, rigidwater=False)
            else:
                self.mm_theory2 = None
        elif self.config.xmlfiles is not None:
            self.mm_theory_full = OpenMMTheory(
                numcores=self.config.numcores, pdbfile=self.config.pdbfile, xmlfiles=self.config.xmlfiles,
                autoconstraints=None, rigidwater=False)
            if self.is_fragment1_mm:
                self.mm_theory1 = OpenMMTheory(
                    numcores=self.config.numcores, pdbfile=self.pdbfile1, xmlfiles=self.config.xmlfiles,
                    autoconstraints=None, rigidwater=False)
            else:
                self.mm_theory1 = None
            if self.is_fragment2_mm:
                self.mm_theory2 = OpenMMTheory(
                    numcores=self.config.numcores, pdbfile=self.pdbfile2, xmlfiles=self.config.xmlfiles,
                    autoconstraints=None, rigidwater=False)
            else:
                self.mm_theory2 = None

    def _prepare_qmmm_theories(self):
        self.qmmm_theory_full = QMMMTheory_EDA(
            fragment=self.full_fragment,
            qm_theory=self.qm_theory_full,
            mm_theory=self.mm_theory_full,
            qmatoms=self.config.qm_atoms,
            embedding='Elstat',
            do_qm=False)
        if self.is_fragment1_mm:
            self.qmmm_theory1 = QMMMTheory_EDA(
                fragment=self.fragment1,
                qm_theory=self.qm_theory1,
                mm_theory=self.mm_theory1,
                qmatoms=self.qm_atoms1,
                embedding='Elstat',
                do_qm=False)
        if self.is_fragment2_mm:
            self.qmmm_theory2 = QMMMTheory_EDA(
                fragment=self.fragment2,
                qm_theory=self.qm_theory2,
                mm_theory=self.mm_theory2,
                qmatoms=self.qm_atoms2,
                embedding='Elstat',
                do_qm=False)

    def run(self) -> None:
        self._prepare_theories()
        self._run_qmmm()

    def _run_qmmm(self):
        try:
            self._calculate_supermolecule_energy()
            self._calculate_monomer1_energy()
            self._calculate_monomer2_energy()
        except Exception as e:
            print(f"Error in run_qmmm: {e}")
            raise

    def _calculate_supermolecule_energy(self):
        self.energy_sup = self._calculate_singlepoint(
            theory=self.qmmm_theory_full,
            fragment=self.full_fragment,
            charge=self.config.charge,
            mult=self.config.mult
        )

    def _calculate_monomer1_energy(self):
        theory = self.qmmm_theory1 if self.is_fragment1_mm else self.qm_theory1
        self.energy_monomer1 = self._calculate_singlepoint(
            theory=theory,
            fragment=self.fragment1,
            charge=self.config.eda_charge[0],
            mult=self.config.eda_mult[0]
        )

    def _calculate_monomer2_energy(self):
        theory = self.qmmm_theory2 if self.is_fragment2_mm else self.qm_theory2
        self.energy_monomer2 = self._calculate_singlepoint(
            theory=theory,
            fragment=self.fragment2,
            charge=self.config.eda_charge[1],
            mult=self.config.eda_mult[1]
        )

    def _calculate_singlepoint(self, theory, fragment, charge, mult):
        return Singlepoint(
            theory=theory,
            fragment=fragment,
            charge=charge,
            mult=mult,
            Grad=False,
            printlevel=-1
        )

    def _cal_mm_vdw(self):

        self.mm_theory_full.zero_LJ_epsilons()
        self.mm_novdw_full = self._calculate_singlepoint(
            self.mm_theory_full, self.full_fragment, self.config.charge, self.config.mult)
        vdw_full = self.energy_sup.mm_energy - self.mm_novdw_full.energy

        if self.is_fragment1_mm:
            self.mm_theory1.zero_LJ_epsilons()
            self.mm_novdw1 = self._calculate_singlepoint(
                self.mm_theory1, self.fragment1, self.config.eda_charge[0], self.config.eda_mult[0])
            vdw1 = self.energy_monomer1.mm_energy - self.mm_novdw1.energy
        if self.is_fragment2_mm:
            self.mm_theory2.zero_LJ_epsilons()
            self.mm_novdw2 = self._calculate_singlepoint(
                self.mm_theory2, self.fragment2, self.config.eda_charge[1], self.config.eda_mult[1])
            vdw2 = self.energy_monomer2.mm_energy - self.mm_novdw2.energy

        if not (self.is_fragment1_mm or self.is_fragment2_mm):
            raise RuntimeError("No monomer found")
        self.vdw_interaction = vdw_full
        if self.is_fragment1_mm:
            self.vdw_interaction -= vdw1
        if self.is_fragment2_mm:
            self.vdw_interaction -= vdw2

    def _cal_mm_elec(self):
        if self.is_fragment1_mm and self.is_fragment2_mm:
            self.mm_theory_full.zero_charges()
            self.mm_theory1.zero_charges()
            self.mm_theory2.zero_charges()
            self.mm_without_nonbonded_full = self._calculate_singlepoint(
                self.mm_theory_full, self.full_fragment, self.config.charge, self.config.mult)
            self.mm_without_nonbonded1 = self._calculate_singlepoint(
                self.mm_theory1, self.fragment1, self.config.eda_charge[0], self.config.eda_mult[0])
            self.mm_without_nonbonded2 = self._calculate_singlepoint(
                self.mm_theory2, self.fragment2, self.config.eda_charge[1], self.config.eda_mult[1])
            self.mm_elec_interaction = (self.mm_novdw_full.energy - self.mm_without_nonbonded_full.energy) - (
                self.mm_novdw1.energy - self.mm_without_nonbonded1.energy) - (self.mm_novdw2.energy - self.mm_without_nonbonded2.energy)

    def bsse_eda_cal(self):
       # QMenergy = self.qm_theory.run(current_coords=used_qmcoords,
        # current_MM_coords=self.pointchargecoords, MMcharges=self.pointcharges, mm_elems=self.mm_elems_for_qmprogram,
        # qm_elems=self.current_qmelems, Grad=False, PC=self.PC, numcores=numcores, charge=charge, mult=mult）
        mono1_qm_elems = self.qmmm_theory1.current_qmelems if self.is_fragment1_mm else self.fragment1.elems
        mono2_qm_elems = self.qmmm_theory2.current_qmelems if self.is_fragment2_mm else self.fragment2.elems
        full_qm_elems = self.qmmm_theory_full.current_qmelems
       
        if (lm1 := len(mono1_qm_elems)) + (lm2 := len(mono2_qm_elems)) == len(full_qm_elems):
            raise RuntimeError("DM_EDA(EB): Monomer QM elements are not consistent with full QM elements!")
        final_qm_elems = mono1_qm_elems + mono2_qm_elems
        qm_eda_atm = [lm1, lm2]
        qm_eda_mult = [self.config.eda_mult[0], self.config.eda_mult[1]]

        mono1_qm_coords = self.qmmm_theory1.qm_coords if self.is_fragment1_mm else self.fragment1.coords
        mono2_qm_coords = self.qmmm_theory2.qm_coords if self.is_fragment2_mm else self.fragment2.coords
        final_qm_coords = np.concatenate([mono1_qm_coords, mono2_qm_coords], axis=0)
        
        eda_fragment = Fragment(coords=final_qm_coords, elems=final_qm_elems,
                                charge=self.config.charge, mult=self.config.mult)
        
        mono1_mm_charges = self.qmmm_theory1.pointcharges if self.is_fragment1_mm else None
        mono2_mm_charges = self.qmmm_theory2.pointcharges if self.is_fragment2_mm else None
        full_mm_charges = self.qmmm_theory_full.pointcharges
        
        len_mono1_mm_charges = len(mono1_mm_charges) if mono1_mm_charges is not None else 0
        len_mono2_mm_charges = len(mono2_mm_charges) if mono2_mm_charges is not None else 0
        if len_mono1_mm_charges + len_mono2_mm_charges == len(full_mm_charges):
            raise RuntimeError("DM_EDA(EB): Monomer MM charges are not consistent with full MM charges!")
        
        mm_charges_list = [full_mm_charges, mono1_mm_charges, mono2_mm_charges]
        
        mono1_mm_coords = self.qmmm_theory1.pointchargecoords if self.is_fragment1_mm else None 
        mono2_mm_coords = self.qmmm_theory2.pointchargecoords if self.is_fragment2_mm else None
        full_mm_coords = self.qmmm_theory_full.pointchargecoords
        
        mm_coords_list = [full_mm_coords, mono1_mm_coords, mono2_mm_coords]
        
        self.eda_obj = XEDA_EB(numcores=self.config.numcores, label='xeda_eb', scf_type=self.config.scf_tpye, 
                               basis=self.config.basis, functional=self.config.functional, eda=True,
                               eda_atm=qm_eda_atm, eda_mult=qm_eda_mult)
        
        energy_components = Energy_decomposition(fragment=eda_fragment, theory=self.eda_obj,dmeda_eb=True, 
                                                 MM_charges=mm_charges_list, MM_coords=mm_coords_list
                                             )
        d_matrices = self.eda_obj.hf.d_matrix
        self.eda_obj.bc.cal_energy([d_matrices[0], d_matrices[2], d_matrices[1]])
        qmmm_ele = sum(qmmm_ele)
        return energy_components, qmmm_ele

    def cal_qmmm_elec(self):

        raise NotImplementedError("Not impl")
