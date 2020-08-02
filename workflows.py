import numpy as np
import functions_coords
import os
import glob
import ash
import subprocess as sp
import shutil
import constants
import math
from functions_ORCA import grab_HF_and_corr_energies
from interface_geometric import *

#Various workflows and associated sub-functions
#Primarily thermochemistry

#Spin-orbit splittings:
#Currently only including neutral atoms. Data in cm-1 from : https://webhome.weizmann.ac.il/home/comartin/w1/so.txt
atom_spinorbitsplittings = {'H': 0.000, 'B': -10.17, 'C' : -29.58, 'N' : 0.00, 'O' : -77.97, 'F' : -134.70,
                      'Al' : -74.71, 'Si' : -149.68, 'P' : 0.00, 'S' : -195.77, 'Cl' : -294.12}

#Core electrons for elements in ORCA
atom_core_electrons = {'H': 0, 'He' : 0, 'Li' : 0, 'Be' : 0, 'B': 2, 'C' : 2, 'N' : 2, 'O' : 2, 'F' : 2, 'Ne' : 2,
                      'Na' : 2, 'Mg' : 2, 'Al' : 10, 'Si' : 10, 'P' : 10, 'S' : 10, 'Cl' : 10, 'Ar' : 10,
                       'K' : 10, 'Ca' : 10, 'Sc' : 10, 'Ti' : 10, 'V' : 10, 'Cr' : 10, 'Mn' : 10, 'Fe' : 10, 'Co' : 10,
                       'Ni' : 10, 'Cu' : 10, 'Zn' : 10, 'Ga' : 18, 'Ge' : 18, 'As' : 18, 'Se' : 18, 'Br' : 18, 'Kr' : 18,
                       'Rb' : 18, 'Sr' : 18, 'Y' : 28, 'Zr' : 28, 'Nb' : 28, 'Mo' : 28, 'Tc' : 28, 'Ru' : 28, 'Rh' : 28,
                       'Pd' : 28, 'Ag' : 28, 'Cd' : 28, 'In' : 36, 'Sn' : 36, 'Sb' : 36, 'Te' : 36, 'I' : 36, 'Xe' : 36,
                       'Cs' : 36, 'Ba' : 36, 'Lu' : 46, 'Hf' : 46, 'Ta' : 46, 'w' : 46, 'Re' : 46, 'Os' : 46, 'Ir' : 46,
                       'Pt' : 46, 'Au' : 46, 'Hg' : 46, 'Tl' : 68, 'Pb' : 68, 'Bi' : 68, 'Po' : 68, 'At' : 68, 'Rn' : 68}

def Extrapolation_W1_SCF_3point(E):
    """
    Extrapolation function for old-style 3-point SCF in W1 theory
    :param E: list of HF energies (floats)
    :return:
    Note: Reading list backwards
    """
    SCF_CBS = E[-1]-(E[-1]-E[-2])**2/(E[-1]-2*E[-2]+E[-3])
    return SCF_CBS


#https://www.cup.uni-muenchen.de/oc/zipse/teaching/computational-chemistry-2/topics/overview-of-weizmann-theories/weizmann-1-theory/
def Extrapolation_W1_SCF_2point(E):
    """
    Extrapolation function for new-style 2-point SCF in W1 theory
    :param E: list of HF energies (floats)
    :return:
    Note: Reading list backwards
    """
    print("This has not been tested. Proceed with caution")
    exit()
    SCF_CBS = E[-1]+(E[-1]-E[-2])/((4/3)**5 - 1)
    return SCF_CBS

def Extrapolation_W1F12_SCF_2point(E):
    """
    Extrapolation function for new-style 2-point SCF in W1-F12 theory
    :param E: list of HF energies (floats)
    :return:
    Note: Reading list backwards
    """
    SCF_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**5 - 1)
    return SCF_CBS



def Extrapolation_W1_CCSD(E):
    """
    Extrapolation function (A+B/l^3) for 2-point CCSD in W1 theory
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((4/3)**3.22 - 1)
    return CCSDcorr_CBS

def Extrapolation_W1F12_CCSD(E):
    """
    Extrapolation function (A+B/l^3) for 2-point CCSD in W1 theory
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**3.67 - 1)
    return CCSDcorr_CBS

def Extrapolation_W1F12_triples(E):
    """
    Extrapolation function (A+B/l^3) for 2-point triples in W1-F12 theory.
    Note: Uses regular CCSD(T) energies, not F12
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**3.22 - 1)
    return CCSDcorr_CBS



def Extrapolation_W2_CCSD(E):
    """
    Extrapolation function (A+B/l^3) for 2-point CCSD in W2 theory
    :param E: list of CCSD energies (floats)
    :return:
    Note: Reading list backwards
    """
    CCSDcorr_CBS = E[-1]+(E[-1]-E[-2])/((5/4)**3 - 1)
    return CCSDcorr_CBS

def Extrapolation_W1_triples(E):
    """
    Extrapolation function  for 2-point (T) in W1 theory
    :param E: list of triples correlation energies (floats)
    :return:
    Note: Reading list backwards
    """
    triples_CBS = E[-1]+(E[-1]-E[-2])/((3/2)**3.22 - 1)
    return triples_CBS

def Extrapolation_W2_triples(E):
    """
    Extrapolation function  for 2-point (T) in W2 theory
    :param E: list of triples correlation energies (floats)
    :return:
    Note: Reading list backwards
    """
    triples_CBS = E[-1]+(E[-1]-E[-2])/((4/3)**3 - 1)
    return triples_CBS

def Extrapolation_twopoint(scf_energies, corr_energies, cardinals, basis_family):
    """
    Extrapolation function for general 2-point extrapolations
    :param scf_energies: list of SCF energies
    :param corr_energies: list of correlation energies
    :param cardinals: list of basis-cardinal numbers
    :param basis_family: string (e.g. cc, def2, aug-cc)
    :return: extrapolated SCF energy and correlation energy
    """
    #Dictionary of extrapolation parameters. Key: Basisfamilyandcardinals Value: list: [alpha, beta]
    extrapolation_parameters_dict = { 'cc_23' : [4.42, 2.460], 'aug-cc_23' : [4.30, 2.510], 'cc_34' : [5.46, 3.050], 'aug-cc_34' : [5.790, 3.050],
    'def2_23' : [10.390,2.4], 'def2_34' : [7.880,2.970], 'pc_23' : [7.02, 2.01], 'pc_34': [9.78, 4.09]}

    #NOTE: pc-n family uses different numbering. pc-1 is DZ(cardinal 2), pc-2 is TZ(cardinal 3), pc-4 is QZ(cardinal 4).
    if basis_family=='cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='cc_23'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='cc_34'
    elif basis_family=='aug-cc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='aug-cc_23'
    elif basis_family=='def2' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='def2_23'
    elif basis_family=='def2' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='def2_34'
    elif basis_family=='pc' and all(x in cardinals for x in [2, 3]):
        extrap_dict_key='pc_23'
    elif basis_family=='pc' and all(x in cardinals for x in [3, 4]):
        extrap_dict_key='pc_34'

    #Print energies
    print("Basis family is:", basis_family)
    print("SCF energies are:", scf_energies[0], "and", scf_energies[1])
    print("Correlation energies are:", corr_energies[0], "and", corr_energies[1])

    print("Extrapolation parameters:")
    alpha=extrapolation_parameters_dict[extrap_dict_key][0]
    beta=extrapolation_parameters_dict[extrap_dict_key][1]
    print("alpha :",alpha)
    print("beta :", beta)
    eX=math.exp(-1*alpha*math.sqrt(cardinals[0]))
    eY=math.exp(-1*alpha*math.sqrt(cardinals[1]))
    SCFextrap=(scf_energies[0]*eY-scf_energies[1]*eX)/(eY-eX)
    corrextrap=(math.pow(cardinals[0],beta)*corr_energies[0] - math.pow(cardinals[1],beta) * corr_energies[1])/(math.pow(cardinals[0],beta)-math.pow(cardinals[1],beta))

    print("SCF Extrapolated value is", SCFextrap)
    print("Correlation Extrapolated value is", corrextrap)
    print("Total Extrapolated value is", SCFextrap+corrextrap)

    return SCFextrap, corrextrap

def num_core_electrons(fragment):
    sum=0
    formula_list = functions_coords.molformulatolist(fragment.formula)
    for i in formula_list:
        els = atom_core_electrons[i]
        sum+=els
    return sum

#Note: Inner-shell correlation information: https://webhome.weizmann.ac.il/home/comartin/preprints/w1/node6.html
# Idea: Instead of CCSD(T), try out CEPA or pCCSD as alternative method. Hopefully as accurate as CCSD(T).
# Or DLPNO-CCSD(T) with LoosePNO ?

def W1theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1, memory=5000, HFreference=None):
    """
    Single-point W1 theory workflow.
    Differences: Basis sets may not be the same if 2nd-row element. TO BE CHECKED
    Scalar-relativistic step done via DKH. Same as modern W1 implementation.
    HF reference is important. UHF is not recommended. QRO gives very similar results to ROHF. To be set as default?
    Based on :
    https://webhome.weizmann.ac.il/home/comartin/w1/example.txt
    https://www.cup.uni-muenchen.de/oc/zipse/teaching/computational-chemistry-2/topics/overview-of-weizmann-theories/weizmann-1-theory/

    :param fragment: ASH fragment object
    :param charge: integer
    :param orcadir: string (path to ORCA)
    :param mult: integer (spin multiplicity)
    :param stabilityanalysis: boolean (currently not active)
    :param numcores: integer
    :param memory: integer (in MB)
    :param HFreference: string (UHF, QRO, ROHF)
    :return:
    """
    print("-----------------------------")
    print("W1theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("HFreference : ", HFreference)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)
    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #TODO: Add Stability analysis option  here later
    blocks="""
    %maxcore {}
    %scf
    maxiter 200
    end
    """.format(memory)

    #HF reference to use
    #If UHF then UHF will be enforced, also for closed-shell. unncessarily expensive
    if HFreference == 'UHF':
        print("HF reference = UHF chosen. Will enforce UHF (also for closed-shell)")
        hfkeyword='UHF'
    #ROHF option in ORCA requires dual-job. to be finished.
    elif HFreference == 'ROHF':
        print("ROHF reference is not yet available")
        exit()
    #QRO option is similar to ROHF. Recommended. Same as used by ORCA in DLPNO-CC.
    elif HFreference == 'QRO':
        print("HF reference = QRO chosen. Will use QROs for open-shell species)")
        hfkeyword='UNO'
    else:
        print("No HF reference chosen. Will use RHF for closed-shell and UHF for open-shell")
        hfkeyword=''

    ############################################################
    #Frozen-core calcs
    ############################################################
    #Special basis for H.
    # TODO: Add special basis for 2nd row block: Al-Ar
    # Or does ORCA W1-DZ choose this?
    ccsdt_dz_line="! CCSD(T) W1-DZ tightscf " + hfkeyword

    ccsdt_tz_line="! CCSD(T) W1-TZ tightscf " + hfkeyword

    ccsd_qz_line="! CCSD W1-QZ tightscf " + hfkeyword

    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsd_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out')
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out')
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_qz)
    CCSD_QZ_dict = grab_HF_and_corr_energies('orca-input.out')
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSD_QZ' + '.out')
    print("CCSD_QZ_dict:", CCSD_QZ_dict)

    #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDT_DZ_dict['HF'], CCSDT_TZ_dict['HF'], CCSD_QZ_dict['HF']]
    ccsdcorr_energies = [CCSDT_DZ_dict['CCSD_corr'], CCSDT_TZ_dict['CCSD_corr'], CCSD_QZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #Choice for SCF: old 3-point formula or new 2-point formula. Need to check which is recommended nowadays
    E_SCF_CBS = Extrapolation_W1_SCF_3point(scf_energies) #3-point extrapolation
    #E_SCF_CBS = Extrapolation_W1_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W1_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W1_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! CCSD(T) DKH W1-mtsmall  tightscf nofrozencore " + hfkeyword
    ccsdt_mtsmall_FC_line="! CCSD(T) W1-mtsmall tightscf " + hfkeyword

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W1_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W1 energy :", W1_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W1_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W1_total, E_dict


def W1F12theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1, memory=5000, HFreference='QRO'):
    """
    Single-point W1-F12 theory workflow.
    Differences: TBD
    Based on :
    https://webhome.weizmann.ac.il/home/comartin/OAreprints/240.pdf

    Differences: Core-valence and Rel done togeth using MTSmall as in W1 at the moment. TO be changed?
    No DBOC term
    

    :param fragment: ASH fragment object
    :param charge: integer
    :param orcadir: string (path to ORCA)
    :param mult: integer (spin multiplicity)
    :param stabilityanalysis: boolean (currently not active)
    :param numcores: integer
    :param memory: integer (in MB)
    :param HFreference: string (UHF, QRO, ROHF)
    :return:
    """
    print("-----------------------------")
    print("W1-F12 theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("HFreference : ", HFreference)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)
    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #TODO: Add Stability analysis option  here later
    blocks="""
    %maxcore {}
    %scf
    maxiter 200
    end
    """.format(memory)

    #HF reference to use
    #If UHF then UHF will be enforced, also for closed-shell. unncessarily expensive
    if HFreference == 'UHF':
        print("HF reference = UHF chosen. Will enforce UHF (also for closed-shell)")
        hfkeyword='UHF'
    #ROHF option in ORCA requires dual-job. to be finished.
    elif HFreference == 'ROHF':
        print("ROHF reference is not yet available")
        exit()
    #QRO option is similar to ROHF. Recommended. Same as used by ORCA in DLPNO-CC.
    elif HFreference == 'QRO':
        print("HF reference = QRO chosen. Will use QROs for open-shell species)")
        hfkeyword='UNO'
    else:
        print("No HF reference chosen. Will use RHF for closed-shell and UHF for open-shell")
        hfkeyword=''

    ############################################################
    #Frozen-core calcs
    ############################################################
    #Special basis for H.

    #F12-calculations for SCF and CCSD
    ccsdf12_dz_line="! CCSD-F12 cc-pVDZ-F12 cc-pVDZ-F12-CABS tightscf " + hfkeyword
    ccsdf12_tz_line="! CCSD-F12 cc-pVTZ-F12 cc-pVTZ-F12-CABS tightscf " + hfkeyword

    #Regular triples
    ccsdt_dz_line="! CCSD(T) W1-DZ tightscf " + hfkeyword
    ccsdt_tz_line="! CCSD(T) W1-TZ tightscf " + hfkeyword

    #F12
    ccsdf12_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdf12_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdf12_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    #Regular
    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)    
    
    
    ash.Singlepoint(fragment=fragment, theory=ccsdf12_dz)
    CCSDF12_DZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_DZ' + '.out')
    print("CCSDF12_DZ_dict:", CCSDF12_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdf12_tz)
    CCSDF12_TZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDF12_TZ' + '.out')
    print("CCSDF12_TZ_dict:", CCSDF12_TZ_dict)

    #Regular CCSD(T)
    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=False)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', F12=False)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)


    #List of all SCF energies (F12DZ,F12TZ), all CCSD-corr energies (F12DZ,F12TZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDF12_DZ_dict['HF'], CCSDF12_TZ_dict['HF']]
    ccsdcorr_energies = [CCSDF12_DZ_dict['CCSD_corr'], CCSDF12_TZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #2-point SCF extrapolation of modified HF energies
    E_SCF_CBS = Extrapolation_W1F12_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W1F12_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W1F12_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! CCSD(T) DKH W1-mtsmall  tightscf nofrozencore " + hfkeyword
    ccsdt_mtsmall_FC_line="! CCSD(T) W1-mtsmall tightscf " + hfkeyword

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W1F12_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W1-F12 energy :", W1F12_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W1F12_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W1F12_total, E_dict












#DLPNO-test
def DLPNO_W1theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF'):
    """
    WORK IN PROGRESS
    DLPNO-version of single-point W1 theory workflow.
    Differences: DLPNO-CC enforces QRO reference (similar to ROHF). No other reference possible.

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off . Not currently active
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    print("-----------------------------")
    print("DLPNO_W1theory_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("T1 : ", T1)
    print("SCF setting: ", scfsetting)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W1_total = -0.500000
        print("Using hardcoded value: ", W1_total)
        E_dict = {'Total_E': W1_total, 'E_SCF_CBS': W1_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #TODO: Add Stability analysis option  here later
    blocks="""
    %maxcore {}
    %scf
    maxiter 200
    end
    %mdci
    UseFullLMP2Guess false
    end

    """.format(memory)

    #Whether to use diffuse basis set or not
    #Note: this may fuck up basis set extrapolation
    #if noaug is True:
    #    prefix=''
    #else:
    #    prefix='aug-'

    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)'


    ############################################################s
    #Frozen-core calcs
    ############################################################
    #ccsdt_dz_line="! DLPNO-CCSD(T) {}cc-pVDZ {} tightscf ".format(prefix,auxbasis)
    #ccsdt_tz_line="! DLPNO-CCSD(T) {}cc-pVTZ {} tightscf ".format(prefix,auxbasis)
    #ccsd_qz_line="! DLPNO-CCSD {}cc-pVQZ {} tightscf ".format(prefix,auxbasis)
    ccsdt_dz_line="! {} W1-DZ {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)
    ccsdt_tz_line="! {} W1-TZ {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)
    ccsd_qz_line="! DLPNO-CCSD     W1-QZ {} {} {}".format(auxbasis, pnosetting, scfsetting)


    ccsdt_dz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_dz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsd_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_dz)
    CCSDT_DZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_DZ' + '.out')
    print("CCSDT_DZ_dict:", CCSDT_DZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_qz)
    CCSD_QZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSD_QZ' + '.out')
    print("CCSD_QZ_dict:", CCSD_QZ_dict)

    #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDT_DZ_dict['HF'], CCSDT_TZ_dict['HF'], CCSD_QZ_dict['HF']]
    ccsdcorr_energies = [CCSDT_DZ_dict['CCSD_corr'], CCSDT_TZ_dict['CCSD_corr'], CCSD_QZ_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_DZ_dict['CCSD(T)_corr'], CCSDT_TZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #Choice for SCF: old 3-point formula or new 2-point formula. Need to check which is recommended nowadays
    E_SCF_CBS = Extrapolation_W1_SCF_3point(scf_energies) #3-point extrapolation
    #E_SCF_CBS = Extrapolation_W1_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W1_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W1_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! {} DKH W1-mtsmall  {} {} nofrozencore {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)
    ccsdt_mtsmall_FC_line="! {} W1-mtsmall {}  {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W1_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W1 energy :", W1_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W1_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W1_total, E_dict

#DLPNO-F12
#Test: DLPNO-CCSD(T)-F12 protocol including CV+SR
def DLPNO_F12_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF', F12level='DZ'):
    """
    WORK IN PROGRESS
    DLPNO-CCSD(T)-F12 version of single-point W1-ish workflow.

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off . Not currently active
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    print("-----------------------------")
    print("DLPNO_F12_SP PROTOCOL")
    print("-----------------------------")
    print("Settings:")
    print("Number of cores: ", numcores)
    print("Maxcore setting: ", memory, "MB")
    print("")
    print("PNO setting: ", pnosetting)
    print("T1 : ", T1)
    print("SCF setting: ", scfsetting)
    print("F12 basis level : ", F12level)
    print("")
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        E_total = -0.500000
        print("Using hardcoded value: ", E_total)
        E_dict = {'Total_E': E_total, 'E_SCF_CBS': E_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return E_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #TODO: Add Stability analysis option  here later
    blocks="""
    %maxcore {}
    %scf
    maxiter 200
    end
    %mdci
    UseFullLMP2Guess false
    end

    """.format(memory)

    #Whether to use diffuse basis set or not
    #Note: this may fuck up basis set extrapolation
    #if noaug is True:
    #    prefix=''
    #else:
    #    prefix='aug-'

    #Auxiliary basis set. One big one
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        
        print("test...")
        exit()
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)-F12'


    ############################################################s
    #Frozen-core F12 calcs
    ############################################################

    ccsdt_f12_line="! {} cc-pV{}-F12 cc-pV{}-F12-CABS {} {} {}".format(ccsdtkeyword, F12level, F12level,auxbasis, pnosetting, scfsetting)


    ccsdt_f12 = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_f12_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_f12)
    CCSDT_F12_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True,F12=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_F12' + '.out')
    print("CCSDT_F12_dict:", CCSDT_F12_dict)

    #List of all SCF energies (DZ,TZ,QZ), all CCSD-corr energies (DZ,TZ,QZ) and all (T) corr energies (DZ,TZ)
    scf_energies = [CCSDT_F12_dict['HF']]
    ccsdcorr_energies = [CCSDT_F12_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_F12_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Final F12 energis
    E_SCF_CBS=scf_energies[0]
    E_CCSDcorr_CBS=ccsdcorr_energies[0]
    E_triplescorr_CBS=triplescorr_energies[0]

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    # Done regularly, not F12
    ############################################################
    print("Doing CV+SR at normal non-F12 level for now")
    ccsdt_mtsmall_NoFC_line="! DLPNO-CCSD(T) DKH W1-mtsmall  {} {} nofrozencore {}".format(auxbasis, pnosetting, scfsetting)
    ccsdt_mtsmall_FC_line="! DLPNO-CCSD(T) W1-mtsmall {}  {} {}".format(auxbasis, pnosetting, scfsetting)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    E_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final DLPNO-CCSD(T)-F12 energy :", E_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : E_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return E_total, E_dict


def DLPNO_W2theory_SP(fragment=None, charge=None, orcadir=None, mult=None, stabilityanalysis=False, numcores=1,
                      memory=5000, pnosetting='NormalPNO', T1=False, scfsetting='TightSCF'):
    """
    WORK IN PROGRESS
    DLPNO-version of single-point W2 theory workflow.
    Differences: DLPNO-CC enforces QRO reference (similar to ROHF). No other reference possible.

    :param fragment: ASH fragment
    :param charge: Charge of fragment (to be replaced)?
    :param orcadir: ORCA directory
    :param mult: Multiplicity of fragment (to be replaced)?
    :param stabilityanalysis: stability analysis on or off . Not currently active
    :param numcores: number of cores
    :param memory: Memory in MB
    :param scfsetting: ORCA keyword (e.g. NormalSCF, TightSCF, VeryTightSCF)
    :param pnosetting: ORCA keyword: NormalPNO, LoosePNO, TightPNO
    ;param T1: Boolean (whether to do expensive iterative triples or not)
    :return: energy and dictionary with energy-components
    """
    print("-----------------------------")
    print("DLPNO_W2theory_SP PROTOCOL")
    print("-----------------------------")
    print("Not active yet")
    exit()
    calc_label = "Frag" + str(fragment.formula) + "_" + str(fragment.charge) + "_"
    print("Calculation label: ", calc_label)

    numelectrons = int(fragment.nuccharge - charge)

    #if 1-electron species like Hydrogen atom then we either need to code special HF-based procedure or just hardcode values
    #Currently hardcoding H-atom case. Replace with proper extrapolated value later.
    if numelectrons == 1:
        print("Number of electrons is 1")
        print("Assuming hydrogen atom and skipping calculation")
        W2_total = -0.500000
        print("Using hardcoded value: ", W2_total)
        E_dict = {'Total_E': W2_total, 'E_SCF_CBS': W2_total, 'E_CCSDcorr_CBS': 0.0,
                  'E_triplescorr_CBS': 0.0, 'E_corecorr_and_SR': 0.0, 'E_SO': 0.0}
        return W1_total, E_dict

    #Reducing numcores if fewer active electron pairs than numcores.
    core_electrons = num_core_electrons(fragment)
    print("core_electrons:", core_electrons)
    valence_electrons = (numelectrons - core_electrons)
    electronpairs = int(valence_electrons / 2)
    if electronpairs  < numcores:
        print("Number of electrons in fragment:", numelectrons)
        print("Number of valence electrons :", valence_electrons )
        print("Number of valence electron pairs :", electronpairs )
        print("Setting numcores to number of electron pairs")
        numcores=int(electronpairs)

    #Block input for SCF/MDCI block options.
    #Disabling FullLMP2 guess in general as not available for open-shell
    #TODO: Add Stability analysis option  here later
    blocks="""
    %maxcore {}
    %scf
    maxiter 200
    end
    %mdci
    UseFullLMP2Guess false
    end

    """.format(memory)

    #Whether to use diffuse basis set or not
    #Note: this may fuck up basis set extrapolation
    #if noaug is True:
    #    prefix=''
    #else:
    #    prefix='aug-'

    #Auxiliary basis set. One big one
    #Todo: check whether it should be bigger
    auxbasis='cc-pV5Z/C'

    #Whether to use iterative triples or not. Default: regular DLPNO-CCSD(T)
    if T1 is True:
        ccsdtkeyword='DLPNO-CCSD(T1)'
    else:
        ccsdtkeyword='DLPNO-CCSD(T)'


    ############################################################s
    #Frozen-core calcs
    ############################################################
    #ccsdt_dz_line="! DLPNO-CCSD(T) {}cc-pVDZ {} tightscf ".format(prefix,auxbasis)
    #ccsdt_tz_line="! DLPNO-CCSD(T) {}cc-pVTZ {} tightscf ".format(prefix,auxbasis)
    #ccsd_qz_line="! DLPNO-CCSD {}cc-pVQZ {} tightscf ".format(prefix,auxbasis)
    ccsdt_tz_line="! {} W1-TZ {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)
    ccsdt_qz_line="! {} W1-QZ {} {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)
    ccsd_5z_line="! DLPNO-CCSD  haV5Z(+d) {} {} {}".format(auxbasis, pnosetting, scfsetting)

    print("Need to check better if correct basis set.")

    #Defining W2 5Z basis
    #quintblocks = blocks + """%basis newgto H "cc-pV5Z" end
    #"""

    ccsdt_tz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_tz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_qz = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_qz_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsd_5z = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsd_5z_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_tz)
    CCSDT_TZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_TZ' + '.out')
    print("CCSDT_TZ_dict:", CCSDT_TZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsdt_qz)
    CCSDT_QZ_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_QZ' + '.out')
    print("CCSDT_QZ_dict:", CCSDT_QZ_dict)

    ash.Singlepoint(fragment=fragment, theory=ccsd_5z)
    CCSD_5Z_dict = grab_HF_and_corr_energies('orca-input.out', DLPNO=True)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSD_5Z' + '.out')
    print("CCSD_5Z_dict:", CCSD_5Z_dict)

    #List of all SCF energies (TZ,QZ,5Z), all CCSD-corr energies (TZ,QZ,5Z) and all (T) corr energies (TZ,qZ)
    scf_energies = [CCSDT_TZ_dict['HF'], CCSDT_QZ_dict['HF'], CCSD_5Z_dict['HF']]
    ccsdcorr_energies = [CCSDT_TZ_dict['CCSD_corr'], CCSDT_QZ_dict['CCSD_corr'], CCSD_5Z_dict['CCSD_corr']]
    triplescorr_energies = [CCSDT_TZ_dict['CCSD(T)_corr'], CCSDT_QZ_dict['CCSD(T)_corr']]

    print("")
    print("scf_energies :", scf_energies)
    print("ccsdcorr_energies :", ccsdcorr_energies)
    print("triplescorr_energies :", triplescorr_energies)

    #Extrapolations
    #Choice for SCF: old 3-point formula or new 2-point formula. Need to check which is recommended nowadays
    E_SCF_CBS = Extrapolation_W1_SCF_3point(scf_energies) #3-point extrapolation
    #E_SCF_CBS = Extrapolation_W1_SCF_2point(scf_energies) #2-point extrapolation

    E_CCSDcorr_CBS = Extrapolation_W2_CCSD(ccsdcorr_energies) #2-point extrapolation
    E_triplescorr_CBS = Extrapolation_W2_triples(triplescorr_energies) #2-point extrapolation

    print("E_SCF_CBS:", E_SCF_CBS)
    print("E_CCSDcorr_CBS:", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS:", E_triplescorr_CBS)

    ############################################################
    #Core-correlation + scalar relativistic as joint correction
    ############################################################
    ccsdt_mtsmall_NoFC_line="! {} DKH W1-mtsmall  {} {} nofrozencore {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)
    ccsdt_mtsmall_FC_line="! {} W1-mtsmall {}  {} {}".format(ccsdtkeyword, auxbasis, pnosetting, scfsetting)

    ccsdt_mtsmall_NoFC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_NoFC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)
    ccsdt_mtsmall_FC = ash.ORCATheory(orcadir=orcadir, orcasimpleinput=ccsdt_mtsmall_FC_line, orcablocks=blocks, nprocs=numcores, charge=charge, mult=mult)

    energy_ccsdt_mtsmall_nofc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_NoFC)
    shutil.copyfile('orca-input.out', './'+ calc_label + 'CCSDT_MTsmall_NoFC_DKH' + '.out')
    energy_ccsdt_mtsmall_fc = ash.Singlepoint(fragment=fragment, theory=ccsdt_mtsmall_FC)
    shutil.copyfile('orca-input.out', './' + calc_label + 'CCSDT_MTsmall_FC_noDKH' + '.out')

    #Core-correlation is total energy difference between NoFC-DKH and FC-norel
    E_corecorr_and_SR = energy_ccsdt_mtsmall_nofc - energy_ccsdt_mtsmall_fc
    print("E_corecorr_and_SR:", E_corecorr_and_SR)

    ############################################################
    #Spin-orbit correction for atoms.
    ############################################################
    if fragment.numatoms == 1:
        print("Fragment is an atom. Looking up atomic spin-orbit splitting value")
        E_SO = atom_spinorbitsplittings[fragment.elems[0]] / constants.hartocm
    else :
        E_SO = 0.0

    print("Spin-orbit correction (E_SO):", E_SO)

    ############################################################
    #FINAL RESULT
    ############################################################
    print("")
    print("")
    W2_total = E_SCF_CBS + E_CCSDcorr_CBS + E_triplescorr_CBS +E_corecorr_and_SR  + E_SO
    print("Final W2 energy :", W2_total, "Eh")
    print("")
    print("Contributions:")
    print("--------------")
    print("E_SCF_CBS : ", E_SCF_CBS)
    print("E_CCSDcorr_CBS : ", E_CCSDcorr_CBS)
    print("E_triplescorr_CBS : ", E_triplescorr_CBS)
    print("E_corecorr_and_SR : ", E_corecorr_and_SR)
    print("E_SO : ", E_SO)

    E_dict = {'Total_E' : W2_total, 'E_SCF_CBS' : E_SCF_CBS, 'E_CCSDcorr_CBS' : E_CCSDcorr_CBS, 'E_triplescorr_CBS' : E_triplescorr_CBS,
             'E_corecorr_and_SR' : E_corecorr_and_SR, 'E_SO' : E_SO}


    #Cleanup GBW file. Full cleanup ??
    # TODO: Keep output files for each step
    os.remove('orca-input.gbw')

    #return final energy and also dictionary with energy components
    return W2_total, E_dict

#Thermochemistry protocol. Take list of fragments, stoichiometry, etc
#Requires orcadir, inputline for geo-opt. ORCA-bssed
#Make more general. Not sure. ORCA makes most sense for geo-opt and HL theory
def thermochemprotocol(SPprotocol=None, fraglist=None, stoichiometry=None, orcadir=None, numcores=None,
                       Opt_protocol_inputline=None, Opt_protocol_blocks=None, pnosetting='NormalPNO', F12level='DZ'):
    if Opt_protocol_blocks is None:
        Opt_protocol_blocks=""

    #DFT Opt+Freq  and Single-point High-level workflow
    FinalEnergies = []; list_of_dicts = []; ZPVE_Energies=[]
    for species in fraglist:
        #Only Opt+Freq for molecules, not atoms
        if species.numatoms != 1:
            #DFT-opt
            ORCAcalc = ash.ORCATheory(orcadir=orcadir, charge=species.charge, mult=species.mult,
                orcasimpleinput=Opt_protocol_inputline, orcablocks=Opt_protocol_blocks, nprocs=numcores)
            geomeTRICOptimizer(theory=ORCAcalc,fragment=species)
            #DFT-FREQ
            thermochem = ash.NumFreq(fragment=species, theory=ORCAcalc, npoint=2, runmode='serial')
            ZPVE = thermochem['ZPVE']
        else:
            #Setting ZPVE to 0.0.
            ZPVE=0.0
        #Single-point W1
        if SPprotocol == 'W1':
            FinalE, componentsdict = W1theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, HFreference='QRO')
        elif SPprotocol == 'DLPNO-W1':
            FinalE, componentsdict = DLPNO_W1theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting, T1=False)
        elif SPprotocol == 'DLPNO-F12':
            FinalE, componentsdict = DLPNO_F12_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting, T1=False, F12level=F12level)
        elif SPprotocol == 'W1-F12':
            FinalE, componentsdict = W1F12theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, HFreference='QRO')
        elif SPprotocol == 'DLPNO-W1-F12':
            FinalE, componentsdict = DLPNO_W1F12theory_SP(fragment=species, charge=species.charge,
                        mult=species.mult, orcadir=orcadir, numcores=numcores, memory=5000, pnosetting=pnosetting, T1=False,)
        else:
            print("Unknown Singlepoint protocol")
            exit()
        FinalEnergies.append(FinalE+ZPVE); list_of_dicts.append(componentsdict)
        ZPVE_Energies.append(ZPVE)


    #Reaction Energy via list of total energies:
    scf_parts=[dict['E_SCF_CBS'] for dict in list_of_dicts]
    ccsd_parts=[dict['E_CCSDcorr_CBS'] for dict in list_of_dicts]
    triples_parts=[dict['E_triplescorr_CBS'] for dict in list_of_dicts]
    CV_SR_parts=[dict['E_corecorr_and_SR'] for dict in list_of_dicts]
    SO_parts=[dict['E_SO'] for dict in list_of_dicts]

    #Reaction Energy of total energiese and also different contributions
    print("")
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=scf_parts, unit='kcalpermol', label='ΔSCF')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ccsd_parts, unit='kcalpermol', label='ΔCCSD')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=triples_parts, unit='kcalpermol', label='Δ(T)')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=CV_SR_parts, unit='kcalpermol', label='ΔCV+SR')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=SO_parts, unit='kcalpermol', label='ΔSO')
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=ZPVE_Energies, unit='kcalpermol', label='ΔZPVE')
    print("----------------------------------------------")
    ash.ReactionEnergy(stoichiometry=stoichiometry, list_of_fragments=fraglist, list_of_energies=FinalEnergies, unit='kcalpermol', label='Total ΔE')

    ash.print_time_rel(settings_ash.init_time,modulename='Entire thermochemprotocol')