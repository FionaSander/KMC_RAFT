"""
Functions used for RAFT polymerisation simulations
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
from pathlib import Path
import shutil
from biopandas.pdb import PandasPdb 
import math
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random


def initiate_particle_count(Species, n): 
    """
    Initiation of X arrays which contain the count of each species defined in the beginning and are updated with every MC step
    """
    S = np.array(Species)
    S_sum = np.sum(S)
    X = np.array([])
    Species_count = np.array([])
    x = 0
    t = 0

    for i in range(0, len(S)):
        x = round((S[i]/S_sum) * n ) 
        X = np.append(X, x)
        Species_count = np.append(Species_count, x)

    return Species_count, S_sum, X


def calculate_reaction_constants(n, N_A, S_sum, f, k_d, k_f ,k_d_exp, k_pi_exp, k_p_exp, k_a_exp, k_f_exp, k_tc_exp, k_td_exp, k_ct_exp):
    V = n/(N_A*S_sum)

    #calculate reaction constants
    k_pi = k_pi_exp/(V*N_A)
    k_p = k_p_exp/(V*N_A)
    k_a = k_a_exp/(V*N_A)
    k_tc = k_tc_exp/(V*N_A)
    k_td = k_td_exp/(V*N_A)
    k_ct = k_ct_exp/(V*N_A)

    k = np.array([(f*k_d), k_p, k_a, (k_f/2), (k_f/2), k_a, (k_f/2), (k_f/2), k_tc, k_td, k_ct, k_ct])

    return k

def calculate_reaction_constants_copoly(n, N_A, S_sum, f, k_d, k_pi_1_exp, k_pi_2_exp, k_p_1_exp, k_p_2_exp, k_a_1_exp, k_a_2_exp, k_f_1, k_f_2, k_tc_11_exp, k_tc_12_exp, k_tc_21_exp, k_tc_22_exp, k_td_11_exp, k_td_12_exp, k_td_21_exp, k_td_22_exp, k_ct_exp):
    V = n/(N_A*S_sum)

    #calculate reaction constants
    k_pi_1 = (k_pi_1_exp)/(V*N_A)
    k_pi_2 = (k_pi_2_exp)/(V*N_A)
    k_p_1 = (k_p_1_exp)/(V*N_A)
    k_p_2 = (k_p_2_exp)/(V*N_A)
    k_a_1 = (k_a_1_exp)/(V*N_A)
    k_a_2 = (k_a_2_exp)/(V*N_A)
    k_tc_11 = k_tc_11_exp/(V*N_A)
    k_tc_12 = k_tc_12_exp/(V*N_A)
    k_tc_21 = k_tc_21_exp/(V*N_A)
    k_tc_22 = k_tc_22_exp/(V*N_A)
    k_td_11 = k_td_11_exp/(V*N_A)
    k_td_12 = k_td_12_exp/(V*N_A)
    k_td_21 = k_td_21_exp/(V*N_A)
    k_td_22 = k_td_22_exp/(V*N_A)

    k_ct = k_ct_exp/(V*N_A)
    k = np.array([(f*k_d), k_pi_1, k_pi_2, k_p_1, k_p_2, k_a_1, k_a_2,(k_f_1/2), (k_f_2/2),(k_f_1/2), (k_f_2/2), k_a_1, k_a_2, (k_f_1/2), (k_f_2/2),(k_f_1/2), (k_f_2/2), k_tc_11, k_tc_12, k_tc_21, k_tc_22, k_td_11,k_td_12, k_td_21, k_td_22, k_ct, k_ct])

    return k

def calculate_reaction_constants_copoly_zapata(n, N_A, S_sum, f, k_d, k_pi_1_exp, k_pi_2_exp, k_p_1_exp, k_p_2_exp, k_a_1_exp, k_a_2_exp, k_f_1, k_f_2, k_tc_11_exp, k_tc_12_exp, k_tc_21_exp, k_tc_22_exp, k_td_11_exp, k_td_12_exp, k_td_21_exp, k_td_22_exp, k_ct_exp):
    V = n/(N_A*S_sum)

    #calculate reaction constants
    k_p_1 = (k_p_1_exp)/(V*N_A)
    k_p_2 = (k_p_2_exp)/(V*N_A)
    k_a_1 = (k_a_1_exp)/(V*N_A)
    k_a_2 = (k_a_2_exp)/(V*N_A)
    k_tc_11 = k_tc_11_exp/(V*N_A)
    k_tc_12 = k_tc_12_exp/(V*N_A)
    k_tc_21 = k_tc_21_exp/(V*N_A)
    k_tc_22 = k_tc_22_exp/(V*N_A)
    k_td_11 = k_td_11_exp/(V*N_A)
    k_td_12 = k_td_12_exp/(V*N_A)
    k_td_21 = k_td_21_exp/(V*N_A)
    k_td_22 = k_td_22_exp/(V*N_A)
    k_ct = k_ct_exp/(V*N_A)

    k = np.array([(f*k_d), k_p_1, k_p_2, k_a_1, k_a_2,(k_f_1/2), (k_f_2/2),(k_f_1/2), (k_f_2/2), k_a_1, k_a_2, (k_f_1/2), (k_f_2/2),(k_f_1/2), (k_f_2/2), k_tc_11, k_tc_12, k_tc_21, k_tc_22, k_td_11,k_td_12, k_td_21, k_td_22, k_ct, k_ct])

    return k

def create_polymer_dicts():
    Polymers = {}
    T_Polymers = {}
    Polymers_inactive = {}
    Polymers_adduct_n = {}
    Polymers_adduct_m = {}
    Polymers_dead = {}

    return Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead

def create_polymer_arrays():
    Polymers = np.array([])
    T_Polymers = np.array([])
    Polymers_inactive = np.array([])
    Polymers_adduct_n = np.array([])
    Polymers_adduct_m = np.array([])
    Polymers_dead = np.array([])

    return Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead


def calculate_reaction_rates(k, X):

    r_1 = k[0] * X[0] 
    r_2 = k[1] * X[3]*X[1]
    r_3 = k[2] * X[3]*X[2]
    r_4 = k[3] * X[7]
    r_5 = k[4] * X[7]
    r_6 = k[5] * X[3]*X[4]
    r_7 = k[6] * X[5]
    r_8 = k[7] * X[5]
    r_9 = k[8] * X[3]*X[3]
    r_10 = k[9] * X[3]*X[3]
    r_11 = k[10] * X[3]*X[5]
    r_12 = k[11] * X[3]*X[5]

    R = np.array([r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, r_9, r_10, r_11, r_12])     
    R_sum = np.sum(R)
    m = len(R) #number of reactions in the kinetic model

    return R, R_sum, m


def calculate_reaction_rates_copoly_modified(k, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead):

    r_1 = k[0] * X[0] #create new polymer
    r_2a = k[1] * X[4]*X[1] 
    r_2b = k[2] * X[4]*X[2]
    r_3a = k[3] * X[4]*X[1]
    r_3b = k[4] * X[4]*X[2]
    r_4a = k[5] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*X[3]
    r_4b = k[6] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*X[3]
    r_5a = k[7] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M1'])
    r_5b = k[8] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M2'])
    r_6a = k[9] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M1'])
    r_6b = k[10] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M2'])
    r_7a = k[11] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*X[5]
    r_7b = k[12] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*X[5]
    r_8a = k[13] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M1'])
    r_8b = k[14] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M2'])
    r_9a = k[15] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M1'])
    r_9b = k[16] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M2'])
    r_10a = k[17] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1']) 
    r_10b = k[18] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_10c = 0#k[19] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])
    r_10d = k[20] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_11a = k[21] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])
    r_11b = k[22] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_11c = 0#k[23] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])
    r_11d = k[24] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_12 = k[25] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*X[6]
    r_13 = k[26] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*X[6]

    R = np.array([r_1, r_2a, r_2b, r_3a, r_3b,r_4a, r_4b, r_5a, r_5b, r_6a, r_6b, r_7a, r_7b, r_8a, r_8b, r_9a, r_9b, r_10a, r_10b, r_10c, r_10d, r_11a, r_11b, r_11c, r_11d, r_12, r_13])     
    R_sum = np.sum(R)   
    m = len(R) 

    return R, R_sum, m

def calculate_reaction_rates_copoly_minimised_zapata(k, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead):

    r_1 = k[0] * X[0] #create new polymer
    r_3a = k[3] * X[4]*X[1]
    r_3b = k[4] * X[4]*X[2]
    r_4a = k[5] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*X[3]
    r_4b = k[6] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*X[3]
    r_5a = k[7] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M1'])
    r_5b = k[8] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M2'])
    r_6a = k[9] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M1'])
    r_6b = k[10] * len([polymer for polymer in T_Polymers.values() if list(polymer)[-1] == 'M2'])
    r_7a = k[11] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*X[5]
    r_7b = k[12] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*X[5]
    r_8a = k[13] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M1'])
    r_8b = k[14] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M2'])
    r_9a = k[15] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M1'])
    r_9b = k[16] * len([polymer for polymer in Polymers_adduct_n.values() if list(polymer)[-1] == 'M2'])
    r_10a = k[17] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1']) 
    r_10b = k[18] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_10c = 0#k[19] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])
    r_10d = k[20] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_11a = k[21] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])
    r_11b = k[22] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_11c = 0#k[23] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])
    r_11d = k[24] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])
    r_12 = k[25] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M1'])*X[6]
    r_13 = k[26] * len([polymer for polymer in Polymers.values() if list(polymer)[-1] == 'M2'])*X[6]
    R = np.array([r_1, r_3a, r_3b,r_4a, r_4b, r_5a, r_5b, r_6a, r_6b, r_7a, r_7b, r_8a, r_8b, r_9a, r_9b, r_10a, r_10b, r_10c, r_10d, r_11a, r_11b, r_11c, r_11d, r_12, r_13])     
    R_sum = np.sum(R)   
    m = len(R) 
    return R, R_sum, m

def calculate_probabilities(m, P, R, R_sum):
    p = 0
    for i in range(0, m):
        p = R[i]/R_sum
        P = np.append(P, p)
    return P

def define_reaction_indices(P):
    index_array = np.arange(0, len(P))
    index = np.random.choice(index_array,  p = P)
    return index

def calculate_timestep(R_sum, t):
    r4 = np.random.uniform(0, 1)
    Tau = (np.log(1/r4))/(R_sum)
    t = t + Tau
    return t

def calculate_timestep_speedy(R_sum, t):
    r4 = np.random.exponential(scale=1.0)
    Tau = r4/R_sum
    t = t + Tau
    return t


def save_values(X, Species_count, output_period, index, index_list, time, t, loops):
    """
    Save the counts for all species in a new array every number of MC loops which is defined by output_period (e.g. every 1000 MC loops). The timestep is saved for the output period.
    """

    if loops % output_period == 0 :
        Species_count_0 = np.append(Species_count[0], X[0]) 
        Species_count_1 = np.append(Species_count[1], X[1]) 
        Species_count_2 = np.append(Species_count[2], X[2]) 
        Species_count_3 = np.append(Species_count[3], X[3]) 
        Species_count_4 = np.append(Species_count[4], X[4])
        Species_count_5 = np.append(Species_count[5], X[5])
        Species_count_6 = np.append(Species_count[6], X[6])
        Species_count_7 = np.append(Species_count[7], X[7])

        index_list = np.append(index_list, index)
        time = np.append(time, t)
        Species_count = np.array([Species_count_0,Species_count_1,Species_count_2,Species_count_3,Species_count_4,Species_count_5,Species_count_6,Species_count_7])

    return Species_count, time

def save_values_copoly(X, Species_count, output_period, index, index_list, time, t, loops):
    """
    Save the counts for all species in a new array every number of MC loops which is defined by output_period (e.g. every 1000 MC loops). The timestep is saved for the output period.
    """

    if loops % output_period == 0 :
        Species_count_0 = np.append(Species_count[0], X[0]) 
        Species_count_1 = np.append(Species_count[1], X[1]) 
        Species_count_2 = np.append(Species_count[2], X[2]) 
        Species_count_3 = np.append(Species_count[3], X[3]) 
        Species_count_4 = np.append(Species_count[4], X[4])
        Species_count_5 = np.append(Species_count[5], X[5])
        Species_count_6 = np.append(Species_count[6], X[6])
        Species_count_7 = np.append(Species_count[7], X[7])
        Species_count_8 = np.append(Species_count[8], X[8])

        index_list = np.append(index_list, index)
        time = np.append(time, t)
        Species_count = np.array([Species_count_0,Species_count_1,Species_count_2,Species_count_3,Species_count_4,Species_count_5,Species_count_6,Species_count_7,Species_count_8])

    return Species_count

def save_values_co3poly(X, Species_count, output_period, index, index_list, time, t, loops):
    """
    Save the counts for all species in a new array every number of MC loops which is defined by output_period (e.g. every 1000 MC loops). The timestep is saved for the output period.
    """

    if loops % output_period == 0 :
        Species_count_0 = np.append(Species_count[0], X[0])
        Species_count_1 = np.append(Species_count[1], X[1]) 
        Species_count_2 = np.append(Species_count[2], X[2]) 
        Species_count_3 = np.append(Species_count[3], X[3]) 
        Species_count_4 = np.append(Species_count[4], X[4])
        Species_count_5 = np.append(Species_count[5], X[5])
        Species_count_6 = np.append(Species_count[6], X[6])
        Species_count_7 = np.append(Species_count[7], X[7])
        Species_count_8 = np.append(Species_count[8], X[8])
        Species_count_9 = np.append(Species_count[9], X[9])

        index_list = np.append(index_list, index)
        time = np.append(time, t)
        Species_count = np.array([Species_count_0,Species_count_1,Species_count_2,Species_count_3,Species_count_4,Species_count_5,Species_count_6,Species_count_7,Species_count_8,Species_count_9])


    return Species_count

def save_values_multipoly(M_number, X, Species_count, output_period, index, index_list, time, t, loops):
    """
    Save the counts for all species in a new array every number of MC loops which is defined by output_period (e.g. every 1000 MC loops). The timestep is saved for the output period.
    """

    if loops % output_period == 0 :
        species_number = 7 + M_number
        for species in range(species_number):
            Species_count_species = np.append(Species_count[species], X[species])
            Species_count = np.append(Species_count, Species_count_species)

        index_list = np.append(index_list, index)
        time = np.append(time, t)


    return Species_count


def calculate_molecular_weights(Species_count, theory, poly_type, Polymers, Polymers_inactive, Polymers_adduct, Polymers_dead, TR_M_act, co_number, initial_M_m ):
    """
    MW calcualtion fro all species depending on the theory and poly_type used for the simulation. 
    """

    if (theory == 'SF' and poly_type == 'homo') or (theory == 'IRTO' and poly_type == 'homo'):
        I_m = np.array(Species_count[0]) 
        M_m = np.array(Species_count[1])
        RAFT_m = np.array(Species_count[2])
        R_M_act_m = np.array(Polymers) * 1 + (TR_M_act * 1) 
        R_M_inact_m = np.array(Polymers_inactive) 
        R_m_n_m = np.array(Polymers_adduct) + 1
        D_m = np.array(Polymers_dead) 

        I_m_sum = I_m[-1]
        M_m_sum = M_m[-1]
        RAFT_m_sum = RAFT_m[-1]
        R_M_act_m_sum = np.sum(R_M_act_m)
        R_M_inact_m_sum = np.sum(R_M_inact_m)  
        R_m_n_m_sum = np.sum(R_m_n_m) 
        D_m_sum = np.sum(D_m) 

    elif (theory == 'SF' and poly_type == 'co') or (theory == 'IRTO' and poly_type == 'co'):
        I_m = np.array(Species_count[0])
        RAFT_m = np.array(Species_count[(co_number+1)])

        R_M_act_m = np.array([])
        R_M_inact_m = np.array([])
        R_m_n_m = np.array([])
        D_m = np.array([])

        M_1 = 0
        M_2 = 0
        for polymer in Polymers:
            M_1 += Polymers[polymer].count('M1')
            M_2 += Polymers[polymer].count('M2')
        R_M_act_m = np.append(R_M_act_m, ((M_1*1) + (M_2*1)))

        M_1 = 0
        M_2 = 0
        for polymer in Polymers_inactive:
            M_1 += Polymers_inactive[polymer].count('M1')
            M_2 += Polymers_inactive[polymer].count('M2')
        R_M_inact_m = np.append(R_M_inact_m, ((M_1*1) + (M_2*1)))

        M_1 = 0
        M_2 = 0
        for polymer in Polymers_adduct.keys():
            M_1 += Polymers_adduct[polymer].count('M1')
            M_2 += Polymers_adduct[polymer].count('M2')
        R_m_n_m = np.append(R_m_n_m, ((M_1*1) + (M_2*1)))

        M_1 = 0
        M_2 = 0
        for polymer in Polymers_dead.keys():
            M_1 += Polymers_dead[polymer].count('M1')
            M_2 += Polymers_dead[polymer].count('M2')
        D_m = np.append(D_m, ((M_1*1) + (M_2*1)))
        
        I_m_sum = I_m[-1]
        RAFT_m_sum = RAFT_m[-1]
        R_M_act_m_sum = np.sum(R_M_act_m)
        R_M_inact_m_sum = np.sum(R_M_inact_m)  
        R_m_n_m_sum = np.sum(R_m_n_m) 
        D_m_sum = np.sum(D_m)   \

        M_m = np.array([])
        M_m_sum = np.array([])
        for i in range(co_number-1):

            M_m = np.append(M_m, np.array(Species_count[i+1]))

    elif (theory == 'SF' and poly_type == 'co_3') or (theory == 'IRTO' and poly_type == 'co_3'):
        I_m = np.array(Species_count[0])
        RAFT_m = np.array(Species_count[(co_number+1)])

        R_M_act_m = np.array([])
        R_M_inact_m = np.array([])
        R_m_n_m = np.array([])
        D_m = np.array([])

        M_1 = 0
        M_2 = 0
        M_3 = 0
        for polymer in Polymers:
            M_1 += Polymers[polymer].count('M1')
            M_2 += Polymers[polymer].count('M2')
            M_3 += Polymers[polymer].count('M3')
        R_M_act_m = np.append(R_M_act_m, ((M_1*1) + (M_2*1)+ (M_3*1)))

        M_1 = 0
        M_2 = 0
        M_3 = 0
        for polymer in Polymers_inactive:
            M_1 += Polymers_inactive[polymer].count('M1')
            M_2 += Polymers_inactive[polymer].count('M2')
            M_3 += Polymers_inactive[polymer].count('M3')
        R_M_inact_m = np.append(R_M_inact_m, ((M_1*1) + (M_2*1)+ (M_3*1)))

        M_1 = 0
        M_2 = 0
        M_3 = 0
        for polymer in Polymers_adduct.keys():
            M_1 += Polymers_adduct[polymer].count('M1')
            M_2 += Polymers_adduct[polymer].count('M2')
            M_3 += Polymers_adduct[polymer].count('M3')
        R_m_n_m = np.append(R_m_n_m, ((M_1*1) + (M_2*1)+ (M_3*1)))

        M_1 = 0
        M_2 = 0
        M_3 = 0
        for polymer in Polymers_dead.keys():
            M_1 += Polymers_dead[polymer].count('M1')
            M_2 += Polymers_dead[polymer].count('M2')
            M_3 += Polymers_dead[polymer].count('M3')

        D_m = np.append(D_m, ((M_1*1) + (M_2*1)+ (M_3*1)))
        
        I_m_sum = I_m[-1]
        RAFT_m_sum = RAFT_m[-1]
        R_M_act_m_sum = np.sum(R_M_act_m)
        R_M_inact_m_sum = np.sum(R_M_inact_m)  
        R_m_n_m_sum = np.sum(R_m_n_m) 
        D_m_sum = np.sum(D_m)   \

        M_m = np.array([])
        M_m_sum = np.array([])
        for i in range(co_number-1):

            M_m = np.append(M_m, np.array(Species_count[i+1]))


    if poly_type == 'co' or poly_type == 'co_3':
        Molecular_weight = np.sum([ R_M_act_m_sum, R_M_inact_m_sum, R_m_n_m_sum, D_m_sum] )

    elif poly_type == 'homo':
        Molecular_weight = np.sum([RAFT_m_sum, R_M_act_m_sum, R_M_inact_m_sum, R_m_n_m_sum, D_m_sum] )

    return Molecular_weight


def calculate_initial_molecular_weights(Species_count, initial_I_m, initial_M_m,initial_RAFT_m,poly_type, co_number):
    """
    MW calculation for all species present in the beginning of the simulation
    """
    initial_I = Species_count[0][0]

    if (poly_type == 'co') or (poly_type == 'co_3'):
        initial_RAFT = Species_count[co_number+1] 
        initial_M = []
        for i in range(1, co_number):
            initial_M = np.append(initial_M, Species_count[i])
        beginning_molecular_weight = (initial_I*initial_I_m) + np.sum(initial_M) + (initial_RAFT*initial_RAFT_m)    
    
    elif poly_type == 'homo':
        initial_M = Species_count[1][0]
        initial_RAFT = Species_count[2][0]
        beginning_molecular_weight = (initial_I*initial_I_m) + (initial_M*initial_M_m) + (initial_RAFT*initial_RAFT_m)    

    return beginning_molecular_weight

def save_polymers(Polymers, Polymers_inactive, Polymers_dead, Polymers_adduct):
    """
    save the polymer arrays in one array for plotting
    """
    all_new_polymers = []
    new_polymers = []
    new_polymers_inactive = []
    new_polymers_dead = []
    new_polymers_adduct = []

    for polymer in Polymers.values():
        new_polymers = np.append(new_polymers, len(polymer))
        all_new_polymers = np.append(all_new_polymers, len(polymer))

    for polymer_inactive in Polymers_inactive.values():
        new_polymers_inactive = np.append(new_polymers_inactive, len(polymer_inactive))
        all_new_polymers = np.append(all_new_polymers, len(polymer_inactive))

    for polymer_dead in Polymers_dead.values():
        new_polymers_dead = np.append(new_polymers_dead, len(polymer_dead))
        all_new_polymers = np.append(all_new_polymers, len(polymer_dead))

    for polymer_adduct in Polymers_adduct.values():
        new_polymers_adduct = np.append(new_polymers_adduct, len(polymer_adduct))
        all_new_polymers = np.append(all_new_polymers, len(polymer_adduct))

    return all_new_polymers, new_polymers, new_polymers_inactive,new_polymers_dead,new_polymers_adduct


def get_MW_fraction(df, Molecular_weight):
    """
    calculate the MW of each occuring chain length for each polymer type 
    """

    MW = df['chain length']
    all_MW_fraction = pd.DataFrame()

    for column in df:
        count = df[column]
        MW_fraction = count.multiply(MW, axis = 0)
        MW_fraction = MW_fraction.div(Molecular_weight)
        all_MW_fraction['{}_MW'.format(column)] = pd.Series(MW_fraction)

    all_MW_fraction = all_MW_fraction.drop(['chain length_MW'], axis=1)
    return all_MW_fraction

def concatenate_dataframes(df, all_MW_fraction):
    """
    concatenate MW fraction, count and chain length columns into one dataframe for plotting
    """
    print('pd.Series(all_MW_fraction)',all_MW_fraction)
    Polymer_species = pd.concat([df, all_MW_fraction], axis=1)
    Polymer_species.columns = ['chain length in monomers', 'all Polymers', 'Polymers', 'inactive Polymers', 'dead Polymers', '2-arm adduct Polymers', 'MW fraction all Polymers','MW fraction Polymers','MW fraction inactive Polymers','MW fraction dead Polymers', 'MW fraction adduct Polymers',]
    return Polymer_species

def concatenate_dataframes_pintos(df, all_MW_fraction):
    """
    concatenate MW fraction, count and chain length columns into one dataframe for plotting
    """
    print('pd.Series(all_MW_fraction)',all_MW_fraction)
    Polymer_species = pd.concat([df, all_MW_fraction], axis=1)
    Polymer_species.columns = ['chain length in monomers', 'all Polymers', 'Polymers', 'inactive Polymers', 'dead Polymers', '2-arm adduct Polymers', 'MW fraction all Polymers','MW fraction Polymers','MW fraction T Polymers','MW fraction inactive Polymers','MW fraction dead Polymers', 'MW fraction adduct Polymers',]
    return Polymer_species

def save_PDI_index(Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead, MW, all_PDI, loops,output_period):
    """
    concatenate MW fraction, count and chain length columns into one dataframe for plotting
    """
    #merge Polymers and Polymers_dead (these are the only polymer species in a radical polymerisation)
    
    if loops % output_period == 0 :
    
        Polymers_all = np.concatenate([Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead], axis=0)

        if sum(Polymers_all) != 0:
            weight = np.multiply(MW, Polymers_all)

            multi_weight_poly = np.multiply(weight, Polymers_all)
            M_n = np.divide(sum(multi_weight_poly), sum(Polymers_all))

            weight_2 = np.multiply(weight, weight)
            multi_weight2_poly = np.multiply(weight_2, Polymers_all)
            M_w = np.divide(sum(multi_weight2_poly), sum(multi_weight_poly))

            PDI = M_w/M_n
            all_PDI = np.append(all_PDI, PDI)
    
    return all_PDI 

def save_degree_of_polymerisation(Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead, MW, all_DPn, loops, output_period):
    if loops % output_period == 0 :
        Polymers_all = np.concatenate([Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead], axis=0)
        if sum(Polymers_all) != 0:
            DPn = sum(Polymers_all)/len(Polymers_all)
            all_DPn =  np.append(all_DPn, DPn)
    
    return all_DPn

def plot_species_dist(saved_species):
    x = np.arange(0, len(saved_species[0]))
    d = {'x' : x,'Initiator': saved_species[0], 'Monomer 1': saved_species[1],'Monomer 2': saved_species[2], 'RAFT agent': saved_species[3], 'active radicals': saved_species[4], 'inactive radicals': saved_species[5], 'adduct radicals': saved_species[6], 'Dead polymers': saved_species[7], 'active radicals + RAFT end group': saved_species[8]}
    df = pd.DataFrame(data = d)
    return df

def delete_polymer(polymer):
    del polymer

def select_reaction_for_homopolymerisation_and_SF(index, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead):
    """
    select reaction for homopolymerisation and slow fragmentation theory
    """
    if index == 0:
        X[3] = X[3] + 2
        Polymers = np.append(Polymers, [0, 0]) 
        X[0] = X[0] - 1

    elif index == 1:
        X[1] = X[1] - 1
        r2 = random.randint(0, len(Polymers)-1)
        Polymers[r2] = Polymers[r2] + 1

    elif index == 2:
        X[2] = X[2] - 1
        X[3] = X[3] - 1
        X[7] = X[7] + 1
        r2 = random.randint(0, len(Polymers)-1)
        T_Polymers = np.append(T_Polymers, Polymers[r2])
        Polymers = np.delete(Polymers, r2)

    elif index == 3:
        X[2] = X[2] + 1
        X[3] = X[3] + 1
        X[7] = X[7] - 1
        r2 = random.randint(0, len(T_Polymers)-1)
        Polymers = np.append(Polymers, T_Polymers[r2])
        T_Polymers = np.delete(T_Polymers, r2)

    elif index == 4:
        X[4] = X[4] + 1
        X[3] = X[3] + 1
        X[7] = X[7] - 1
        r2 = random.randint(0, len(T_Polymers)-1)
        Polymers_inactive = np.append(Polymers_inactive, T_Polymers[r2])
        T_Polymers = np.delete(T_Polymers, r2)
        Polymers = np.append(Polymers, 0)
        
    elif index == 5:
        r2 = random.randint(0, len(Polymers)-1)
        Polymers_adduct_n = np.append(Polymers_adduct_n, Polymers[r2])
        Polymers = np.delete(Polymers, r2)
        X[3] = X[3] - 1
        r3 = random.randint(0, len(Polymers_inactive)-1)
        Polymers_adduct_m = np.append(Polymers_adduct_m, Polymers_inactive[r3])
        Polymers_inactive = np.delete(Polymers_inactive, r3)   
        X[4] = X[4] - 1 
        X[5] = X[5] + 1    
    
    elif index == 6:
        r2 = random.randint(0, len(Polymers_adduct_m)-1)
        Polymers = np.append(Polymers, Polymers_adduct_n[r2])
        Polymers_adduct_n = np.delete(Polymers_adduct_n, r2)  
        Polymers_inactive = np.append(Polymers_inactive, Polymers_adduct_m[r2])
        Polymers_adduct_m = np.delete(Polymers_adduct_m, r2)  
        X[5] = X[5] - 1       
        X[3] = X[3] + 1
        X[4] = X[4] + 1  

    elif index == 7:
        r2 = random.randint(0, len(Polymers_adduct_m)-1)
        Polymers = np.append(Polymers, Polymers_adduct_m[r2])
        Polymers_adduct_m = np.delete(Polymers_adduct_m, r2)  
        Polymers_inactive = np.append(Polymers_inactive, Polymers_adduct_n[r2])
        Polymers_adduct_n = np.delete(Polymers_adduct_n, r2)  
        X[5] = X[5] - 1       
        X[3] = X[3] + 1
        X[4] = X[4] + 1  

    elif index == 8:
        if X[3] >= 2:
            r2 = random.randint(0, len(Polymers)-1)
            r3 = random.randint(0, len(Polymers)-1)
            while r3 == r2:
                r3 = random.randint(0, len(Polymers)-1)
            Polymers_sum = (Polymers[r2] + Polymers[r3])
            Polymers_dead = np.append(Polymers_dead, Polymers_sum)
            Polymers = np.delete(Polymers, r2) 
            Polymers = np.delete(Polymers, (r3-1)) 
            X[3] = X[3] - 2
            X[6] = X[6] + 1    

    elif index == 9:
        if X[3] >= 2:
            r2 = random.randint(0, len(Polymers)-1)
            Polymers_dead = np.append(Polymers_dead, Polymers[r2])
            Polymers = np.delete(Polymers, r2) 
            X[3] = X[3] - 1
            r3 = random.randint(0, len(Polymers)-1)
            Polymers_dead = np.append(Polymers_dead, Polymers[r3])
            Polymers = np.delete(Polymers, r3) 
            X[3] = X[3] - 1
            X[6] = X[6] + 2

    elif index == 10:
        if min(Polymers) == 0:
            r2 = random.randint(0, len(Polymers_adduct_n)-1)
            Polymers_dead = np.append(Polymers_dead, (Polymers_adduct_n[r2] + Polymers_adduct_m[r2]))
            Polymers_adduct_n = np.delete(Polymers_adduct_n, r2) 
            Polymers_adduct_m = np.delete(Polymers_adduct_m, r2) 
            X[5] = X[5] - 1
            a = np.where(Polymers == 0) #Polymers.index(0)
            d = np.random.choice(a[0], size = 1)
            Polymers = np.delete(Polymers, d)
            X[3] = X[3] - 1

    elif index == 11:
        if min(Polymers) <= 2:
            r2 = random.randint(0, len(Polymers_adduct_n)-1)
            Polymers_dead = np.append(Polymers_dead, (Polymers_adduct_n[r2] + Polymers_adduct_m[r2]))
            Polymers_adduct_n = np.delete(Polymers_adduct_n, r2) 
            Polymers_adduct_m = np.delete(Polymers_adduct_m, r2) 
            X[5] = X[5] - 1
            if min(Polymers) == 0:
                a = np.where(Polymers == 0)
                d = np.random.choice(a[0], size = 1)

            elif min(Polymers) == 1:
                b = np.where(Polymers == 1) 
                d = np.random.choice(b[0], size = 1) 

            elif min(Polymers) == 2:
                c = np.where(Polymers == 2)
                d = np.random.choice(c[0], size = 1) 
            Polymers = np.delete(Polymers, d)
            X[3] = X[3] - 1
     
    return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead


def select_reaction_for_copolymerisation_and_SF_new_zapata(index, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead):
    """
    select reaction for copolymerisation and slow fragmentation theory
    """
    if index == 0:
        X[4] = X[4] + 2
        for zeros in range(2):
            if len(Polymers) == 0:
                Polymers['Polymers_0'] = ([0])
            else:
                last_polymer = list(Polymers.keys())[-1]
                Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] = ([0])
        X[0] = X[0] - 1

    elif index == 1:
        if len(Polymers) != 0:
            X[1] = X[1] - 1
            r2 = random.choice(list(Polymers.keys()))
            Polymers['{}'.format(r2)].append('M1')

    elif index == 2:
        if len(Polymers) != 0:
            X[2] = X[2] - 1
            r2 = random.choice(list(Polymers.keys()))
            Polymers['{}'.format(r2)].append('M2')

    elif index == 3:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[3] = X[3] - 1
        X[4] = X[4] - 1
        X[8] = X[8] + 1

        if len(T_Polymers) == 0:
            T_Polymers['{}'.format('T_Polymers_0')] =  Polymers['{}'.format(r2)]
        else:
            last_polymer = list(T_Polymers.keys())[-1]
            T_Polymers['{}'.format('T_Polymers_' + str(int(last_polymer[11::]) +1 ) )  ] =  Polymers['{}'.format(r2)]

        Polymers.pop('{}'.format(r2))

    elif index == 4:
        temp_Polymers = dict()

        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead

        X[3] = X[3] - 1
        X[4] = X[4] - 1
        X[8] = X[8] + 1
        if len(T_Polymers) == 0:
            T_Polymers['{}'.format('T_Polymers_0')] =  Polymers['{}'.format(r2)]
        else:
            last_polymer = list(T_Polymers.keys())[-1]
            T_Polymers['{}'.format('T_Polymers_' + str(int(last_polymer[11::]) +1 ) )  ] =  Polymers['{}'.format(r2)]

        Polymers.pop('{}'.format(r2))    
            
    elif index == 5:
        temp_T_Polymers = dict()
        for (key, value) in T_Polymers.items():
            if value[-1] == 'M1':
                temp_T_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_T_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[3] = X[3] + 1
        X[4] = X[4] + 1
        X[8] = X[8] - 1
        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] =  T_Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] =  T_Polymers['{}'.format(r2)]
        T_Polymers.pop('{}'.format(r2))

    elif index == 6:
        temp_T_Polymers = dict()
        for (key, value) in T_Polymers.items():
            if value[-1] == 'M2':
                temp_T_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_T_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[3] = X[3] + 1
        X[4] = X[4] + 1
        X[8] = X[8] - 1
        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] =  T_Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] =  T_Polymers['{}'.format(r2)]
        T_Polymers.pop('{}'.format(r2))

    elif index == 7:
        temp_T_Polymers = dict()
        for (key, value) in T_Polymers.items():
            if value[-1] == 'M1':
                temp_T_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_T_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[5] = X[5] + 1
        X[4] = X[4] + 1
        X[8] = X[8] - 1
        if len(Polymers_inactive) == 0:
            Polymers_inactive['{}'.format('Polymers_inactive_0')] =  T_Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_inactive.keys())[-1]
            Polymers_inactive['{}'.format('Polymers_inactive_' + str(int(last_polymer[18::]) +1 ) )  ] =  T_Polymers['{}'.format(r2)]

        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] = ([0])
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_'+ str(int(last_polymer[9::])+1) )  ] = ([0])

        T_Polymers.pop('{}'.format(r2))

    elif index == 8:
        temp_T_Polymers = dict()

        for (key, value) in T_Polymers.items():
            if value[-1] == 'M2':
                temp_T_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_T_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead

        X[5] = X[5] + 1
        X[4] = X[4] + 1
        X[8] = X[8] - 1
        if len(Polymers_inactive) == 0:
            Polymers_inactive['{}'.format('Polymers_inactive_0')] = T_Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_inactive.keys())[-1]
            Polymers_inactive['{}'.format('Polymers_inactive_' + str(int(last_polymer[18::]) +1 ) )  ] =  T_Polymers['{}'.format(r2)]

        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] = ([0])
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_'+ str(int(last_polymer[9::])+1))   ] = ([0])
        T_Polymers.pop('{}'.format(r2))

    elif index == 9:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 1
        X[5] = X[5] - 1
        X[6] = X[6] + 1
        if len(Polymers_adduct_n) == 0:
            Polymers_adduct_n['{}'.format('Polymers_adduct_0')] = Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_adduct_n.keys())[-1]
            Polymers_adduct_n['{}'.format('Polymers_adduct_' + str(int(last_polymer[16::]) +1 ) )  ] =  Polymers['{}'.format(r2)]

        Polymers.pop('{}'.format(r2))

        r3 = random.choice(list(Polymers_inactive.keys()))
        if len(Polymers_adduct_m) == 0:
            Polymers_adduct_m['{}'.format('Polymers_adduct_0')] = Polymers_inactive['{}'.format(r3)]
        else:
            last_polymer = list(Polymers_adduct_m.keys())[-1]
            Polymers_adduct_m['{}'.format('Polymers_adduct_' + str(int(last_polymer[16::]) +1 ) )  ] =  Polymers_inactive['{}'.format(r3)]

        Polymers_inactive.pop('{}'.format(r3))

    elif index == 10:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 1
        X[5] = X[5] - 1
        X[6] = X[6] + 1
        if len(Polymers_adduct_n) == 0:
            Polymers_adduct_n['{}'.format('Polymers_adduct_0')] = Polymers['{}'.format(r2)] 
        else:
            last_polymer = list(Polymers_adduct_n.keys())[-1]
            Polymers_adduct_n['{}'.format('Polymers_adduct_' + str(int(last_polymer[16::]) +1 ) )  ] =  Polymers['{}'.format(r2)]
        Polymers.pop('{}'.format(r2))
        r3 = random.choice(list(Polymers_inactive.keys()))
        if len(Polymers_adduct_m) == 0:
            Polymers_adduct_m['{}'.format('Polymers_adduct_0')] = Polymers_inactive['{}'.format(r3)]   
        else:     
            last_polymer = list(Polymers_adduct_m.keys())[-1]
            Polymers_adduct_m['{}'.format('Polymers_adduct_' + str(int(last_polymer[16::]) +1 ) )  ] =  Polymers_inactive['{}'.format(r3)]
        Polymers_inactive.pop('{}'.format(r3))

    elif index == 11:
        temp_Polymers_adduct_n = dict()
        for (key, value) in Polymers_adduct_n.items():
            if value[-1] == 'M1':
                temp_Polymers_adduct_n[key] = value
        try:
            r2 = random.choice(list(temp_Polymers_adduct_n.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[6] = X[6] - 1
        X[4] = X[4] + 1
        X[5] = X[5] + 1
        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] = Polymers_adduct_n['{}'.format(r2)]
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] =  Polymers_adduct_n['{}'.format(r2)]

        Polymers_adduct_n.pop('{}'.format(r2))
        if len(Polymers_inactive) == 0:
            Polymers_inactive['{}'.format('Polymers_inactive_0')] = Polymers_adduct_m['{}'.format(r2)]

        last_polymer = list(Polymers_inactive.keys())[-1]
        Polymers_inactive['{}'.format('Polymers_inactive_' + str(int(last_polymer[18::]) +1 ) )  ] =  Polymers_adduct_m['{}'.format(r2)]
    
        Polymers_adduct_m.pop('{}'.format(r2))

    elif index == 12:
        temp_Polymers_adduct_n = dict()
        for (key, value) in Polymers_adduct_n.items():
            if value[-1] == 'M2':
                temp_Polymers_adduct_n[key] = value
        try:
            r2 = random.choice(list(temp_Polymers_adduct_n.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[6] = X[6] - 1
        X[4] = X[4] + 1
        X[5] = X[5] + 1
        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] = Polymers_adduct_n['{}'.format(r2)]
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] =  Polymers_adduct_n['{}'.format(r2)]
        Polymers_adduct_n.pop('{}'.format(r2))
        if len(Polymers_inactive) == 0:
            Polymers_inactive['{}'.format('Polymers_inactive_0')] = Polymers_adduct_m['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_inactive.keys())[-1]
            Polymers_inactive['{}'.format('Polymers_inactive_' + str(int(last_polymer[18::]) +1 ) )  ] =  Polymers_adduct_m['{}'.format(r2)]
        Polymers_adduct_m.pop('{}'.format(r2))

    elif index == 13:
        temp_Polymers_adduct_m = dict()
        for (key, value) in Polymers_adduct_m.items():
            if value[-1] == 'M1':
                temp_Polymers_adduct_m[key] = value
        try:
            r2 = random.choice(list(temp_Polymers_adduct_m.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[6] = X[6] - 1
        X[4] = X[4] + 1
        X[5] = X[5] + 1
        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] = Polymers_adduct_m['{}'.format(r2)]
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] =  Polymers_adduct_m['{}'.format(r2)]

        Polymers_adduct_m.pop('{}'.format(r2))
        if len(Polymers_inactive) == 0:
            Polymers_inactive['{}'.format('Polymers_inactive_0')] = Polymers_adduct_n['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_inactive.keys())[-1]
            Polymers_inactive['{}'.format('Polymers_inactive_' + str(int(last_polymer[18::]) +1 ) )  ] =  Polymers_adduct_n['{}'.format(r2)]
        Polymers_adduct_n.pop('{}'.format(r2))

    elif index == 14:
        temp_Polymers_adduct_m = dict()
        for (key, value) in Polymers_adduct_m.items():
            if value[-1] == 'M2':
                temp_Polymers_adduct_m[key] = value
        try:
            r2 = random.choice(list(temp_Polymers_adduct_m.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[6] = X[6] - 1
        X[4] = X[4] + 1
        X[5] = X[5] + 1
        if len(Polymers) == 0:
            Polymers['{}'.format('Polymers_0')] = Polymers_adduct_m['{}'.format(r2)]
        else:
            last_polymer = list(Polymers.keys())[-1]
            Polymers['{}'.format('Polymers_' + str(int(last_polymer[9::]) +1 ) )  ] =  Polymers_adduct_m['{}'.format(r2)]

        Polymers_adduct_m.pop('{}'.format(r2))
        if len(Polymers_inactive) == 0:
            Polymers_inactive['{}'.format('Polymers_inactive_0')] = Polymers_adduct_n['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_inactive.keys())[-1]
            Polymers_inactive['{}'.format('Polymers_inactive_' + str(int(last_polymer[18::]) +1 ) )  ] =  Polymers_adduct_n['{}'.format(r2)]
        Polymers_adduct_n.pop('{}'.format(r2))

    elif index == 15:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r1, r2 = np.random.choice(list(temp_Polymers.keys()), 2, replace=False)
        except ValueError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead

        X[4] = X[4] - 2
        X[7] = X[7] + 1
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = (*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r1)] )#Polymers['{}'.format(r2)].append(Polymers['{}'.format(r1)])
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] = (*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r1)] )

        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r1))


    elif index == 16:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r3 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead

        X[4] = X[4] - 2
        X[7] = X[7] + 1
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] =(*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r3)] )
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] = (*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r3)] )
        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r3))

    elif index == 17:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r3 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 2
        X[7] = X[7] + 1
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] =(*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r3)] )#Polymers['{}'.format(r2)].append(Polymers['{}'.format(r3)])
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] = (*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r3)] )#Polymers['{}'.format(r2)].append(Polymers['{}'.format(r3)])
        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r3))
    
    elif index == 18:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r1, r2 = np.random.choice(list(temp_Polymers.keys()), 2, replace=False)#random.randint(0, len(Polymers)-1)
        except ValueError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 2
        X[7] = X[7] + 1
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = (*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r1)] )#Polymers['{}'.format(r2)].append(Polymers['{}'.format(r1)])
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ]  = (*Polymers['{}'.format(r2)] , *Polymers['{}'.format(r1)] )#Polymers['{}'.format(r2)].append(Polymers['{}'.format(r1)])
        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r1))


    elif index == 19:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r1, r2 = np.random.choice(list(temp_Polymers.keys()), 2, replace=False)#random.randint(0, len(Polymers)-1)
        except ValueError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 2
        X[7] = X[7] + 2
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r2)]
        last_polymer = list(Polymers_dead.keys())[-1]
        Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r1)]

        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r1))

    elif index == 20:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r3 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 2
        X[7] = X[7] + 2
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r2)]
        last_polymer = list(Polymers_dead.keys())[-1]
        Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r3)]
        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r3))

    elif index == 21:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M1':
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))#random.randint(0, len(Polymers)-1)
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r3 = random.choice(list(temp_Polymers.keys()))#random.randint(0, len(Polymers)-1)
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 2
        X[7] = X[7] + 2
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r2)]
        last_polymer = list(Polymers_dead.keys())[-1]
        Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r3)]
        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r3))

    elif index == 22:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if value[-1] == 'M2':
                temp_Polymers[key] = value
        try:
            r1, r2 = np.random.choice(list(temp_Polymers.keys()), 2, replace=False)#random.randint(0, len(Polymers)-1)
        except ValueError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[4] = X[4] - 2
        X[7] = X[7] + 2
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = Polymers['{}'.format(r2)]
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r2)]
        last_polymer = list(Polymers_dead.keys())[-1]
        Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers['{}'.format(r1)]

        Polymers.pop('{}'.format(r2))
        Polymers.pop('{}'.format(r1))

    elif index == 23:
        temp_Polymers_adduct_n = dict()
        for (key, value) in Polymers_adduct_n.items():
            if value[-1] == '0':
                temp_Polymers_adduct_n[key] = value
        try:
            r2 = random.choice(list(temp_Polymers_adduct_n.keys()))#random.randint(0, len(Polymers)-1)
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        X[3] = X[3] - 1
        X[5] = X[5] - 1
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = (*Polymers_adduct_n['{}'.format(r2)] , *Polymers_adduct_m['{}'.format(r2)] )#Polymers_adduct_n['{}'.format(r2)].append(Polymers_adduct_m['{}'.format(r2)])
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  (*Polymers_adduct_n['{}'.format(r2)] , *Polymers_adduct_m['{}'.format(r2)] )##Polymers_adduct_n['{}'.format(r2)].append(Polymers_adduct_m['{}'.format(r2)])

        Polymers_adduct_m.pop('{}'.format(r2))
        Polymers_adduct_n.pop('{}'.format(r2))

    elif index == 24:
        temp_Polymers = dict()
        for (key, value) in Polymers.items():
            if len(value) == 2:
                temp_Polymers[key] = value
        try:
            r2 = random.choice(list(temp_Polymers.keys()))
        except IndexError:
            return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
        Polymers.pop('{}'.format(r2))
        r3 = random.choice(list(Polymers_adduct_n.keys()))
        X[3] = X[3] - 1
        X[5] = X[5] - 1
        if len(Polymers_dead) == 0:
            Polymers_dead['{}'.format('Polymers_dead_0')] = Polymers_adduct_n['{}'.format(r3)].append(Polymers_adduct_m['{}'.format(r3)])
        else:
            last_polymer = list(Polymers_dead.keys())[-1]
            Polymers_dead['{}'.format('Polymers_dead_' + str(int(last_polymer[14::]) +1 ) )  ] =  Polymers_adduct_n['{}'.format(r3)].append(Polymers_adduct_m['{}'.format(r3)])
        Polymers_adduct_m.pop('{}'.format(r3))
        Polymers_adduct_n.pop('{}'.format(r3))

    return X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead
