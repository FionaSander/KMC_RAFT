#!/usr/bin/env python3
"""
Created on Monday October 12 2020
@author: fionasander

-------
Monte Carlo simulation of RAFT polymerisations
-------
Monte Carlo steps:
 1. Initiation
 2. Reaction selection
 3. time step calculation
 4. reaction simulation
 5. update
"""

import numpy as np
import pandas as pd
import itertools
import random
import plotly.express as px
from pandas import Series, DataFrame
import plotly.graph_objs as go
import seaborn as sns
import plotly.figure_factory as ff
import math
from collections import Counter
from RAFT_functions import *
from datetime import datetime

def main(args):

    startTime = datetime.now()

    #----------
    # Definition of basic parameters:
    #----------
    N_A = 6.02214086 * 10**23 #avogadro constant in mol
    t = 0 #time of reaction

    #----------
    # Setting initial molecular weights to zero for all species that are not present at the start of the polymerisation
    #----------
    R_M_act_m = 0 
    R_M_inact_m = 0 
    TR_M_act_m = 0 
    R_m_n_m = 0 
    D_m = 0 

    #---------
    # User input for MC simulation
    #---------
    poly_type = 'co' #What type of polymerisation should be used (homo or co)?
    theory = 'SF'#Which polymerisation kinetics theory should be used (SF or IRTO)?
    n = args['n']

    T = 375 #polymeristion temperature in K (e.g. 375)
    initial_I_m = 100 #Molecular weight of a initiator molecules (e.g. 164)
    initial_M_m = []
    initial_M_m.append(100) #Molecular weight of one monomer - if copolymerisation, name the MW of one species (e.g. 100)
    initial_RAFT_m = 346 #Molecular weight of one RAFT molecule (e.g. 346)
    prefix = 'RAFT_polymerisation' #prefix for output files

    x = 0
    co_number = 0
    if poly_type == 'co': #for co-polymerisations, add the additional initial monomer counts
        initial_M_m.append(100) #modify if other value than 100 is required.
        co_number = len(initial_M_m)
    elif poly_type == 'homo':
        initial_M_m = initial_M_m[0]

    #---------
    # Starting MC simulation
    #---------
    if poly_type == 'homo':
        if theory == 'SF':
            #----------
            # Chemical species involved - Values for SF model (slow fragmentation)
            #----------
            I = 5*10**-3   #Initiator (mol*L**-1)
            M = 5  #monomer (mol*L**-1)
            RAFT = 10**-2  #raft agent (mol*L**-1)
            R_M_act = 0 #active radicals with n units of M
            R_M_inact = 0 #inactive radicals with n units of M
            TR_M_act = 0 #inactive radicals with n units of M and RAFT end group
            R_m_n = 0 #adduct radicals with two arms of lengths n and m
            D = 0 #dead polymer chaisn of length n
            f = 0.5 #initiator efficiency

            #----------
            # Reaction constants
            #----------
            k_d_exp = k_d = 0.036 #Initiation (h^-1)
            k_pi_exp = 3.6*10**6  #initiation propagation(L·mol^−1*h^−1)
            k_p_exp = 3.6*10**6 #propagation(L·mol^−1*h^−1)
            k_a_exp = 3.6*10**9 #addition constant (L·mol^−1*h^−1)
            k_f_exp = k_f = 36  #fragmentation constant (h^-1)
            k_tc_exp = 3.6*10**10  #termination by combination (L*mol^−1*h^−1)
            k_td_exp =3.6*10**10  #termination by disproportionation (L*mol^−1*h^−1)
            k_ct_exp = 0 #cross-termination

        elif theory == 'IRTO':
            #----------
            #Chemical species involved - Values for IRTO model
            #----------
            I = 5*10**-3   #Initiator (mol*L**-1)
            M = 5  #monomer (mol*L**-1)
            RAFT = 10**-2  #raft agent (mol*L**-1)
            R_M_act = 0 #active radicals with n units of M
            R_M_inact = 0 #inactive radicals with n units of M
            TR_M_act = 0 #inactive radicals with n units of M and RAFT end group
            R_m_n = 0 #adduct radicals with two arms of lengths n and m
            D = 0 #dead polymer chaisn of length n
            f = 0.5 #initiator efficiency

            #----------
            #Reaction constants
            #----------
            k_d_exp = k_d = 0.036 #Initiation (h^-1)
            k_pi_exp = 3.6*10**6  #initiation propagation(L·mol^−1*h^−1)
            k_p_exp = 3.6*10**6 #propagation(L·mol^−1*h^−1)
            k_a_exp = 3.6*10**9 #addition constant (L·mol^−1*h^−1)
            k_f_exp = k_f = 3.6*10**7  #fragmentation constant (h^-1)
            k_tc_exp = 3.6*10**10  #termination by combination (L*mol^−1*h^−1)
            k_td_exp =3.6*10**10  #termination by disproportionation (L*mol^−1*h^−1)
            k_ct_exp = 3.6*10**10 #cross-termination

        #---------
        #Initiation Step (STEP 1)
        #---------
        Species = [I, M, RAFT, R_M_act, R_M_inact, R_m_n, D, TR_M_act]
        Species_count, S_sum, X = initiate_particle_count(Species, n)
        k = calculate_reaction_constants(n, N_A, S_sum, f, k_d, k_f ,k_d_exp, k_pi_exp, k_p_exp, k_a_exp, k_f_exp, k_tc_exp, k_td_exp, k_ct_exp)

        Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead = create_polymer_arrays()

        time = []
        index_list = []

        #-----------
        # Monte Carlo Simulation Loop (STEPS 2 - 5)
        #-----------
        loops = 0

        all_PDI = []
        all_DPn = []

        time = np.append(time, 0)
        first_monomer_count = X[1]

        while (X[1] > 0.7 * first_monomer_count):#(X[1] > 0): #> 0.70 *n):#n):   # or  t < 34

            loops = loops + 1
            R = [] # Reaction rate array
            P = [] # Probability array
            R, R_sum, m = calculate_reaction_rates(k, X) #calc reaction rates R from reaction constant k for differen species X
            P = calculate_probabilities(m, P, R, R_sum)
            
            #--------
            #Reaction selection based on probabilities
            #--------
            X_old = X.copy()
            index = define_reaction_indices(P)
            X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead = select_reaction_for_homopolymerisation_and_SF(index, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead)

            #--------
            #Time step calculation
            #--------
            X_comparison = X_old == X
            time_step_stagnation = X_comparison.all()

            if time_step_stagnation == True:
                t = t
            else:
                t = calculate_timestep_speedy(R_sum, t)

            #--------
            #arrays containing the number of different chemical species over number of loops 
            #--------
            Species_count, time = save_values(X, Species_count, 1000, index, index_list, time, t, loops)
            all_PDI = save_PDI_index(Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead, initial_M_m, all_PDI, loops, 1000)
            all_DPn = save_degree_of_polymerisation(Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead, initial_M_m, all_DPn, loops, 1000)

        with open('all_PDI.txt', 'w') as f:
            for item in all_PDI:
                f.write("%s\n" % item)

        with open('all_DPn.txt', 'w') as g:
            for item in all_DPn:
                g.write("%s\n" % item)
        #---------
        # calculation of molecular weight of each species
        #---------
        Polymers_adduct = np.add(Polymers_adduct_n, Polymers_adduct_m)
        Molecular_weight = calculate_molecular_weights(Species_count, theory, poly_type, Polymers, Polymers_inactive, Polymers_adduct, Polymers_dead, TR_M_act, co_number,initial_M_m)

        #---------
        # molecular weight of all species together at the end of simulation (plus comparison with starting configuration)
        #---------
        beginning_molecular_weight = calculate_initial_molecular_weights(Species_count, initial_I_m,initial_M_m,initial_RAFT_m,poly_type, co_number)

        #--------
        # Save number of molecules in chemical species into dataframe
        #--------
        x = np.arange(0, len(Species_count[0]))
        d = {'x' : x,'time':time,'Initiator': Species_count[0], 'Monomer': Species_count[1], 'RAFT agent': Species_count[2], 'active radicals': Species_count[3], 'inactive radicals': Species_count[4], 'adduct radicals': Species_count[5], 'Dead polymers': Species_count[6], 'active radicals + RAFT end group': Species_count[7]}
        species_dist = pd.DataFrame(data = d)
        species_dist.to_pickle('SpeciesDist_{}poly_{}_{}_{}.pkl'.format(poly_type, theory, n, prefix) , protocol = 3)

        #--------
        # Plot of MW fraction over chain length
        #--------
        new_polymers = save_polymers(Polymers, Polymers_inactive, Polymers_dead, Polymers_adduct)
        count_polymers = Counter(new_polymers)
        df = pd.DataFrame.from_dict(count_polymers, orient='index').reset_index()
        df.columns = ['chain length', 'count']

        #--------
        # Plot of MW fraction over chain length
        #--------
        all_new_polymers, new_polymers, new_polymers_inactive,new_polymers_dead,new_polymers_adduct = save_homopolymers(Polymers, Polymers_inactive, Polymers_dead, Polymers_adduct)
        count_all_polymers = Counter(all_new_polymers)
        count_polymers = Counter(new_polymers)
        count_polymers_inactive = Counter(new_polymers_inactive)
        count_polymers_dead = Counter(new_polymers_dead)
        count_polymers_adduct = Counter(new_polymers_adduct)

        df = pd.DataFrame.from_dict(count_all_polymers, orient='index').reset_index()
        df.columns = ['chain length', 'count all polymers']

        df['count polymers'] = df['chain length'].map(count_polymers) 
        df['count inactive polymers'] = df['chain length'].map(count_polymers_inactive)  
        df['count dead polymers'] = df['chain length'].map(count_polymers_dead) 
        df['count adduct polymers'] = df['chain length'].map(count_polymers_adduct) 

        MW_fraction = get_MW_fraction(df, Molecular_weight) #calculate the MW fraction from the pd.DataFrame 
        Polymer_species = concatenate_dataframes(df, MW_fraction)

        #--------
        # Scatter plot and save into pickle
        #--------
        Polymer_species.to_pickle('{}poly_{}_{}_{}.pkl'.format(poly_type, theory, n, prefix) , protocol = 3)
        raw_polymers = pd.DataFrame([Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead])
        raw_polymers_T = raw_polymers.T
        raw_polymers_T.to_pickle('rawpolys_{}poly_{}_{}_{}.pkl'.format(poly_type, theory, n, prefix) , protocol = 3)


        #--------
        # Plotting Polydispersity index PDI 
        #--------
        plot_PDI_index(Polymer_species, MW)

    elif poly_type == 'co':
        if theory == 'SF':
            #----------
            # Chemical species involved - Values for SF model (slow fragmentation)
            #----------
            I = 5*10**-3   #Initiator (mol*L**-1)
            M1 = 2.5  #monomer (mol*L**-1)
            M2 = 2.5
            RAFT = 10**-2  #10**-2  #raft agent (mol*L**-1)
            R_M_act = 0 #active radicals with n units of M
            R_M_inact = 0 #inactive radicals with n units of M
            TR_M_act = 0 #inactive radicals with n units of M and RAFT end group
            R_m_n = 0 #adduct radicals with two arms of lengths n and m
            D = 0 #dead polymer chaisn of length n
            f = 0.5 #initiator efficiency

            #----------
            # Reaction constants
            #----------
            k_d_exp = k_d = 0.036 #Initiation (h^-1)
            k_pi_1_exp = (3.6*10**6)/2 #initiation propagation(L·mol^−1*h^−1) MMA
            k_pi_2_exp = (3.6*10**6)/2  #initiation propagation(L·mol^−1*h^−1) Styrene
            k_p_1_exp = (3.6*10**6)/2   #propagation(L·mol^−1*h^−1)
            k_p_2_exp = (3.6*10**6)/2   #propagation(L·mol^−1*h^−1)
            k_a_1_exp = (3.6*10**9)/2 #addition constant (L·mol^−1*h^−1) MMA
            k_a_2_exp = (3.6*10**9)/2 #addition constant (L·mol^−1*h^−1) Styrene
            k_f_1_exp = k_f_1 = 36#/2 #/2 #fragmentation constant (h^-1)
            k_f_2_exp = k_f_2 = 36#/2 #/2 #fragmentation constant (h^-1)
            k_tc_11_exp = (3.6*10**10)/3  #termination by combination (L*mol^−1*h^−1)
            k_tc_12_exp = (3.6*10**10)/3  #termination by combination (L*mol^−1*h^−1)
            k_tc_21_exp = (3.6*10**10)/3 #termination by combination (L*mol^−1*h^−1)
            k_tc_22_exp = (3.6*10**10)/3 #termination by combination (L*mol^−1*h^−1)
            k_td_11_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_td_12_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_td_21_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_td_22_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_ct_exp = 0 #cross-termination

        elif theory == 'IRTO':
            #----------
            #Chemical species involved - Values for IRTO model
            #----------
            I = 5*10**-3   #Initiator (mol*L**-1)
            M1 = 2.5  #monomer (mol*L**-1)
            M2 = 2.5
            RAFT = 10**-2  #raft agent (mol*L**-1)
            R_M_act = 0 #active radicals with n units of M
            R_M_inact = 0 #inactive radicals with n units of M
            TR_M_act = 0 #inactive radicals with n units of M and RAFT end group
            R_m_n = 0 #adduct radicals with two arms of lengths n and m
            D = 0 #dead polymer chaisn of length n
            f = 0.5 #initiator efficiency

            k_d_exp = k_d = 0.036 #Initiation (h^-1)
            k_pi_1_exp = (3.6*10**6)/2 #initiation propagation(L·mol^−1*h^−1) MMA
            k_pi_2_exp = (3.6*10**6)/2 #initiation propagation(L·mol^−1*h^−1) Styrene
            k_p_1_exp = (3.6*10**6)/2 #propagation(L·mol^−1*h^−1)
            k_p_2_exp = (3.6*10**6)/2 #propagation(L·mol^−1*h^−1)
            k_a_1_exp = (3.6*10**9)/2#addition constant (L·mol^−1*h^−1) MMA
            k_a_2_exp = (3.6*10**9)/2#addition constant (L·mol^−1*h^−1) Styrene
            k_f_1_exp = k_f_1 = (3.6*10**7)#/2 #fragmentation constant (h^-1)
            k_f_2_exp = k_f_2 = (3.6*10**7)#/2 #fragmentation constant (h^-1)
            k_tc_11_exp = (3.6*10**10)/3  #termination by combination (L*mol^−1*h^−1)
            k_tc_12_exp = (3.6*10**10)/3  #termination by combination (L*mol^−1*h^−1)
            k_tc_21_exp = (3.6*10**10)/3 #termination by combination (L*mol^−1*h^−1)
            k_tc_22_exp = (3.6*10**10)/3 #termination by combination (L*mol^−1*h^−1)
            k_td_11_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_td_12_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_td_21_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_td_22_exp = (3.6*10**10)/3  #termination by disproportionation (L*mol^−1*h^−1)
            k_ct_exp = 3.6*10**10 #cross-termination

        #---------
        # Initiation Step (STEP 1)
        #---------
        Species = [I, M1,M2, RAFT, R_M_act, R_M_inact, R_m_n, D, TR_M_act]
        Species_count, S_sum, X = initiate_particle_count(Species, n)
        k = calculate_reaction_constants_copoly(n, N_A, S_sum, f, k_d, k_pi_1_exp, k_pi_2_exp, k_p_1_exp, k_p_2_exp, k_a_1_exp, k_a_2_exp, k_f_1, k_f_2, k_tc_11_exp, k_tc_12_exp, k_tc_21_exp, k_tc_22_exp, k_td_11_exp, k_td_12_exp, k_td_21_exp, k_td_22_exp, k_ct_exp)

        Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead = create_polymer_dicts()

        time = []
        index_list = []

        #-----------
        # Monte Carlo Simulation Loop (STEPS 2 - 5)
        #-----------
        loops = 0
        first_monomer_count = X[1]
        MC_steps = 0
        while (X[1] > 0.7 * first_monomer_count):
            MC_steps += 1
            loops = loops + 1
            R = [] # Reaction rate array
            P = [] # Probability array
            R, R_sum, m = calculate_reaction_rates_copoly_minimised_zapata(k, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead) #calc reaction rates R 
            P = calculate_probabilities(m, P, R, R_sum)

            #--------
            # Reaction selection based on probabilities
            #--------
            index = define_reaction_indices(P)
            X_old = X.copy()
            X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead = select_reaction_for_copolymerisation_and_SF_new_zapata(index, X, Polymers, T_Polymers, Polymers_inactive, Polymers_adduct_n, Polymers_adduct_m, Polymers_dead)

            #--------
            # Time step calculation
            #--------
            X_comparison = X_old == X
            time_step_stagnation = X_comparison.all()
            if time_step_stagnation == True:
                t = t
            else:
                t = calculate_timestep(R_sum, t)

            #--------
            # arrays containing the number of different chemical species over number of loops 
            #--------
            Species_count = save_values_copoly(X, Species_count, 1000, index, index_list, time, t, loops)

        #---------
        # calculation of molecular weight of each species
        #---------
        Polymers_adduct = Polymers_adduct_n.copy()
        for key in Polymers_adduct.keys():
            Polymers_adduct[str(key)] = list(np.append(Polymers_adduct[str(key)], Polymers_adduct_m[str(key)]))
        Molecular_weight = calculate_molecular_weights(Species_count, theory, poly_type, Polymers, Polymers_inactive, Polymers_adduct, Polymers_dead, TR_M_act, co_number, initial_M_m)

        #---------
        # molecular weight of all species together at the end of simulation (plus comparison with starting configuration)
        #---------
        beginning_molecular_weight = calculate_initial_molecular_weights(Species_count, initial_I_m,initial_M_m,initial_RAFT_m,poly_type, co_number)

        #--------
        # Save number of molecules in chemical species into dataframe
        #--------
        x = np.arange(0, len(Species_count[0]))
        d = {'x' : x,'Initiator': Species_count[0], 'Monomer 1': Species_count[1], 'Monomer 2': Species_count[2], 'RAFT agent': Species_count[3], 'active radicals': Species_count[4], 'inactive radicals': Species_count[5], 'adduct radicals': Species_count[6], 'Dead polymers': Species_count[7], 'active radicals + RAFT end group': Species_count[8]}
        species_dist = pd.DataFrame(data = d)
        species_dist.to_pickle('SpeciesDist_{}poly_{}_{}_{}.pkl'.format(poly_type, theory, n, prefix) , protocol = 3)

        #--------
        # Plot of MW fraction over chain length
        #--------
        all_new_polymers, new_polymers, new_polymers_inactive,new_polymers_dead,new_polymers_adduct = save_polymers(Polymers, Polymers_inactive, Polymers_dead, Polymers_adduct)
        count_all_polymers = Counter(all_new_polymers)
        count_polymers = Counter(new_polymers)
        count_polymers_inactive = Counter(new_polymers_inactive)
        count_polymers_dead = Counter(new_polymers_dead)
        count_polymers_adduct = Counter(new_polymers_adduct)
        df = pd.DataFrame.from_records( list(dict(count_all_polymers).items())   , columns=['chain length','count all polymers'])
        df['count polymers'] = df['chain length'].map(count_polymers)     
        df['count polymers inactive'] = df['chain length'].map(count_polymers_inactive)     
        df['count polymers dead'] = df['chain length'].map(count_polymers_dead)     
        df['count polymers adduct'] = df['chain length'].map(count_polymers_adduct)     
 
        all_MW_fraction = get_MW_fraction(df, Molecular_weight) #calculate the MW fraction from pd.DataFrame
        Polymer_species = concatenate_dataframes(df, all_MW_fraction)

        #--------
        # Scatter plot and save into pickle
        #--------
        Polymer_species.to_pickle('{}poly_{}_{}_{}.pkl'.format(poly_type, theory, n, prefix) , protocol = 3)

        #--------
        # Plotting Polydispersity index PDI 
        #--------
        plot_PDI_index(Polymer_species, MW)

    print('simulation time:', datetime.now() - startTime)
    print('reaction time:', t)

    return

if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("-n", "--n", type=int, default=10000000, help='number of molecules in the system (e.g. 10000000)')
    main(vars(parser.parse_args()))
