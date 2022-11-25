#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 10:21:55 2022

@author: vishalkamat
"""

import pandas as pd
import numpy as np
from npdemand import *

#------------------------------------------------------------------------------
# Load data
#------------------------------------------------------------------------------

dataagg                 =  pickle.load(open("voucher_datause.pkl",'rb'))
pi, beta_u, Sigma, n, B = dataagg['pi'], dataagg['beta_u'], dataagg['Sigma'], dataagg['n'], dataagg['B'] 
data_sch                = pd.read_csv('school_char.txt', sep=",", header=0)

#------------------------------------------------------------------------------
# Define options + prices
#------------------------------------------------------------------------------

J_true      = len(data_sch)
p0          = {}
for j in range(2,J_true):
    p0[j] = data_sch.iloc[j,0+1]
P_uniq      = list(set([p0[j] for j in range(2,J_true)]))
P_uniq.sort()
J           = range(len(P_uniq)+2)

#------------------------------------------------------------------------------
# Observed price variation and shares
#------------------------------------------------------------------------------ 

tau_sq                  = 7500
p_with_vouch, p_wo_vuch = np.zeros(len(J)), np.zeros(len(J))
for j in J:
    if j in [0,1]:
        p_wo_vuch[j]    = 0
    else:
        p_wo_vuch[j]    = P_uniq[j-2] #int(round(P_uniq[j-2] / float(500))) * 500 #CHANGE
    p_with_vouch[j]     = max(p_wo_vuch[j] - tau_sq,0)
P_obs                   = [p_wo_vuch,p_with_vouch]    

share                   = [beta_u[0][0:53],beta_u[0][53:106]]

#------------------------------------------------------------------------------
# Two prices at which we want to do welfare effects
#------------------------------------------------------------------------------ 

p_b, p_a    = p_with_vouch, p_wo_vuch

#------------------------------------------------------------------------------
# Compute AB,AC,AS
#------------------------------------------------------------------------------ 

#AB

g_ab, g_b, g_a  = 1, np.zeros(len(J)), np.zeros(len(J))
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPB')
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPS')
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='PS',K=1,grid_size=6)
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='PS',K=3,grid_size=6)

#AC
g_ab, g_b, g_a    = 0, np.zeros(len(J)), np.zeros(len(J))
for j in J:
    if j == 0:
        g_b[j], g_a[j] = 5355, -5355
    elif j == 1:
        g_b[j], g_a[j] = 0, 0
    else:
        g_b[j], g_a[j] = (min(7500,p_wo_vuch[j]) + 200), 0
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPB')
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPS')
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='PS',K=1,grid_size=6)
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='PS',K=3,grid_size=6)

#AS
g_ab, g_b, g_a    = 1, np.zeros(len(J)), np.zeros(len(J))
for j in J:
    if j == 0:
        g_b[j], g_a[j] = -5355, 5355
    elif j == 1:
        g_b[j], g_a[j] = 0, 0
    else:
        g_b[j], g_a[j] = -(min(7500,p_wo_vuch[j]) + 200), 0
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPB')
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPS')
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='PS',K=1,grid_size=6)
est             = npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='PS',K=3,grid_size=6)