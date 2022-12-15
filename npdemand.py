#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 11:57:36 2022

@author: vishalkamat
"""

import numpy as np
from gurobipy import *
import math
#import scipy
#import scipy.special
import gurobipy as gp
import scipy.sparse as sp

def est_npdemand(A_u,A_k,beta_u,beta_k,A_obj,meth):
    
    ###########################################################################
    # Max/min A_obj x subject to A_u x = b_u, A_k >= b_k
    # A_u, b_u = matrix and RHS for estimated componets, i.e. data matching restrictions
    # A_k, b_k = matrix and RHS for non-estimated components, i.e. shape restrictions
    # If meth == 1 then do not use data matching restrictions  and if meth == 2 then use error criterion
    ###########################################################################
    
    #Rescale objective for numerical stability
    scale     = max(abs(np.max(A_obj)),abs(np.min(A_obj)))
    A_obj     = A_obj / max(scale,1)
    
    #Store dimensions and combine the data matching + shape restrictions
    p_u, d    = A_u.shape
    p_k, d    = A_k.shape
    
    #Define linear program and store results in est
    m         = gp.Model()
    x         = m.addMVar(d, lb=-GRB.INFINITY)
    m.addConstr(A_k @ x >= beta_k.reshape((p_k,)))
    
    if meth == 0:
        m.addConstr(A_u @ x == beta_u.reshape((p_u,)))
    elif meth == 2:
        obj_m   = m.addMVar(p_u, lb=0)
        obj_p   = m.addMVar(p_u, lb=0)
        m.addConstr(beta_u.reshape((p_u,)) - A_u @ x ==  obj_m - obj_p )
        m.setObjective(np.ones((1,p_u)) @ obj_m + np.ones((1,p_u)) @ obj_p, GRB.MINIMIZE)
        m.setParam('OutputFlag',False)
        m.setParam('Method',2)
        m.setParam('Crossover',0)
        m.reset(0)
        m.optimize() 
        m.addConstr(np.ones((1,p_u)) @ obj_m + np.ones((1,p_u)) @ obj_p <= m.objVal * (1.00001))
    
    est       = {}
    status    = 0
    for bound in [0,1]:
        if bound == 0: 
            m.setObjective(A_obj @ x, GRB.MINIMIZE)
        elif bound == 1:
            m.setObjective(A_obj @ x, GRB.MAXIMIZE)
        m.setParam('OutputFlag',False)
        try:
            m.setParam('Method',2)
            m.setParam('Crossover',0)
            m.reset(0)
            m.optimize()    
            if m.status != 2:
                m.setParam('Method',-1)
                m.reset(0)
                m.optimize()   
            est[bound] = m.objVal * scale
        except:
            try:
                m.setParam('Method',1)
                m.reset(0)
                m.optimize()    
                est[bound] = m.objVal * scale
            except:
                m.setParam('Method',-1)
                m.reset(0)
                m.optimize()
                est[bound] = m.objVal * scale
        if m.status != 2:
            status += 1
    est[2]    = status
    
    return est   

def test_npdemand(A_u,A_k,beta_u,beta_k,A_obj,Sigma,B,theta_0,n,kap,alpha):
    
    ###########################################################################
    # Test there exists x s.t. A_u x = b_u, A_k x >= b_k, A_obj x = theta using
    # Bugni et al (2017) test procedure.
    # A_u, b_u = matrix and RHS for estimated componets, i.e. data matching restrictions
    # A_k, b_k = matrix and RHS for non-estimated components, i.e. shape restrictions
    ###########################################################################
    
    #-------------------------------------------------------------------------
    # 0. Introduce variables for the dimensions of the matrices etc.
    #-------------------------------------------------------------------------
    
    scale         = max(abs(np.max(A_obj)),abs(np.min(A_obj)),abs(theta_0))
    A_obj         = A_obj / max(scale,1)
    theta_0       = theta_0 / max(scale,1)
    
    beta_obj      = np.zeros((1,1))
    beta_obj[0,0] = theta_0
    A_k           = sp.vstack((A_k,A_obj))
    beta_k        = np.vstack((beta_k,beta_obj))
    A_k           = sp.vstack((A_k,-A_obj))
    beta_k        = np.vstack((beta_k,-beta_obj))
    
    p_u, d        = A_u.shape
    p_k, d        = A_k.shape
    p             = p_u + p_k
    
    status        = 0
    
    #-------------------------------------------------------------------------
    # 1. Compute TS
    #-------------------------------------------------------------------------
    
    m         = gp.Model()
    x         = m.addMVar(d, lb=-GRB.INFINITY)
    
    m.addConstr(A_k @ x >= beta_k.reshape((p_k,)))
    
    Sig_inv = np.linalg.inv(np.diag(np.diag(Sigma)))
    Sig_inv = Sig_inv / np.max(np.abs(Sig_inv))
    
    obj_t   = m.addMVar(p_u, lb=-GRB.INFINITY)
    m.addConstr(A_u @ x - beta_u[0].reshape((p_u,)) == obj_t)
    m.setObjective(obj_t @ Sig_inv @ obj_t, GRB.MINIMIZE)
    m.setParam('OutputFlag',False)
    m.setParam('Method',2)
    m.setParam('Crossover',0)
    m.reset(0)
    m.optimize()
    if m.status != 2:
        m.setParam('Method',0)
        m.reset(0)
        m.optimize()
    if m.status != 2:
        m.setParam('Method',1)
        m.reset(0)
        m.optimize()    
    
    TS = n * m.objVal
    
    if m.status != 2:
        status += 1
    
    #-------------------------------------------------------------------------
    # 2. Compute TS^DR_b and TS^PR_b for each bootstrap sample
    #-------------------------------------------------------------------------
    
    TS_b = []
    m       = gp.Model()
    x       = m.addMVar(d, lb=-GRB.INFINITY)
    m.addConstr(A_k @ x >= beta_k.reshape((p_k,)))
    for j in range(1,B+1):
        
        TS_DR = n * ((np.transpose(beta_u[0] - beta_u[j]) @ Sig_inv) @ (beta_u[0] - beta_u[j]))
        
        obj_t   = m.addMVar(p_u, lb=-GRB.INFINITY)
        m.addConstr((beta_u[0] - beta_u[j]).reshape((p_u,)) + (1 / kap) * (A_u @ x - beta_u[0].reshape((p_u,))) == obj_t)
        m.setObjective(obj_t @ Sig_inv @ obj_t, GRB.MINIMIZE)
        m.setParam('OutputFlag',False)
        m.setParam('Method',2)
        m.setParam('Crossover',0)
        m.reset(0)
        m.optimize()
        if m.status != 2:
            m.setParam('Method',0)
            m.reset(0)
            m.optimize()
        if m.status != 2:
            m.setParam('Method',1)
            m.reset(0)
            m.optimize()  
        
        TS_PR   = n * m.objVal
        
        if m.status != 2:
            status += 1
            
        TS_b.append(min(TS_DR[0,0],TS_PR))
    
    #-------------------------------------------------------------------------
    # 3. Perform test
    #-------------------------------------------------------------------------
    
    pval = len([i for i in TS_b if i  >= TS]) / B   
    phi  = 1 * (alpha > pval)

    return phi, status

def npdemand(p_a,p_b,P_obs,share,g_a,g_b,g_ab,spec='NPB',K=3,grid_size=5,conf=0,n=0,Sigma='bs',share_b={},level=0.90,incr='auto'):
    
    ###########################################################################
    # npdemand: Python package to implement tools in Kamat and Norris (2022)
    ###########################################################################
    
    #--------------------------------------------------------------------------
    # Options
    #--------------------------------------------------------------------------
    
    #p_a       = numpy array of size J containing higher prices in price decrease. (NOT OPTIONAL)
    #p_b       = numpy array of size J containing lower prices in price decrease. (NOT OPTIONAL)
    #P_obs     = list of numpy arrays of size J containing the observed discrete variation. (NOT OPTIONAL)
    #share     = list of numpy arrays of size J containing the shares at each observed price. (NOT OPTIONAL)
    #g_a       = numpy array of size J containing the weight for demand at price p_a in parameter of interest. (NOT OPTIONAL)
    #g_b       = numpy array of size J containing the weight for demand at price p_b in parameter of interest. (NOT OPTIONAL)
    #g_ab      = numeric value corresponding to the weight for the average willing-to-pay for price decrease from p_b to p_a. (NOT OPTIONAL)
    #spec      = specification taking values in "NPB", "NPS", and "PS". (OPTIONAL, DEFAULT = "NPB")
    #K         = if spec = "PS", then degree of polynomial. (OPTIONAL, DEFAULT = 1)
    #p_lower   = numpy array of size J corresponding to lower value of price support. (OPTIONAL, DEFAULT = min across p_a,p_b,P_obs)
    #p_upper   = numpy array of size J corresponding to upper value of price support. (OPTIONAL, DEFAULT = max across p_a,p_b,P_obs)
    #grid_size = size of gequidistant grid between p_lower and p_upper at which parameter restrictions are evluated. (OPTIONAL, DEFAULT = 5)
    #conf      = if conf == 1 then construct confidence intervals using Bugni et al (2017) (OPTIONAL, DEFAULT = 0)
    #n         = sample size (NOT OPTIONAL if conf == 1)
    #Sigma     = Estimate of var/covaraince matrix of shares. (OPTIONAL, DEFAULT = Compute using bootstrap, i.e. using share and share_b)
    #share_b   = dictionary where each entry is a share computed in bootstrap draw (NOT OPTIONAL if conf == 1)
    #level     = Level for confidence interval (OPTIONAL, DEFAULT = 0.90)
    #incr      = increment to use in test inversion proecure for confidence interval construction (OPTIONAL, DEFAULT = construct using standard deviation of bootstrap bound estimates)
    
    #--------------------------------------------------------------------------
    # Error messages
    #--------------------------------------------------------------------------
    
    #Give error if spec is not allowed
    if spec not in ["NPB","NPS","PS"]:
        print("Error: Specification value is not in [''NPB'',''NPS'',''PS'']")
        return
    
    #Give error that if if conf == 1 then one must input bootstrap shares and sample
    if conf == 1:
        if share_b == []:
            print("Error: Need to input shares computed in bootstrap sample in option ``share_b'' ")
            return
        if n == 0:
            print("Error: Need to input sample size in option ``n'' ")
            return
        
    #Give error if it is not price decrease then g_ab has to be set to 0
    if sum([1*(p_a[i] < p_b[i]) for i in range(len(p_a))]) > 0: 
        print("Warning: p_b is not smaller than p_a. g_ab set to 0.")
        g_ab = 0
    
    #--------------------------------------------------------------------------
    # Store number of alternatives and lower and upper bounds of prices
    #--------------------------------------------------------------------------
    
    J_n     = len(p_a)           #Number of alternatives
    J       = range(J_n)         #Set of alternatives
    P_dis   = [p_b,p_a] + P_obs  #List of all discrete prices in data and parameter
    
    #if no bounds on prices then take min and mac of discrete prices
    p_lower = np.zeros(len(p_a))
    p_upper = np.zeros(len(p_a))
    for i in range(len(p_a)):
        p_lower[i] = min([p[i] for p in P_dis])
        p_upper[i] = max([p[i] for p in P_dis])
        
    if spec == "NPB":
        
        #--------------------------------------------------------------------------
        # Compute Partition for nonparametric case
        #--------------------------------------------------------------------------
        
        #Compute Delta, i.e. ordered values of p_b[j] - p_a[j]
        
        Delta   = list(set([abs(p_a[j] - p_b[j]) for j in J])) 
        Delta.sort()
        
        #Compute partition V
        
        if g_ab != 0:
            A       = list(set(Delta + [max(min(p_a[j],P_obs[l][j]),p_b[j]) - p_b[j] for j in J for l in range(len(P_obs))]))
            A.sort()
        
        W       = {}
        if g_ab != 0:
            #Sets relevant for WTP
            for l in range(len(A)-1):
                for j in J:
                    W[0,j,l,0] = min(p_a[j],p_b[j] + A[l])
                    W[1,j,l,0] = min(p_a[j],p_b[j] + A[l+1])
        #Sets relevant for p_a,p_b,P_obs
        for l in range(len(P_dis)):
            for j in J:
                W[0,j,l,1] = P_dis[l][j]
                W[1,j,l,1] = P_dis[l][j]
                
        #----------------------------------------------------------------------
        # Compute matrices A, b, and c for the optimization program of 
        # max/min A_obj'x subject to  A_u x = b_u and A_k x >= b_k
        #----------------------------------------------------------------------
        
        A_u, A_k, A_sm,  b_u, b_k, A_obj = {}, {}, {}, {}, {}, {}
        d, id_x, id_x_k, id_x_u          = -1, {}, {}, {}  #Number of "true" variables, and dictionaries corresponds to variable and shape and data restrictions where variable plays a role
        d_sm, id_sm, id_sm_v             = -1, {}, {} #Number of "slack" variables introduced to capture monotonicity, and dicitonaries corresponding to variable and which restriction they play a role
        p_u, p_k                         = -1, -1  #Number of data and shape restrictions
        
        #Define variable
        if g_ab != 0:
            for l in range(len(A)-1):
                for j in J:
                    d                                 += 1
                    id_x[j,l,0], id_x_k[d], id_x_u[d]  = d, [], []
                    A_obj[d]                           = 0
        for l in range(len(P_dis)):
            for j in J:
                d                                 += 1
                id_x[j,l,1], id_x_k[d], id_x_u[d]  = d, [], []
                A_obj[d]                           = 0
        
        # Add constraint: variables sum to 1
        if g_ab != 0:
            for l in range(len(A)-1):
                p_k       += 1
                b_k[p_k]   = 1
                for j in J:
                    i_x  = id_x[j,l,0]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = 1
                p_k       += 1
                b_k[p_k]   = -1
                for j in J:
                    i_x  = id_x[j,l,0]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = -1
        for l in range(len(P_dis)):
            p_k       += 1
            b_k[p_k]   = 1
            for j in J:
                i_x  = id_x[j,l,1]
                id_x_k[i_x].append(p_k)
                A_k[p_k,i_x] = 1
            p_k       += 1
            b_k[p_k]   = -1
            for j in J:
                i_x  = id_x[j,l,1]
                id_x_k[i_x].append(p_k)
                A_k[p_k,i_x] = -1
        
        #Non-negativity constraint    
        if g_ab != 0:
            for l in range(len(A)-1):
                for j in J:
                    p_k       += 1
                    b_k[p_k]   = 0
                    i_x  = id_x[j,l,0]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = 1
        for l in range(len(P_dis)):
            for j in J:
                p_k       += 1
                b_k[p_k]   = 0
                i_x  = id_x[j,l,1]
                id_x_k[i_x].append(p_k)
                A_k[p_k,i_x] = 1
                
        
        #Add constraint: monotonicity
        if g_ab != 0:
            if len(A) > 1:
                #Monotonicity between sets in WTP sets
                for l in range(len(A)-2):
                    J_temp     = [j for j in J if W[1,j,l+1,0] == W[1,j,l,0] and W[0,j,l+1,0] == W[0,j,l,0]]  
                    for j in J_temp:
                        p_k         += 1
                        b_k[p_k]     = 0
                        
                        i_x          = id_x[j,l+1,0]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = 1
                        
                        i_x          = id_x[j,l,0]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = -1
                    
                #Monotonicity between sets in WTP sets and [p_a,p_b]
                J_temp     = [j for j in J if W[1,j,0,0] == W[1,j,0,1] and W[0,j,0,0] == W[0,j,0,1]]  
                for j in J_temp:
                    p_k         += 1
                    b_k[p_k]     = 0
                    
                    i_x          = id_x[j,0,0]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = 1
                    
                    i_x          = id_x[j,0,1]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = -1
                
                J_temp = [j for j in J if W[1,j,1,1] == W[1,j,len(A)-2,0] and W[0,j,1,1] == W[0,j,len(A)-2,0]]  
                for j in J_temp:
                    p_k         += 1
                    b_k[p_k]     = 0
                    
                    i_x          = id_x[j,1,1]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = 1
                    
                    i_x          = id_x[j,len(A)-2,0]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = -1
        
        #Monotonicity between sets in P_obs and [p_a,p_b] + WTP sets
        var_c = {}
        
        if g_ab != 0:
            t_range = [0,1]
        else:
            t_range = [1]
        
        for t in t_range:
            if t == 0:
                T_t = list(range(len(A)-1))
            elif t == 1:
                T_t = [0,1]
            for l in T_t:
                for k in range(2,len(P_dis)):
                    J0     = [i for i in J if W[1,i,l,t] > W[1,i,k,1] or W[0,i,l,t] > W[0,i,k,1]]
                    J1     = [i for i in J if W[0,i,l,t] < W[0,i,k,1] or W[1,i,l,t] < W[1,i,k,1]]
                    J2     = list(set(J) - set(J0) - set(J1))             
                    if J0 == []:
                       for i in J2:
                           p_k         += 1
                           b_k[p_k]     = 0
                           
                           i_x          = id_x[i,k,1]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x] = 1
                           
                           i_x          = id_x[i,l,t]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x] = -1
                    if J1 == []:
                       for i in J2:
                           p_k       += 1
                           b_k[p_k]   = 0
                           
                           i_x          = id_x[i,l,t]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x] = 1
                           
                           i_x          = id_x[i,k,1]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x] = -1
                    if J0 != [] and J1 != []:
                       #var_c[0,l,k,t]  = m.addVars(J,lb=0)
                       for j in J:
                           d_sm               += 1
                           id_sm_v[0,l,k,t,j]  = d_sm 
                           id_sm[d_sm]         = []
                       #var_c[1,l,k,t]  = m.addVars(J,lb=0)
                       for j in J:
                           d_sm               += 1
                           id_sm_v[1,l,k,t,j]  = d_sm 
                           id_sm[d_sm]         = []
                       
                       for j in J:
                           p_k             += 1
                           b_k[p_k]         = 0
                           i_sm           = id_sm_v[0,l,k,t,j]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm] = 1
                       for j in J:
                           p_k             += 1
                           b_k[p_k]         = 0
                           i_sm           = id_sm_v[1,l,k,t,j]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm] = 1  
                           
                       p_k             += 1
                       b_k[p_k]         = 1
                       for j in J:
                           i_sm           = id_sm_v[0,l,k,t,j]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm] = 1
                       p_k             += 1
                       b_k[p_k]         = -1
                       for j in J:
                           i_sm           = id_sm_v[0,l,k,t,j]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm] = -1
                       
                       p_k             += 1
                       b_k[p_k]         = 1
                       for j in J:
                           i_sm           = id_sm_v[1,l,k,t,j]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm] = 1
                       p_k             += 1
                       b_k[p_k]         = -1
                       for j in J:
                           i_sm           = id_sm_v[1,l,k,t,j]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm] = -1
                       
                       for i in J0+J2:
                           p_k             += 1
                           b_k[p_k]         = 0
                           
                           i_x              = id_x[i,l,t]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x]     = -1
                           
                           i_sm             = id_sm_v[0,l,k,t,i]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm]   = 1
                           
                       for i in J1+J2:
                           p_k             += 1
                           b_k[p_k]         = 0
                           
                           i_x              = id_x[i,k,1]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x]     = -1
                           
                           i_sm             = id_sm_v[0,l,k,t,i]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm]   = 1
                           
                       for i in J1+J2:
                           p_k             += 1
                           b_k[p_k]         = 0
                           
                           i_x              = id_x[i,l,t]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x]     = 1
                           
                           i_sm             = id_sm_v[1,l,k,t,i]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm]   = -1
                           
                       for i in J0+J2:
                           p_k             += 1
                           b_k[p_k]         = 0
                           
                           i_x              = id_x[i,k,1]
                           id_x_k[i_x].append(p_k)
                           A_k[p_k,i_x]     = 1
                           
                           i_sm             = id_sm_v[1,l,k,t,i]
                           id_sm[i_sm].append(p_k)
                           A_sm[p_k,i_sm]   = -1
               
        #Add constraint: data  
        for l in range(2,len(P_dis)):
            for j in J:
                p_u             += 1
                b_u[p_u]         = share[l-2][j]
                
                i_x              = id_x[j,l,1]
                id_x_u[i_x].append(p_u)
                A_u[p_u,i_x]     = 1
                
                
        #Define parameter of interes
        if g_ab != 0:
            for l in range(len(Delta)-1):
                for k in range(len(A)-1):
                    if all(W[1,j,k,0] <= min(p_a[j],p_b[j] + Delta[l+1]) and W[0,j,k,0] >= min(p_a[j],p_b[j] + Delta[l]) for j in J):
                        for i in [i_t for i_t in J if p_a[i_t] - p_b[i_t] > Delta[l]]:
                            i_x         = id_x[i,k,0]
                            A_obj[i_x] += g_ab * (A[k+1] - A[k])
            for j in J:
                i_x           = id_x[j,1,1]
                A_obj[i_x]   += g_a[j]
                i_x           = id_x[j,0,1]
                A_obj[i_x]   += g_b[j]
        else:
            for j in J:
                i_x           = id_x[j,1,1]
                A_obj[i_x]   += g_a[j]
                i_x           = id_x[j,0,1]
                A_obj[i_x]   += g_b[j]
            
                
    elif spec == "NPS":
        
        #Compute Delta, i.e. ordered values of p_b[j] - p_a[j]
        
        Delta   = list(set([abs(p_a[j] - p_b[j]) for j in J])) 
        Delta.sort()
        
        #Compute partition V
        
        if g_ab != 0:
            A       = list(set(Delta + [max(min(p_a[j],P_obs[l][j]),p_b[j]) - p_b[j] for j in J for l in range(len(P_obs))]))
            A.sort()
        
        #Sets relevant for WTP
        W, ind_W, ind_obs = {}, {}, {}
        for j in J:
            if g_ab != 0:
                A_temp   = list(set([min(p_a[j],p_b[j] + a) for a in A] + [p_lower[j],p_upper[j]] + [p[j] for p in P_dis]))
            else:
                A_temp   = list(set([p_lower[j],p_upper[j]] + [p[j] for p in P_dis]))
            A_temp.sort()
            A_temp_d = list(set([p[j] for p in P_dis]))
            A_temp_d.sort()
            t = -1
            for k in range(len(A_temp_d)):
                if A_temp_d[k] == A_temp[0]:
                    t                                += 1
                    W[0,j,t], W[1,j,t], ind_obs[k,j]  = A_temp[0], A_temp[0], t
            for l in range(len(A_temp)-1):
                t                  += 1
                W[0,j,t], W[1,j,t]  = A_temp[l], A_temp[l+1]
                for k in range(len(A_temp_d)):
                    if A_temp_d[k] == A_temp[l+1]:
                        t                                += 1
                        W[0,j,t], W[1,j,t], ind_obs[k,j]  = A_temp[l+1], A_temp[l+1], t
            ind_W[j] = t + 1
        #Sets relevant for p_a,p_b,P_obs    
        ind_dis = {}
        for l in range(len(P_dis)):
            for j in J:
                A_temp_d = list(set([p[j] for p in P_dis]))
                A_temp_d.sort()
                for k in range(len(A_temp_d)):
                    if P_dis[l][j] == A_temp_d[k]:
                        ind_dis[l,j] = k
        
        #----------------------------------------------------------------------
        # Compute matrices A, b, and c for the optimization program of 
        # max/min A_obj'x subject to  A_u x = b_u and A_k x >= b_k
        #----------------------------------------------------------------------                
        
        A_u, A_k, A_sm,  b_u, b_k, A_obj = {}, {}, {}, {}, {}, {}
        d, id_x, id_x_k, id_x_u          = -1, {}, {}, {}  #Number of "true" variables, and dictionaries corresponds to variable and shape and data restrictions where variable plays a role
        d_sm, id_sm, id_sm_v             = -1, {}, {} #Number of "slack" variables introduced to capture monotonicity, and dicitonaries corresponding to variable and which restriction they play a role
        p_u, p_k                         = -1, -1  #Number of data and shape restrictions
        
        #Define parameter
        for j in J:
            for i in J:
                for l in range(ind_W[i]):
                    for pos in [1]:
                        d                                     += 1
                        id_x[j,i,l,pos], id_x_k[d], id_x_u[d]  = d, [], []
                        A_obj[d]                               = 0  
        
        # Add constraint: non-negative
        for j in J:
            for l in range(ind_W[j]):
                p_k       += 1
                b_k[p_k]   = 0
                for pos in [1]:
                    i_x = id_x[j,j,l,pos]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = pos * 1
                for i in [i_t for i_t in J if i_t != j]:
                    for pos in [1]:
                        i_x = id_x[j,i,0,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = pos * 1
        
        # Add constraint: variables sum to 1
        p_k       += 1
        b_k[p_k]   = 1
        i_x        = -1
        for j in J:
            for i in J:
                for pos in [1]:
                    i_x = id_x[j,i,0,pos]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = pos * 1
        p_k       += 1
        b_k[p_k]   = -1
        i_x        = -1
        for j in J:
            for i in J:
                for pos in [1]:
                    i_x = id_x[j,i,0,pos]
                    id_x_k[i_x].append(p_k)
                    A_k[p_k,i_x] = -pos * 1
                        
        for i in J:
            for l in range(ind_W[i]-1):
                p_k       += 1
                b_k[p_k]   = 0
                for j in J:
                    for pos in [1]:
                        i_x = id_x[j,i,l+1,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = pos * 1
                    
                        i_x = id_x[j,i,l,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = -pos * 1
                        
                p_k       += 1
                b_k[p_k]   = 0
                for j in J:
                    for pos in [1]:
                        i_x = id_x[j,i,l+1,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = -pos * 1
                    
                        i_x = id_x[j,i,l,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = pos * 1
            
        #Add constraint: monotonicity
        for j in J:
            for i in [i_t for i_t in J if i_t != j]:
                for l in range(ind_W[i]-1):
                    p_k       += 1
                    b_k[p_k]   = 0
                    for pos in [1]:
                        i_x = id_x[j,i,l+1,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = pos * 1
                    
                        i_x = id_x[j,i,l,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = -pos * 1
               
        #Add constraint: data  
        for l in range(2,len(P_dis)):
            for j in J:
                p_u       += 1
                b_u[p_u]   = share[l-2][j]
                for i in J:
                    for pos in [1]:
                        i_x = id_x[j,i,ind_obs[ind_dis[l,i],i],pos]
                        id_x_u[i_x].append(p_u)
                        A_u[p_u,i_x] = pos * 1
                
        #Define parameter of interest
        if g_ab != 0:
            #AS = AB - AC
            for l in range(len(Delta)-1):
                for j in [j_t for j_t in J if p_a[j_t] - p_b[j_t] > Delta[l]]:
                    for i in J:
                        if p_a[i] - p_b[i] > Delta[l]:
                            for k in range(ind_W[i]):
                                if W[1,i,k] <= min(p_a[i],p_b[i] + Delta[l+1]) and W[0,i,k] >= min(p_a[i],p_b[i] + Delta[l]):
                                    for pos in [1]:
                                        i_x = id_x[j,i,k,pos]
                                        A_obj[i_x] += g_ab * pos * (W[1,i,k] - W[0,i,k])
                        else:
                            #Obj += (Delta[l+1] - Delta[l]) * beta[j,i,ind_obs[ind_dis[0,i],i]] 
                            for pos in [1]:
                                i_x = id_x[j,i,ind_obs[ind_dis[1,i],i],pos]
                                A_obj[i_x] += g_ab *  pos * (Delta[l+1] - Delta[l])
            for j in J:
                for i in J:
                    for pos in [1]:
                        i_x = id_x[j,i,ind_obs[ind_dis[1,i],i],pos]
                        A_obj[i_x] += pos * g_a[j]
                        i_x = id_x[j,i,ind_obs[ind_dis[0,i],i],pos]
                        A_obj[i_x] += pos * g_b[j]
        else:
            for j in J:
                for i in J:
                    for pos in [1]:
                        i_x = id_x[j,i,ind_obs[ind_dis[1,i],i],pos]
                        A_obj[i_x] += pos * g_a[j]
                        i_x = id_x[j,i,ind_obs[ind_dis[0,i],i],pos]
                        A_obj[i_x] += pos * g_b[j]
                    
    elif spec == "PS":
        
        Delta   = list(set([abs(p_a[j] - p_b[j]) for j in J])) 
        Delta.sort()
        
        #Price grid over which constraints are imposed
        P_grid, L_grid, P_l = {}, {}, {}
        for j in J:
            P_l[j]    = (p_upper[j] > 0) * p_upper[j] + (p_upper[j] == 0) * 1 
            P_grid[j] = list(set([(l / grid_size) for l in range(grid_size+1)] + [P_dis[l][j] / P_l[j] for l in range(2,len(P_dis))]))
            P_grid[j].sort()
            L_grid[j] = len(P_grid[j])
        
        A_u, A_k, b_u, b_k, A_obj = {}, {}, {}, {}, {}
        d, id_x, id_x_k, id_x_u   = -1, {}, {}, {}  #Number of "true" variables, and dictionaries corresponds to variable and shape and data restrictions where variable plays a role
        p_u, p_k                  = -1, -1  #Number of data and shape restrictions
        
        #Define variable
        for j in J:
            for i in J:
                for k in range(K+1):  
                    for pos in [1]:
                        d                                     += 1
                        id_x[j,i,k,pos], id_x_k[d], id_x_u[d]  = d, [], []
                        A_obj[d]                               = 0
        #beta = m.addVars(beta,lb=-GRB.INFINITY) 
        
        # Add constraint: non-negativity
        for j in J:
            for l in range(L_grid[j]):
                p_k       += 1
                b_k[p_k]   = 0
                for k in range(K+1):
                    for pos in [1]:
                        i_x          = id_x[j,j,k,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = pos * (P_grid[j][l] ** k)
                for i in [i_t for i_t in J if i_t != j]:
                    for k in range(K+1):
                        for pos in [1]:
                            i_x          = id_x[j,i,k,pos]
                            id_x_k[i_x].append(p_k)
                            A_k[p_k,i_x] = pos * (P_grid[i][0] ** k)
                
        # Add constraint: variables sum to 1    
        p_k       += 1
        b_k[p_k]   = 1
        for j in J:
            for i in J:
                for k in range(K+1):
                    for pos in [1]:
                        i_x          = id_x[j,i,k,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = pos * (P_grid[i][0] ** k)
        p_k       += 1
        b_k[p_k]   = -1
        for j in J:
            for i in J:
                for k in range(K+1):
                    for pos in [1]:
                        i_x          = id_x[j,i,k,pos]
                        id_x_k[i_x].append(p_k)
                        A_k[p_k,i_x] = -pos * (P_grid[i][0] ** k)
        
        for i in J:
            for l in range(L_grid[i] - 1):
                p_k       += 1
                b_k[p_k]   = 0
                for j in J:
                    for k in range(K+1):
                        for pos in [1]:
                            i_x          = id_x[j,i,k,pos]
                            id_x_k[i_x].append(p_k)
                            A_k[p_k,i_x] = pos * ((P_grid[i][l+1] ** k) - (P_grid[i][l] ** k))
                p_k       += 1
                b_k[p_k]   = 0
                for j in J:
                    for k in range(K+1):
                        for pos in [1]:
                            i_x          = id_x[j,i,k,pos]
                            id_x_k[i_x].append(p_k)
                            A_k[p_k,i_x] = -pos * ((P_grid[i][l+1] ** k) - (P_grid[i][l] ** k))
        
        #Add constraint: monotonicity  
        for j in J:
            for i in [i_t for i_t in J if i_t != j]:
                for l in range(L_grid[i] - 1):
                    p_k       += 1
                    b_k[p_k]   = 0
                    for k in range(K+1):
                        for pos in [1]:
                            i_x          = id_x[j,i,k,pos]
                            id_x_k[i_x].append(p_k)
                            A_k[p_k,i_x] = pos * ((P_grid[i][l+1] ** k) - (P_grid[i][l] ** k))
        
        #Add contrasint: data
        for l in range(2,len(P_dis)):    
            for j in J:
                p_u       += 1
                b_u[p_u]   = share[l-2][j]
                for i in J:
                    for k in range(K+1):
                        for pos in [1]:
                            i_x          = id_x[j,i,k,pos]
                            id_x_u[i_x].append(p_u)
                            A_u[p_u,i_x] = pos * ((P_dis[l][i] / P_l[i]) ** k)
        
        #Define parameter
        if g_ab != 0:
            for l in range(len(Delta)-1):
                for j in [j_t for j_t in J if p_a[j_t] - p_b[j_t] > Delta[l]]:
                    for i in J:
                        if p_a[i] - p_b[i] > Delta[l]:
                            for k in range(K+1):
                                for pos in [1]:
                                    i_x         = id_x[j,i,k,pos]
                                    A_obj[i_x] += g_ab * pos * ((((p_b[i] + Delta[l+1]) / P_l[i]) ** (k+1)) - (((p_b[i] + Delta[l]) / P_l[i]) ** (k+1))) * P_l[i] / (k+1)
                        else:
                            for k in range(K+1):
                                for pos in [1]:
                                    i_x         = id_x[j,i,k,pos]
                                    A_obj[i_x] += g_ab * pos * ((p_a[i] / P_l[i]) ** k) * (Delta[l+1] - Delta[l])
            for j in J:
                for k in range(K+1):
                    for i in J:
                        for pos in [1]:
                            i_x         = id_x[j,i,k,pos]
                            A_obj[i_x] += pos * (g_b[j] * ((p_b[i] / P_l[i]) ** k) + g_a[j] * ((p_a[i] / P_l[i]) ** k))
        else:
            for j in J:
                for k in range(K+1):
                    for i in J:
                        for pos in [1]:
                            i_x         = id_x[j,i,k,pos]
                            A_obj[i_x] += pos * (g_b[j] * ((p_b[i] / P_l[i]) ** k) + g_a[j] * ((p_a[i] / P_l[i]) ** k))
            
    #--------------------------------------------------------------------------
    # Store A_u, A_k, A_obj, b_u, and b_k in sparse format
    #--------------------------------------------------------------------------
    
    if spec == "NPB":
        d_fin = d + d_sm + 2
    else:
        d_fin = d + 1
    
    beta_k = np.zeros((p_k+1,1))
    for i in range(p_k+1):
        beta_k[i,0] = b_k[i]
        
    t                            = -1
    val_temp, row_temp, col_temp = {}, {}, {}
    for j in range(d+1):
        for i in id_x_k[j]:
            t                                   += 1
            val_temp[t], row_temp[t],col_temp[t] = A_k[i,j], i, j
    if spec == "NPB":        
        for j in range(d_sm+1):
            for i in id_sm[j]:
                t                                   += 1
                val_temp[t], row_temp[t],col_temp[t] = A_sm[i,j], i, j + d + 1
    
    val, row, col = np.zeros(t+1), np.zeros(t+1), np.zeros(t+1)
    for i in range(t+1):
        val[i], row[i], col[i] = val_temp[i], row_temp[i], col_temp[i]
    A_k = sp.csr_matrix((val, (row, col)), shape=(p_k+1,d_fin))
    
    t                            = -1
    val_temp, row_temp, col_temp = {}, {}, {}
    for j in range(d+1):
        for i in id_x_u[j]:
            t                                    += 1
            val_temp[t], row_temp[t], col_temp[t] = A_u[i,j], i, j  
    val, row, col = np.zeros(t+1), np.zeros(t+1), np.zeros(t+1)
    for i in range(t+1):
        val[i], row[i], col[i] = val_temp[i], row_temp[i], col_temp[i]
    A_u = sp.csr_matrix((val, (row, col)), shape=(p_u+1,d_fin))
            
    A_obj_t  = np.zeros((1,d_fin))
    for j in range(d+1):
        A_obj_t[0,j] = A_obj[j]
    A_obj    = A_obj_t
    
    #--------------------------------------------------------------------------
    # Obtain estimated bounds
    #--------------------------------------------------------------------------
    
    beta_u   = {}
    data_dim = p_u + 1
    B        = len(share_b)  #Number of bootstrap draws
    for i in range(B+1):
        beta_u[i]   = np.zeros(((data_dim,1)))
        p_u         = -1
        if i == 0:
            for l in range(2,len(P_dis)):    
                for j in J:
                    p_u             += 1
                    beta_u[i][p_u]   = share[l-2][j]
        elif i > 0:
            for l in range(2,len(P_dis)):    
                for j in J:
                    p_u             += 1
                    beta_u[i][p_u]   = share_b[i][l-2][j]
    
    #--------------------------------------------------------------------------
    # Obtain estimated bounds
    #--------------------------------------------------------------------------
    
    est             = {}
    opt_status      = 0
    try:
        temp            = est_npdemand(A_u,A_k,beta_u[0],beta_k,A_obj,0)
    except:
        temp            = est_npdemand(A_u,A_k,beta_u[0],beta_k,A_obj,2)
    est[0], est[1]  = temp[0], temp[1]
    opt_status     += temp[2]
    
    if conf == 1:
        
        ci                 = {}
        
        #Compute worst case bounds
        bnd                = {}
        temp               = est_npdemand(A_u,A_k,beta_u[0],beta_k,A_obj,1)
        bnd[0], bnd[1]     = temp[0], temp[1]
        
        #Compute incr size if not sepcific
        if incr == 'auto':
            std_l, std_u = [], []
            for i in range(1,B+1):
                try:
                    temp            = est_npdemand(A_u,A_k,beta_u[i],beta_k,A_obj,0)
                except:
                    temp            = est_npdemand(A_u,A_k,beta_u[i],beta_k,A_obj,2)
                std_l.append((temp[0] - est[0]) ** 2)
                std_u.append((temp[1] - est[1]) ** 2)
            incr = (np.sqrt(sum(std_l) / B) + np.sqrt(sum(std_l) / B)) / 2
        
        #Compute Sigma if not specificed   
        if Sigma == 'bs':
            Sigma = n * np.diag(sum(((beta_u[0].reshape((len(beta_u[0]),)) - sum(beta_u[i].reshape((len(beta_u[i]),)) for i in range(1,B+1)) / B) ** 2) for j in range(1,B+1)) / B)
        Sigma = np.diag(np.diag(Sigma)) + max(0.012 - np.linalg.det(Sigma),0) * np.eye(len(Sigma))
            
        thresh = incr / 10
        kap    = np.sqrt(math.log(n))
        for bound in [0,1]:
            if bound == 0:
                theta_0 = est[0] 
                phi     = 0
                while phi == 0 and theta_0 > bnd[0]:
                    theta_0          = max(theta_0 - incr,bnd[0])
                    phi, stat        = test_npdemand(A_u,A_k,beta_u,beta_k,A_obj,Sigma,B,theta_0,n,kap,level)
                    opt_status      += stat
                if theta_0 == bnd[0] and phi == 0:
                    a_l = bnd[0]
                    a_u = bnd[0]
                else:
                    a_l = max(theta_0,bnd[0])
                    a_u = min(theta_0 + incr, est[0])
                    while abs(a_l - a_u) >= thresh:
                        theta_0          = (a_l + a_u) / 2
                        phi, stat        = test_npdemand(A_u,A_k,beta_u,beta_k,A_obj,Sigma,B,theta_0,n,kap,level)
                        opt_status      += stat
                        if phi == 0:
                            a_u = theta_0
                        elif phi == 1:
                            a_l = theta_0
            elif bound == 1:
                theta_0 = est[1] 
                phi     = 0
                while phi == 0 and theta_0 < bnd[1]:  #Stop when we first reject
                    theta_0          = min(theta_0 + incr,bnd[1])
                    phi, stat        = test_npdemand(A_u,A_k,beta_u,beta_k,A_obj,Sigma,B,theta_0,n,kap,level)
                    opt_status      += stat
                if theta_0 == bnd[1] and phi == 0:
                    a_l = bnd[1]
                    a_u = bnd[1]
                else:
                    a_l = max(theta_0 - incr,est[1])
                    a_u = min(theta_0, bnd[1])
                    while abs(a_l - a_u) >= thresh:
                        theta_0          = (a_l + a_u) / 2
                        phi, stat        = test_npdemand(A_u,A_k,beta_u,beta_k,A_obj,Sigma,B,theta_0,n,kap,level)
                        opt_status      += stat
                        if phi == 0:
                            a_l = theta_0
                        elif phi == 1: 
                            a_u = theta_0
            ci[bound] = (a_l + a_u) / 2    

    #Print Table with results
    if spec == "NPB":
        print("---------------------------------------------------")
        print("Specification: Nonparametric nonseparable demand")
        print("---------------------------------------------------")
    elif spec == "NPS":
        print("------------------------------------------------")
        print("Specification: Nonparametric separable demand")
        print("------------------------------------------------")
    elif spec == "PS":
        print("-----------------------------------------------------------")
        print("Parametric separable demand with polynomial degree of", K)
        print("-----------------------------------------------------------")
    
    print("Estimated Bounds")
    print("------------------------------------")
    print("Lower bound:", est[0])
    print("Upper bound:", est[1])
    print("------------------------------------")
    if conf == 1:
        print(100*level,"% Confidence Intervals")
        print("------------------------------------")
        print("Lower value:", ci[0])
        print("Upper value:", ci[1])
        print("------------------------------------")
    if opt_status != 0:
        print("Warning: Some of the programs faced numerical issues.")
        print("------------------------------------")
        
    if conf == 1:
        output               = np.zeros(4)
        output[0], output[1] = est[0], est[1]
        output[2], output[3] = ci[0], ci[1]
    else:
        output               = np.zeros(2)
        output[0], output[1] = est[0], est[1]
    
    return output
    