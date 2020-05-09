import pandas as pd
import numpy as np
import random
from scipy.stats import beta
from scipy.optimize import newton

def gamma(x,a0,b0,a1,b1,alpha):
    return 1/((beta.pdf(x, a0, b0, loc=0, scale=1)/beta.pdf(x, a1, b1, loc=0, scale=1))*(1/alpha -1 )+1)


def gamma_1(x,a0,b0,a1,b1,alpha):
    return (beta.pdf(x, a0, b0, loc=0, scale=1)/beta.pdf(x, a1, b1, loc=0, scale=1))*(1/alpha -1 )+1

def P_fair(x,a0,b0,a1,b1,alpha):
    return (beta.pdf(x, a0, b0, loc=0, scale=1) * (1-alpha) + beta.pdf(x, a1, b1, loc=0, scale=1) * alpha)

def balanced_eqn_eqopt(x, alpha_aa,alpha_c, 
                       paa, pc,
                       a_aa0, b_aa0,a_aa1, b_aa1,
                       a_c0, b_c0,a_c1, b_c1):
    return gamma_1(x,a_aa0, b_aa0,a_aa1, b_aa1,alpha_aa) * paa *alpha_aa- 2*(paa * alpha_aa + pc * alpha_c) + pc *alpha_c* gamma_1(x,a_c0, b_c0,a_c1, b_c1,alpha_c) 


def balanced_eqn_un(x, a0,b0,a1,b1,alpha):
    return gamma_1(x,a0,b0,a1,b1,alpha) - 2

def get_policy_eqopt(alpha_aa,alpha_c, 
                     paa, pc,
                     a_aa0, b_aa0,a_aa1, b_aa1,
                     a_c0, b_c0, a_c1, b_c1):
    root=[]
    for i in np.arange(0.01,0.99,0.01):
        try:
            root.append(newton(balanced_eqn_eqopt, 
                               x0 = i, 
                               maxiter=50, 
                               args = (alpha_aa,alpha_c, 
                                       paa, pc,
                                       a_aa0, b_aa0,a_aa1, b_aa1,
                                       a_c0, b_c0,a_c1, b_c1)))
        except(RuntimeError):
            pass

    root = [float(format(round(r, 4))) for r in root]

    return np.unique(root)[0],np.unique(root)[0],len(np.unique(root)),np.unique(root)[0]

def get_policy_un(alpha_aa,alpha_c,
                  paa, pc,
                  a_aa0, b_aa0,a_aa1, b_aa1,
                  a_c0, b_c0, a_c1, b_c1):
    root_aa=[]
    root_c=[]
    for i in np.arange(0.01,0.99,0.01):
        try:
            root_aa.append(newton(balanced_eqn_un, 
                               x0 = i, 
                               maxiter=50, 
                               args = (a_aa0, b_aa0,a_aa1, b_aa1, alpha_aa)))
        except(RuntimeError):
            pass

        try:
            root_c.append(newton(balanced_eqn_un, 
                               x0 = i, 
                               maxiter=50, 
                               args = (a_c0, b_c0,a_c1, b_c1, alpha_c)))
        except(RuntimeError):
            pass

    root_aa = [float(format(round(r, 4))) for r in root_aa]
    root_c = [float(format(round(r, 4))) for r in root_c]    

    return np.unique(root_aa)[0],np.unique(root_c)[0],len(np.unique(root_aa)),len(np.unique(root_aa))


def eva_policy(theta_aa,theta_c, 
               a_aa0, b_aa0,a_aa1, b_aa1,
               a_c0, b_c0, a_c1, b_c1):
    tpr = []
    fpr = []

    tpr.append(1-beta.cdf(theta_aa, a_aa1, b_aa1)) 
    tpr.append(1-beta.cdf(theta_c, a_c1, b_c1))

    fpr.append(1-beta.cdf(theta_aa, a_aa0, b_aa0))
    fpr.append(1-beta.cdf(theta_c, a_c0, b_c0))

    return tpr,fpr

def eva_classifier(alpha_aa,alpha_c,get_policy,
                   paa, pc,
                   a_aa0, b_aa0,a_aa1, b_aa1,
                   a_c0, b_c0, a_c1, b_c1):
    
    theta_aa, theta_c, _ , _ = get_policy(alpha_aa,alpha_c, paa, pc, a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)
    return eva_policy(theta_aa,theta_c, a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)

def update(alpha,tpr,fpr,T, group):
    g0 = T[0,0,group]*(1-fpr[group]) + T[0,1,group]*fpr[group]
    g1 = T[1,0,group]*(1-tpr[group]) + T[1,1,group]*tpr[group]
    return alpha*g1 + (1-alpha)*g0

def balance_diff(alpha,tpr,fpr,T, group):
    g0 = T[0,0,group]*(1-fpr[group]) + T[0,1,group]*fpr[group]
    g1 = T[1,0,group]*(1-tpr[group]) + T[1,1,group]*tpr[group]    
    return g0 + (g1-g0-1)*alpha 
