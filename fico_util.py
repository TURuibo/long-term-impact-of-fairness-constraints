import pandas as pd
import numpy as np
import random
from scipy.stats import beta
from scipy.optimize import newton
from pynverse import inversefunc


def gamma(x,a0,b0,a1,b1,alpha):
    return 1/((beta.pdf(x, a0, b0, loc=0, scale=1)/beta.pdf(x, a1, b1, loc=0, scale=1))*(1/alpha -1 )+1)


def gamma_1(x,a0,b0,a1,b1,alpha):
    return (beta.pdf(x, a0, b0, loc=0, scale=1)/beta.pdf(x, a1, b1, loc=0, scale=1))*(1/alpha -1 )+1

def P_fair(x,a0,b0,a1,b1,alpha):
    return (beta.pdf(x, a0, b0, loc=0, scale=1) * (1-alpha) + beta.pdf(x, a1, b1, loc=0, scale=1) * alpha)

# DP
# alpha_aa * beta.cdf(x, a_aa1, b_aa1)+(1-alpha_aa)*beta.cdf(x, a_aa0, b_aa0) =
# alpha_c * beta.cdf(fn(x), a_c1, b_c1)+(1-alpha_c)*beta.cdf(fn(x), a_c0, b_c0)

def P_evidence(x,alpha,a0,b0,a1,b1):
    return alpha * beta.pdf(x, a1, b1)+(1-alpha)*beta.pdf(x, a0, b0)

def f_dp(x,alpha_aa,alpha_c,a_aa0,b_aa0,a_c0,b_c0,a_aa1,b_aa1,a_c1,b_c1):
  	f_c = (lambda x_c: alpha_c * beta.cdf(x_c, a_c1, b_c1,0,1)+(1-alpha_c)*beta.cdf(x_c, a_c0, b_c0,0,1))
  	inv_f_c = inversefunc(f_c, domain=[0,1], open_domain=[False,False])
  	return float(inv_f_c( alpha_aa * beta.cdf(x, a_aa1, b_aa1)+(1-alpha_aa)*beta.cdf(x, a_aa0, b_aa0)))
def Pdp_aa(x,alpha_aa,a_aa0,b_aa0,a_aa1,b_aa1):
    return alpha_aa * beta.pdf(x, a_aa1, b_aa1)+(1-alpha_aa)*beta.pdf(x, a_aa0, b_aa0)

def Pdp_c(x,alpha_c,a_c0,b_c0,a_c1,b_c1):
    return alpha_c * beta.pdf(x, a_c1, b_c1)+(1-alpha_c)*beta.pdf(x, a_c0, b_c0)

# EqOpt
# beta.cdf(x_aa, a_aa1, b_aa1) = beta.cdf(x_c, a_c1, b_c1)

def f_eqopt(x,alpha_aa,alpha_c,a_aa0,b_aa0,a_c0,b_c0,a_aa1,b_aa1,a_c1,b_c1):
  	return beta.ppf(beta.cdf(x, a_aa1, b_aa1),a_c1,b_c1)

def Peqopt_aa(x,alpha_aa,a_aa0,b_aa0,a_aa1,b_aa1):
    return beta.pdf(x, a_aa1, b_aa1)

def Peqopt_c(x,alpha_c,a_c0,b_c0,a_c1,b_c1):
    return beta.pdf(x, a_c1, b_c1)

# EO 
# beta.pdf(x, a_aa0, b_aa0) = beta.pdf(fn(x), a_c0, b_c0)

def f_eo(x,alpha_aa,alpha_c,a_aa0,b_aa0,a_c0,b_c0,a_aa1,b_aa1,a_c1,b_c1):
    return beta.ppf(beta.cdf(x, a_c0, b_c0), a_aa0, b_aa0)

def Peo_aa(x,alpha_aa,a_aa0,b_aa0,a_aa1,b_aa1):
    return beta.pdf(x, a_aa0, b_aa0)

def Peo_c(x,alpha_c,a_c0,b_c0,a_c1,b_c1):
    return beta.pdf(x, a_c0, b_c0)

def balanced_eqn_fn(x, alpha_aa,alpha_c, fn, Pf_aa, Pf_c,paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0,a_c1, b_c1):
    if x < 0 or x == 0:
      x = 0.001
    if x > 1 or x == 1:
      x = 0.999
    x_aa = x
    x_c = fn(x,alpha_aa,alpha_c,a_aa0,b_aa0,a_c0,b_c0,a_aa1,b_aa1,a_c1,b_c1)
    return paa*(gamma(x,a_aa0,b_aa0,a_aa1,b_aa1,alpha_aa)-0.5)*P_evidence(x,alpha_aa,a_aa0,b_aa0,a_aa1,b_aa1)/Pf_aa(x,alpha_aa,a_aa0,b_aa0,a_aa1,b_aa1)+pc*(gamma(x_c,a_c0,b_c0,a_c1,b_c1,alpha_c)-0.5)*P_evidence(x_c,alpha_c,a_c0,b_c0,a_c1,b_c1)/Pf_c(x_c,alpha_c,a_c0,b_c0,a_c1,b_c1)

def balanced_eqn_un(x, a0,b0,a1,b1,alpha):
    if x < 0 or x == 0:
      x = 0.001
    if x > 1 or x == 1:
      x = 0.999
    return gamma_1(x,a0,b0,a1,b1,alpha) - 2

def get_policy_fn(alpha_aa, alpha_c, fn, Pf_aa, Pf_c, paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1):
    root=[]
    for i in np.arange(0.01,0.99,0.01):
        try:
            root.append(newton(balanced_eqn_fn, 
                               x0 = i, 
                               maxiter=50, 
                               args = (alpha_aa,alpha_c, fn, Pf_aa, Pf_c,paa, pc,
                                       a_aa0, b_aa0,a_aa1, b_aa1,
                                       a_c0, b_c0,a_c1, b_c1)))
        except(RuntimeError):
            pass

    root = [float(format(round(r, 4))) for r in root]

    return np.unique(root)[0],fn(np.unique(root)[0],alpha_aa,alpha_c,a_aa0,b_aa0,a_c0,b_c0,a_aa1,b_aa1,a_c1,b_c1),len(np.unique(root)),np.unique(root)[0]


def get_policy_un(alpha_aa,alpha_c, fn, Pf_aa, Pf_c,paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1):
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


def eva_policy(theta_aa,theta_c,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1):
    tpr = []
    fpr = []

    tpr.append(1-beta.cdf(theta_aa, a_aa1, b_aa1)) 
    tpr.append(1-beta.cdf(theta_c, a_c1, b_c1))

    fpr.append(1-beta.cdf(theta_aa, a_aa0, b_aa0))
    fpr.append(1-beta.cdf(theta_c, a_c0, b_c0))

    return tpr,fpr

def eva_classifier_fn(alpha_aa,alpha_c,policy,fn, Pf_aa, Pf_c,paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1):
	if policy == 'UN' :
		theta_aa, theta_c, _ , _ = get_policy_un(alpha_aa,alpha_c, fn, Pf_aa, Pf_c,paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)
	else:
		theta_aa, theta_c, _ , _ = get_policy_fn(alpha_aa, alpha_c, fn, Pf_aa, Pf_c, paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)
	return eva_policy(theta_aa,theta_c, a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)

def update(alpha,tpr,fpr,T, group):
    g0 = T[0,0,group]*(1-fpr[group]) + T[0,1,group]*fpr[group]
    g1 = T[1,0,group]*(1-tpr[group]) + T[1,1,group]*tpr[group]
    return alpha*g1 + (1-alpha)*g0

def balance_diff(alpha,tpr,fpr,T, group):
    g0 = T[0,0,group]*(1-fpr[group]) + T[0,1,group]*fpr[group]
    g1 = T[1,0,group]*(1-tpr[group]) + T[1,1,group]*tpr[group]    
    return g0 + (g1-g0-1)*alpha 

# def eva_classifier(alpha_aa,alpha_c,get_policy,paa, pc,a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1):
    
#     theta_aa, theta_c, _ , _ = get_policy(alpha_aa,alpha_c, paa, pc, a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)
#     return eva_policy(theta_aa,theta_c, a_aa0, b_aa0,a_aa1, b_aa1,a_c0, b_c0, a_c1, b_c1)

# def get_policy_eqopt(alpha_aa,alpha_c, 
#                      paa, pc,
#                      a_aa0, b_aa0,a_aa1, b_aa1,
#                      a_c0, b_c0, a_c1, b_c1):
#     root=[]
#     for i in np.arange(0.01,0.99,0.01):
#         try:
#             root.append(newton(balanced_eqn_eqopt, 
#                                x0 = i, 
#                                maxiter=50, 
#                                args = (alpha_aa,alpha_c, 
#                                        paa, pc,
#                                        a_aa0, b_aa0,a_aa1, b_aa1,
#                                        a_c0, b_c0,a_c1, b_c1)))
#         except(RuntimeError):
#             pass

#     root = [float(format(round(r, 4))) for r in root]

#     return np.unique(root)[0],np.unique(root)[0],len(np.unique(root)),np.unique(root)[0]
# 
# def balanced_eqn_eqopt(x, alpha_aa,alpha_c, 
#                        paa, pc,
#                        a_aa0, b_aa0,a_aa1, b_aa1,
#                        a_c0, b_c0,a_c1, b_c1):
#     return gamma_1(x,a_aa0, b_aa0,a_aa1, b_aa1,alpha_aa) * paa *alpha_aa- 2*(paa * alpha_aa + pc * alpha_c) + pc *alpha_c* gamma_1(x,a_c0, b_c0,a_c1, b_c1,alpha_c) 
