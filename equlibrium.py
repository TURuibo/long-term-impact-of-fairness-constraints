import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import OrderedDict

cmaps = OrderedDict()
plt.rcParams.update({'font.size': 12})

def CDF(mu0,mu1,sigma,b_c,alpha):
	theta = 0.5*(mu0+mu1) - sigma**2*np.log(b_c*alpha/(1-alpha))/(mu1-mu0)
	G1 = norm.cdf(theta,loc=mu1, scale=sigma)
	G0 = norm.cdf(theta,loc=mu0, scale=sigma)
	return G1, G0

def equlibrium(alphaList,LHS,RHS):
	idx = np.argmin(np.abs(np.array(LHS)-np.array(RHS)))
	return alphaList[idx]

def fixP(mu0,mu1,sigma,b_c,T11,T10,T01,T00):
	alphaList = np.arange(0.01,1,0.01)
	LHS,RHS = [],[]
	for alpha in alphaList:
		G1,G0 = CDF(mu0,mu1,sigma,b_c,alpha)
		LHS.append(1/alpha -1)
		RHS.append((1-(T11 - (T11-T01)*G1))/(T10 - (T10-T00)*G0))
	point = equlibrium(alphaList,LHS,RHS)
	return point


def main():
	b_c = 1
	mu1,mu0 = 5,-5
	sigma = 5

	T10 = 0.5
	T01 = 0.5
	T11_list = np.arange(T10,0.99,0.01)
	T00_list = np.arange(0.01,T01,0.01)
	i= -1
	Img = -1*np.ones([len(T00_list),len(T11_list)])
	for T11 in T11_list:
		i += 1
		j = -1
		for T00 in T00_list:
			j += 1
			Img[i,j] = fixP(mu0,mu1,sigma,b_c,T11,T10,T01,T00)
	print(Img)
	plt.imshow(Img,extent=[T10,0.99,0.01,T01],cmap=plt.get_cmap('seismic'))
	plt.xlabel(r'$T_{11}$')
	plt.ylabel(r'$T_{00}$')
	plt.colorbar()
	plt.savefig('105_5.eps')
	plt.show()

def feasible(b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType):
	thetaOpt_a = 0.5*(mu0_a+mu1_a) - sigma_a**2*np.log(b_c*alpha_a/(1-alpha_a))/(mu1_a-mu0_a)
	thetaOpt_b = 0.5*(mu0_b+mu1_b) - sigma_b**2*np.log(b_c*alpha_b/(1-alpha_b))/(mu1_b-mu0_b)
	FNRopt_a1 = norm.cdf(thetaOpt_a,loc=mu1_a, scale=sigma_a)
	FNRopt_b1 = norm.cdf(thetaOpt_b,loc=mu1_b, scale=sigma_b)

	if fairType == 'EqOpt':
		fair_a = norm.ppf(FNRopt_b1,loc=mu1_a, scale=sigma_a) # (fair_a,thetaOpt_b) is a fair pair 
		fair_b = norm.ppf(FNRopt_a1,loc=mu1_b, scale=sigma_b) # (thetaOpt_a,fair_b) is a fair pair 
		theta_a = np.arange(min(fair_a,thetaOpt_a),max(fair_a,thetaOpt_a),0.01)
		theta_b = [norm.ppf(norm.cdf(i,loc=mu1_a, scale=sigma_a),loc=mu1_b, scale=sigma_b) for i in theta_a]

	elif fairType == 'DP':
		TNRopt_a0 = norm.cdf(thetaOpt_a,loc=mu0_a, scale=sigma_a)
		TNRopt_b0 = norm.cdf(thetaOpt_b,loc=mu0_b, scale=sigma_b)
		NRopt_a = TNRopt_a0*(1-alpha_a) + FNRopt_a1*alpha_a
		NRopt_b = TNRopt_b0*(1-alpha_b) + FNRopt_b1*alpha_b
		ub_a = norm.ppf(TNRopt_b0,loc=mu1_a, scale=sigma_a)
		ub_b = norm.ppf(TNRopt_a0,loc=mu1_b, scale=sigma_b)
		lb_a = norm.ppf(FNRopt_b1,loc=mu0_a, scale=sigma_a)
		lb_b = norm.ppf(FNRopt_a1,loc=mu0_b, scale=sigma_b)
		
		theta_a = np.arange(lb_a,ub_a,0.01)
		theta_tmp_b = np.arange(lb_b,ub_b,0.001)
		NR_b = [norm.cdf(i,loc=mu0_b, scale=sigma_b)*(1-alpha_b) + norm.cdf(i,loc=mu1_b, scale=sigma_b)*alpha_a for i in theta_tmp_b]
		theta_b = []

		for i in theta_a:
			NR_a = norm.cdf(i,loc=mu0_a, scale=sigma_a)*(1-alpha_a) + norm.cdf(i,loc=mu1_a, scale=sigma_a)*alpha_a
			idx = np.argmin(np.abs(np.array(NR_a) - np.array(NR_b)))
			theta_b.append(theta_tmp_b[idx])

	else:
		print('error in fairType')
	return theta_a,theta_b




def CDF_fair(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType):

	theta_a,theta_b = feasible(b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType)
	#print('=======')
	#print(theta_a)
	#print(theta_b)
	Err_a = [(1-norm.cdf(i,loc=mu0_a, scale=sigma_a))*(1-alpha_a) + norm.cdf(i,loc=mu1_a, scale=sigma_a)*alpha_a for i in theta_a]
	Err_b = [(1-norm.cdf(i,loc=mu0_b, scale=sigma_b))*(1-alpha_b) + norm.cdf(i,loc=mu1_b, scale=sigma_b)*alpha_b for i in theta_b]
	Err_total = np.array(Err_a)*Pa + np.array(Err_b)*(1-Pa)
	#print(Err_total)
	idx = np.argmin(Err_total)
	opt_a,opt_b = theta_a[idx],theta_b[idx]
	G1_a = norm.cdf(opt_a,loc=mu1_a, scale=sigma_a)
	G0_a = norm.cdf(opt_a,loc=mu0_a, scale=sigma_a)
	G1_b = norm.cdf(opt_b,loc=mu1_b, scale=sigma_b)
	G0_b = norm.cdf(opt_b,loc=mu0_b, scale=sigma_b)

	return G1_b,G0_b,G1_a,G0_a
	

			


def fixPFair(Pa,T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b,b_c,mu1_a,mu0_a,mu1_b,mu0_b,sigma_a,sigma_b,fairType):
	alphaList_a = np.arange(0.01,1,0.03)
	alphaList_b = np.arange(0.01,1,0.03)
	
	LHS_a,RHS_a = [],[]
	pointList_a,pointList_b = [],[]
	for alpha_a in alphaList_a:
		LHS_a.append(1/alpha_a -1)
		LHS_b,RHS_b = [],[]
		RHStmp_a = []
		for alpha_b in alphaList_b:
			G1_b,G0_b,G1_a,G0_a = CDF_fair(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType)
			LHS_b.append(1/alpha_b -1)
			RHS_b.append((1-(T11_b - (T11_b-T01_b)*G1_b))/(T10_b - (T10_b-T00_b)*G0_b))
			
			RHStmp_a.append((1-(T11_a - (T11_a-T01_a)*G1_a))/(T10_a - (T10_a-T00_a)*G0_a))
		RHS_a.append(RHStmp_a)
		pointList_b.append(equlibrium(alphaList_b,LHS_b,RHS_b)) # i-th element: \alpha_b(\alpha_a[i])
		print(len(pointList_b))
	RHS_aT = np.array(RHS_a).T 
	for i in range(len(alphaList_b)):
		pointList_a.append(equlibrium(alphaList_a,LHS_a,RHS_aT[i,:])) # i-th element: \alpha_a(\alpha_b[i])
	

	print('====== parameters =======')
	print(mu1_a,mu0_a,mu1_b,mu0_b,sigma_a,sigma_b,Pa)
	print(T00_a,T10_a,T01_a,T11_a,T00_b,T10_b,T01_b,T11_b)
	print('---- fairness ------')
	print(fairType)
	print('----- pointList_b, pointList_a ------')
	print(pointList_b)
	print(pointList_a)

	plt.plot(alphaList_a,pointList_b,'r')
	plt.plot(pointList_a,alphaList_b,'b')
	plt.show()

	dist_min = 10**6
	for i in range(len(alphaList_a)):
		dist = np.linalg.norm(np.array([j- alphaList_a[i] for j in pointList_a])) + np.linalg.norm(np.array([j- pointList_b[i] for j in alphaList_b]))
		if dist < dist_min:
			dist_min = dist
			idx = i

	return alphaList_a[idx], pointList_b[idx]


def mainFair():
	b_c = 1
	mu1_a,mu0_a,mu1_b,mu0_b = 5,-5,5,-5
	sigma_a,sigma_b = 5,4.99
	Pa = 0.5

	T10_a,T10_b = 0.5,0.5
	T01_a,T01_b = 0.5,0.5
	T00_a,T00_b = 0.4,0.1
	T11_a,T11_b = 0.9,0.7

	'''
	mu1_a,mu0_a,mu1_b,mu0_b = 5,-5,5,-5
	sigma_a,sigma_b = 5,4
	Pa = 0.5

	T10_a,T10_b = 0.5,0.5
	T01_a,T01_b = 0.5,0.5
	T00_a,T00_b = 0.4,0.4
	T11_a,T11_b = 0.9,0.9
	'''

	fairType = 'EqOpt'
	alpha_a_eqopt, alpha_b_eqopt = fixPFair(Pa,T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b,b_c,mu1_a,mu0_a,mu1_b,mu0_b,sigma_a,sigma_b,fairType)
	fairType = 'DP'
	alpha_a_dp, alpha_b_dp = fixPFair(Pa,T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b,b_c,mu1_a,mu0_a,mu1_b,mu0_b,sigma_a,sigma_b,fairType)
	alpha_a_un = fixP(mu0_a,mu1_a,sigma_a,b_c,T11_a,T10_a,T01_a,T00_a)
	alpha_b_un = fixP(mu0_b,mu1_b,sigma_b,b_c,T11_b,T10_b,T01_b,T00_b)


	print(alpha_a_eqopt, alpha_b_eqopt)
	print(alpha_a_dp, alpha_b_dp)
	print(alpha_a_un,alpha_b_un)




#main()
mainFair()






