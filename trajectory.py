import numpy as np 
from scipy.stats import norm
import matplotlib.pyplot as plt
#from equlibrium1 import *

plt.rcParams.update({'font.size': 12})


def feasible(b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType):
	thetaOpt_a = 0.5*(mu0_a+mu1_a) - sigma_a**2*np.log(b_c*alpha_a/(1-alpha_a))/(mu1_a-mu0_a)
	thetaOpt_b = 0.5*(mu0_b+mu1_b) - sigma_b**2*np.log(b_c*alpha_b/(1-alpha_b))/(mu1_b-mu0_b)
	FNRopt_a1 = norm.cdf(thetaOpt_a,loc=mu1_a, scale=sigma_a)
	FNRopt_b1 = norm.cdf(thetaOpt_b,loc=mu1_b, scale=sigma_b)

	if fairType == 'EqOpt':
		fair_a = norm.ppf(FNRopt_b1,loc=mu1_a, scale=sigma_a) # (fair_a,thetaOpt_b) is a fair pair 
		fair_b = norm.ppf(FNRopt_a1,loc=mu1_b, scale=sigma_b) # (thetaOpt_a,fair_b) is a fair pair 
		if fair_a == thetaOpt_a:
			theta_a = [fair_a]
		else: 
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
	Err_a = [(1-norm.cdf(i,loc=mu0_a, scale=sigma_a))*(1-alpha_a) + norm.cdf(i,loc=mu1_a, scale=sigma_a)*alpha_a for i in theta_a]
	Err_b = [(1-norm.cdf(i,loc=mu0_b, scale=sigma_b))*(1-alpha_b) + norm.cdf(i,loc=mu1_b, scale=sigma_b)*alpha_b for i in theta_b]
	Err_total = np.array(Err_a)*Pa + np.array(Err_b)*(1-Pa)
	idx = np.argmin(Err_total)
	opt_a,opt_b = theta_a[idx],theta_b[idx]
	G1_a = norm.cdf(opt_a,loc=mu1_a, scale=sigma_a)
	G0_a = norm.cdf(opt_a,loc=mu0_a, scale=sigma_a)
	G1_b = norm.cdf(opt_b,loc=mu1_b, scale=sigma_b)
	G0_b = norm.cdf(opt_b,loc=mu0_b, scale=sigma_b)

	return G1_b,G0_b,G1_a,G0_a

	
def CDF(mu0,mu1,sigma,b_c,alpha):
	theta = 0.5*(mu0+mu1) - sigma**2*np.log(b_c*alpha/(1-alpha))/(mu1-mu0)
	G1 = norm.cdf(theta,loc=mu1, scale=sigma)
	G0 = norm.cdf(theta,loc=mu0, scale=sigma)
	return G1, G0


def fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,fairType,T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b):

	T = 10
	t = 0
	alpha_a,alpha_b = 0.1,0.9
	alpha_bList,alpha_aList = [alpha_b],[alpha_a]
	while t < T:
		t += 1
		if fairType == 'UN':
			G1_a,G0_a = CDF(mu0_a,mu1_a,sigma_a,b_c,alpha_a)
			G1_b,G0_b = CDF(mu0_b,mu1_b,sigma_b,b_c,alpha_b)
		elif fairType == 'intervention':
			G1_a,G0_a = CDF(mu0_a,mu1_a,sigma_a,b_c,alpha_a)
			G1_b,G0_b,_,_ = CDF_fair(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,'DP')
		
		elif fairType == 'intervention1':
			_,_,G1_a,G0_a = CDF_fair(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,'DP')
			G1_b,G0_b = CDF(mu0_b,mu1_b,sigma_b,b_c,alpha_b)

		else:
			G1_b,G0_b,G1_a,G0_a = CDF_fair(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType)
		
		g0_a = T00_a*G0_a + T10_a*(1-G0_a)
		g1_a = T01_a*G1_a + T11_a*(1-G1_a)
		g0_b = T00_b*G0_b + T10_b*(1-G0_b)
		g1_b = T01_b*G1_b + T11_b*(1-G1_b)

		alpha_aList.append(g0_a + (g1_a-g0_a)*alpha_a)
		alpha_bList.append(g0_b + (g1_b-g0_b)*alpha_b) 

		alpha_a = alpha_aList[-1]
		alpha_b = alpha_bList[-1]
	print('-----------')
	print(alpha_bList)
	print(alpha_aList)
	return alpha_a,alpha_b

def Img(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b):

	T00_bList,T11_bList = np.arange(T00_b,0.9,0.05),np.arange(T11_b,0.95,0.05)
	i,j = -1,-1
	Img_eqopt = np.ones([len(T00_bList),len(T11_bList)])
	Img_dp = np.ones([len(T00_bList),len(T11_bList)])
	Img_un = np.ones([len(T00_bList),len(T11_bList)])

	ImgAve_eqopt = np.ones([len(T00_bList),len(T11_bList)])
	ImgAve_dp = np.ones([len(T00_bList),len(T11_bList)])
	ImgAve_un = np.ones([len(T00_bList),len(T11_bList)])
	
	
	for T00_b in T00_bList:
		i += 1
		print(i)
		j = -1
		for T11_b in T11_bList:
			j += 1
			a_eqopt, b_eqopt = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'EqOpt',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
			a_dp, b_dp = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'DP',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
			a_un, b_un = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'UN',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
			
			print(a_eqopt, b_eqopt)
			print(a_dp, b_dp)
			print(a_un, b_un)

			Img_eqopt[i,j] = a_eqopt - b_eqopt
			Img_dp[i,j] = a_dp - b_dp
			Img_un[i,j] = a_un - b_un

			ImgAve_eqopt[i,j] = 0.5*(a_eqopt + b_eqopt)
			ImgAve_dp[i,j] = 0.5*(a_dp + b_dp)
			ImgAve_un[i,j] = 0.5*(a_un + b_un)

	print('-----')
	print(Img_eqopt)
	print('-----')
	print(Img_dp)
	print('-----')
	print(Img_un)
	

	fig=plt.figure(figsize=(15,4.5))
	ax1 = plt.subplot(131)
	ax2 = plt.subplot(132)
	ax3 = plt.subplot(133)

	im1 = ax1.imshow(Img_eqopt,extent=[0.05,0.4,0.5,0.9],cmap=plt.get_cmap('seismic'))
	ax1.set_xlabel(r'$T_{00}^b$')
	ax1.set_ylabel(r'$T_{11}^b$')
	ax1.set_title('EqOpt')
	fig.colorbar(Im1, ax=ax1)

	
	im2 = ax2.imshow(Img_eqopt,extent=[0.05,0.4,0.5,0.9],cmap=plt.get_cmap('seismic'))
	ax2.set_xlabel(r'$T_{00}^b$')
	ax2.set_ylabel(r'$T_{11}^b$')
	ax2.set_title('DP')
	fig.colorbar(Im2, ax=ax2)

	
	im3 = ax3.imshow(Img_eqopt,extent=[0.05,0.4,0.5,0.9],cmap=plt.get_cmap('seismic'))
	ax3.set_xlabel(r'$T_{00}^b$')
	ax3.set_ylabel(r'$T_{11}^b$')
	ax3.set_title('Unconstrained')
	fig.colorbar(Im3, ax=ax3)


	fig=plt.figure(figsize=(15,4.5))
	ax1 = plt.subplot(131)
	ax2 = plt.subplot(132)
	ax3 = plt.subplot(133)

	im1 = ax1.imshow(ImgAve_eqopt,extent=[0.05,0.4,0.5,0.9],cmap=plt.get_cmap('seismic'))
	ax1.set_xlabel(r'$T_{00}^b$')
	ax1.set_ylabel(r'$T_{11}^b$')
	ax1.set_title('EqOpt')
	fig.colorbar(Im1, ax=ax1)

	
	im2 = ax2.imshow(ImgAve_eqopt,extent=[0.05,0.4,0.5,0.9],cmap=plt.get_cmap('seismic'))
	ax2.set_xlabel(r'$T_{00}^b$')
	ax2.set_ylabel(r'$T_{11}^b$')
	ax2.set_title('DP')
	fig.colorbar(Im2, ax=ax2)

	
	im3 = ax3.imshow(ImgAve_eqopt,extent=[0.05,0.4,0.5,0.9],cmap=plt.get_cmap('seismic'))
	ax3.set_xlabel(r'$T_{00}^b$')
	ax3.set_ylabel(r'$T_{11}^b$')
	ax3.set_title('Unconstrained')
	fig.colorbar(Im3, ax=ax3)




def main():
	
	b_c = 0.1
	mu1_a,mu0_a,mu1_b,mu0_b = 5,-5,5,-10
	sigma_a,sigma_b = 10,10
	Pa = 0.5

	T10_a,T10_b = 0.5,0.5
	T01_a,T01_b = 0.5,0.5
	T00_a,T00_b = 0.8,0.7
	T11_a,T11_b = 0.3,0.2


	# impact of Transitions
	T = 10
	t = 0
	eqoptList_a,dpList_a,unList_a = [],[],[]
	eqoptList_b,dpList_b,unList_b = [],[],[]
	while t < T:
		t += 1
		#T10_a,T10_b = np.random.uniform(0,1,1),np.random.uniform(0,1,1)
		#T01_a,T01_b = np.random.uniform(0,1,1),np.random.uniform(0,1,1)
		#T00_a,T00_b = np.random.uniform(0,min(T10_a,T01_a),1),np.random.uniform(0,min(T10_b,T01_b),1)
		#T11_a,T11_b = np.random.uniform(max(T10_a,T01_a),1,1),np.random.uniform(max(T10_b,T01_b),1,1)
		
		#T10_a,T10_b,T01_a,T01_b,T00_a,T00_b,T11_a,T11_b = 0.61005452, 0.15532388, 0.57009774, 0.21506547, 0.38320505, 0.03233628, 0.80444976, 0.92473226
		'''
		T10_a = np.random.uniform(0,1,1)
		T01_a = np.random.uniform(0,1,1)
		T00_a = np.random.uniform(0,min(T10_a,T01_a),1)
		T11_a = np.random.uniform(max(T10_a,T01_a),1,1)
		T10_b = T10_a
		T01_b = T01_a
		T00_b = T00_a
		T11_b = T11_a
		'''
		print('========')
		print(T00_a,T10_a,T01_a,T11_a)
		print(T00_b,T10_b,T01_b,T11_b)
		a_eqopt, b_eqopt = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'EqOpt',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
		a_dp, b_dp = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'DP',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
		a_un, b_un = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'UN',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
		#print('mark')
		#a_inter, b_inter = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'intervention',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
		#a_inter1, b_inter1 = fixP(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,'intervention1',T10_a,T10_b,T00_a,T00_b,T01_a,T01_b,T11_a,T11_b)
			
		print('--------------')
		print(a_eqopt-b_eqopt)
		print(a_dp-b_dp)
		print(a_un-b_un)

		print(a_eqopt,b_eqopt)
		print(a_dp,b_dp)
		print(a_un,b_un)
		'''
		print(a_inter-b_inter)
		print(a_inter1-b_inter1)
		
		print('--------------')
		print(a_eqopt+b_eqopt)
		print(a_dp+b_dp)
		print(a_un+b_un)
		print(a_inter+b_inter)
		print(a_inter1+b_inter1)
		'''

		#print(a_eqopt, b_eqopt)
		#print(a_dp,b_dp)
		#print(a_un,b_un)

		#eqoptList_a.append(a_eqopt)
		#dpList_a.append(a_dp)
		#unList_a.append(a_un) 

		#eqoptList_b.append(b_eqopt)
		#dpList_b.append(b_dp)
		#unList_b.append(b_un)

	#print(dpList_a,dpList_b)
	#print(unList_a,unList_b)

	#plt.scatter(eqoptList_a,eqoptList_b,marker='o',color='m')
	#plt.scatter(dpList_a,dpList_b,marker='d',color='g')
	#plt.scatter(unList_a,unList_b,marker='*',color='b')
	#plt.show()






	




	'''
	fairType = 'DP'
	T = 10
	t = 0
	alpha_a,alpha_b = 0.5,0.5
	alpha_bList,alpha_aList = [alpha_b],[alpha_a]
	print('------')
	while t < T:
		print(t)
		t += 1
		G1_b,G0_b,G1_a,G0_a = CDF_fair(Pa,b_c,mu0_b,mu1_b,sigma_b,mu0_a,mu1_a,sigma_a,alpha_b,alpha_a,fairType)
		g0_a = T00_a*G0_a + T10_a*(1-G0_a)
		g1_a = T01_a*G1_a + T11_a*(1-G1_a)

		g0_b = T00_b*G0_b + T10_b*(1-G0_b)
		g1_b = T01_b*G1_b + T11_b*(1-G1_b)

		alpha_aList.append(g0_a + (g1_a-g0_a)*alpha_a)
		alpha_bList.append(g0_b + (g1_b-g0_b)*alpha_b) 

		alpha_a = alpha_aList[-1]
		alpha_b = alpha_bList[-1]

	print('===== alpha_aList, alpha_bList ========')
	print(alpha_aList)
	print(alpha_bList)

	
	t = 0
	alpha_a,alpha_b = 0.5,0.5
	alpha_bList_un,alpha_aList_un = [alpha_b],[alpha_a]
	while t < T:
		print(t)
		t += 1
		G1_a,G0_a = CDF(mu0_a,mu1_a,sigma_a,b_c,alpha_a)
		G1_b,G0_b = CDF(mu0_b,mu1_b,sigma_b,b_c,alpha_b)
		
		g0_a = T00_a*G0_a + T10_a*(1-G0_a)
		g1_a = T01_a*G1_a + T11_a*(1-G1_a)

		g0_b = T00_b*G0_b + T10_b*(1-G0_b)
		g1_b = T01_b*G1_b + T11_b*(1-G1_b)

		alpha_aList_un.append(g0_a + (g1_a-g0_a)*alpha_a)
		alpha_bList_un.append(g0_b + (g1_b-g0_b)*alpha_b) 

		alpha_a = alpha_aList_un[-1]
		alpha_b = alpha_bList_un[-1]

	print('===== alpha_aList_un, alpha_bList_un ========')
	print(alpha_aList_un)
	print(alpha_bList_un)
	'''



main()
		


		
	

	








