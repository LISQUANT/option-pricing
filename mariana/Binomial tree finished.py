# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 18:01:29 2026

@author: HP
"""

#~~~~~~~~~~~~~~~~~~ LEGEND ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
"""

Calls:
    
EC_MC -> European Call, priced with Monte Carlo
EC_T -> European Call, priced with Binomial tree
EC_BS -> European Call, priced with Black-Scholes
As_C_MC-> Asian Call, priced with Monte Carlo

Puts_

AP_T -> American put, priced with Binomial tree

"""


import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import binom


# ~~~~~~~~~~~~~~~~~~~~~ VARIABLES  ~~~~~~~~~~~~~~~~~~~~~~

T=25
r=0.03
vol=0.2
St=25
K=40
n=1000
dt=T/n

#~~~~~~~~~~~~~~~~~~~  BLACK-SCHOLES  ~~~~~~~~~~~~~~~~~~~~~~


N = norm.cdf

def EC_BS(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

EC_BS(St,K,T,r,vol)

#~~~~~~~~~~~~~~~~~~~  MONTE CARLO  ~~~~~~~~~~~~~~~~~~~~~~

t=np.linspace(0,T,n+1)

def stock_price(Sn,rng,dt_sim,r=r,vol=vol):
    Sn1=Sn*np.exp((r-(vol**2)/2)*dt_sim+(vol*np.sqrt(dt_sim)*rng.normal()))
    return Sn1

def EC_MC(n_sim,St,K,T,r=r,vol=vol,plot_paths= True,seed=None):
    ST=[]
    rng = np.random.default_rng(seed) #important to compute greeks
    dt_sim=T/n
    if plot_paths==True:
        plt.figure()
    for sim in range(n_sim):    #nº of sim must increase when n increases 
        Sn=St
        path=[Sn]
        for i in range(n):
            Sn=stock_price(Sn,rng,dt_sim,r,vol)
            path.append(Sn)
        ST.append(path[-1])
        if plot_paths==True:
            plt.plot(t,path) 
            plt.title('Monte Carlo Simulation')
            plt.xlabel('Time')
            plt.ylabel('St')    
    
    payoff=[max(x-K,0) for x in ST]
    Exp_Payoffs=[1/len(payoff)*x for x in payoff]
    V0_EC_MC=(np.exp(-r*T))*sum(Exp_Payoffs)
    Er_EC_MC=EC_BS(St,K,T,r,vol,0)-V0_EC_MC 
    return V0_EC_MC,Er_EC_MC

EC_MC(100,St,K,T,r,vol,1)
#~~~~~~~~~~~~~~~~~~~  BINOMIAL TREE  ~~~~~~~~~~~~~~~~~~~~~~

def EC_T(K,St,n,r,T):
    k=0
    u=np.exp(vol*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    
    exp_payoff=[max((St*u**k*d**(n-k))-K,0)*math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(n+1)]
    #print(exp_payoff)
    V0_BT=(np.exp(-r*T))*sum(exp_payoff)
    print(V0_BT)
    
EC_T(K, St, n, r, T)    
    
# ~~~~~~~~~~~~~~~ TREE PRICE VS Nº OF STEPS ~~~~~~~~~~~~~~

steps=[]
obs=100
for i in range(1,obs+1):
    pace=50*i
    steps.append(pace)
print(steps)


prices_tree=[]
for i in range(obs):
    n_step=steps[i]
    dt_tree=T/n_step
    k=0
    u=np.exp(vol*np.sqrt(dt_tree))
    d=1/u
    p=(np.exp(r*dt_tree)-d)/(u-d)
    exp_payoff=[max((St*u**k*d**(n_step-k))-K,0)*binom.pmf(k, n_step, p) for k in range(n_step+1)]
    V0_BT=(np.exp(-r*T))*sum(exp_payoff)
    prices_tree.append(V0_BT)
    
print(prices_tree)

plt.figure()
plt.title('Price vs n')
plt.plot(steps,prices_tree)
plt.xlabel('nº of steps')
plt.ylabel('Price')



# ~~~~~~~~~~~ MONTE CARLO ERROR VS Nº OF SIMULATIONS ~~~~~~~~~~~


nº_sim=[]
sims=100
for i in range(1,sims+1):
    pace=1000*i
    nº_sim.append(pace)
#print(nº_sim)


EC_MC_error=[]
for sim in nº_sim:
    price,error= EC_MC(sim,St,K,T,r, vol,0)
    EC_MC_error.append(error)
#print(MC_error)

plt.figure()
plt.title('Monte Carlo Errors vs nº of Simulations')
plt.plot(nº_sim,EC_MC_error)
plt.xlabel('nº of Simulations')
plt.ylabel('Monte Carlo Errors')

# ~~~~~~~~~~~~~~~~ GREEKS BY FINITE DIFFERENCES ~~~~~~~~~~~~~~~

#Using Centered finite difference

Sj=25
sims_greeks=1000

 
'Delta 6 Gamma'

h=0.1

delta=(EC_MC(sims_greeks,Sj+h,K,T,r,vol,0,1)[0]-EC_MC(sims_greeks,Sj-h,T,r,vol,0,1)[0])/(2*h)
print(delta)

gamma=(EC_MC(sims_greeks,Sj-h,K,T,r,vol,0,1)[0]-2*EC_MC(sims_greeks,Sj,K,T,r,vol,0,1)[0]+EC_MC(sims_greeks,Sj+h,K,T,r,vol,0,1)[0])/(h**2)
print(gamma)

'Theta'
ht=0.1
theta=-(EC_MC(sims_greeks,Sj,K,T+ht,r,vol,0,1)[0]-EC_MC(sims_greeks,Sj,K,T-ht,r,vol,1)[0])/(2*ht)
print(theta)

'Vega'
hv=0.01
vega=(EC_MC(sims_greeks,Sj,K,T,r,vol+hv,0,1)[0]-EC_MC(sims_greeks,Sj,K,T,r,vol-hv,0,1)[0])/(2*hv)
print(vega)

'Rho'
hr=0.01
rho=(EC_MC(sims_greeks,Sj,K,T,r+hr,vol,0,1)[0]-EC_MC(sims_greeks,Sj,K,T,r-hr,vol,0,1)[0])/(2*hr)
print(rho)

# ~~~~~~~~~~~~~~~~~~ AMERICAN PUT VALUATION ~~~~~~~~~~~~~~~~~

def AP_T(K,St,n,r,T):
    
    k=0
    u=np.exp(vol*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    
    put_values=[max((K- St*u**k*d**(n-k)),0) for k in range(n+1)]
    
    for i in range(n-1,-1,-1):
        new_put_values=[]
        for k in range(i+1):
            St_node=St*u**k*d**(i-k)
            exercise= max(K-St_node,0)
            continuation= np.exp(-r*dt)*(p*put_values[k+1]+(1-p)*put_values[k])
            '''if exercise>continuation:
                print(f"Step {i}, Node {k}: Early exercise optimal (St={St_node})")'''
            new_value=max(exercise,continuation)
            new_put_values.append(new_value)
        put_values=new_put_values
    
    return(put_values[0])

    
AP_T(K, St, n, r, T)

#CHECK -> American put> European put?

# 1st -> Pricing European Put with Black-Scholes

N = norm.cdf

def EP_BS(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r*T)* N(-d2)- S * N(-d1)

EP_BS(St,K,T,r,vol)

#2nd -> Compare values

if EC_BS(St,K,T,r,vol) > AP_T(K, St, n, r, T):
    print('American Put is not well defined!')

#~~~~~~~~~~~~~~~~~~~~~~~~~~ ASIAN OPTION ~~~~~~~~~~~~~~~~



def AsC_MC(n_sim,St,K,T,r=r,vol=vol,plot_paths= True,seed=None):
    t=np.linspace(0,T,n+1)
    ST_avg=[]
    rng = np.random.default_rng(seed) #important to compute greeks
    dt_sim=T/n
    if plot_paths==True:
        plt.figure()
    for sim in range(n_sim):    #nº of sim must increase when n increases 
        Sn=St
        path=[Sn]
        for i in range(n):
            Sn=stock_price(Sn,rng,dt_sim,r,vol)
            path.append(Sn)
        St_path_avg=sum(path)/len(path)
        ST_avg.append(St_path_avg)
        if plot_paths==True:
            plt.plot(t,path) 
            plt.title('Monte Carlo Simulation')
            plt.xlabel('Time')
            plt.ylabel('St')  
    #print(ST_avg)      
        
    
    payoff=[max(x-K,0) for x in ST_avg]
    Exp_Payoffs=[1/len(payoff)*x for x in payoff]
    V0_AsC_MC=(np.exp(-r*T))*sum(Exp_Payoffs)
    return V0_AsC_MC

AsC_MC(100,St,K,T,r,vol,1)

