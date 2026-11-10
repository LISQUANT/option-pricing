# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 20:40:07 2026

@author: HP
"""


import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
from scipy.stats import binom

T=100
r=0.03
vol=0.2
St=25
K=40
n=1000
dt=T/n


#~~~~~~~~~~~~~~~~~~~  MONTE CARLO  ~~~~~~~~~~~~~~~~~~~~~~

t=np.linspace(0,T,n+1)

def stock_price(Sn):
    Sn1=Sn*np.exp((r-(vol**2)/2)*dt+(vol*np.sqrt(dt)*rng.normal()))
    return Sn1

ST=[]
for sim in range(100000):    #nº of sim must increase when n increases 
    Sn=St
    path=[Sn]
    for i in range(n):
        Sn=stock_price(Sn)
        path.append(Sn)
    plt.plot('Monte Carlo Simulation')
    plt.plot(t,path)
    plt.xlabel('Time')
    plt.ylabel('St')
    ST.append(path[-1])
    
print(ST)
payoff=[max(x-K,0) for x in ST]
Exp_Payoffs=[1/len(payoff)*x for x in payoff]
V0_MC=(np.exp(-r*T))*sum(Exp_Payoffs)
print(V0_MC)


#~~~~~~~~~~~~~~~~~~~  BINOMIAL TREE  ~~~~~~~~~~~~~~~~~~~~~~

k=0
u=np.exp(vol*np.sqrt(dt))
d=1/u
p=(np.exp(r*dt)-d)/(u-d)

exp_payoff=[max((St*u**k*d**(n-k))-K,0)*math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(n+1)]
print(exp_payoff)
V0_BT=(np.exp(-r*T))*sum(exp_payoff)
print(V0_BT)


#~~~~~~~~~~~~~~~~~~~  BLACK-SCHOLES  ~~~~~~~~~~~~~~~~~~~~~~


N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

BS= BS_CALL(St,K,T,r,vol)

#~~~~~~~~~~~~~~~~~ CONVERGENCE WITH BS ~~~~~~~~~~~~~~~~~~~~

E_MC=BS-V0_MC
E_BT=BS-V0_BT
print(E_MC)
print(E_BT)

# ~~~~~~~~~~~~~~~ TREE PRICE VS Nº OF STEPS ~~~~~~~~~~~~~~

steps=[]
obs=100
for i in range(1,obs+1):
    pace=50*i
    steps.append(pace)
print(steps)


prices_tree=[]
for i in range(obs):
    n=steps[i]
    dt=T/n
    k=0
    u=np.exp(vol*np.sqrt(dt))
    d=1/u
    p=(np.exp(r*dt)-d)/(u-d)
    exp_payoff=[max((St*u**k*d**(n-k))-K,0)*binom.pmf(k, n, p) for k in range(n+1)]
    V0_BT=(np.exp(-r*T))*sum(exp_payoff)
    prices_tree.append(V0_BT)
    
print(prices_tree)

plt.title('Price vs nº of steps')
plt.plot(steps,prices_tree)
plt.xlabel('nº of steps')
plt.ylabel('Price')
