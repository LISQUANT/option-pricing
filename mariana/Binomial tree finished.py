# -*- coding: utf-8 -*-

import numpy as np
import math 


T=100
r=0.03
K=40
St=25
vol=0.2
n=150
dt=T/n
k=0
u=np.exp(vol*np.sqrt(dt))
d=1/u
p=(np.exp(r*dt)-d)/(u-d)
allST=[]


exp_payoff=[max((St*u**k*d**(n-k))-K,0)*math.comb(n, k)*p**k*(1-p)**(n-k) for k in range(n+1)]
print(exp_payoff)
V0=(np.exp(-r*T))*sum(exp_payoff)
print(V0)


#Sanity check with Black- Scholes
import numpy as np
from scipy.stats import norm

N = norm.cdf

def BS_CALL(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * N(d1) - K * np.exp(-r*T)* N(d2)

BS_CALL(25,40,100,0.03,0.2)
V0-BS_CALL(25,40,100,0.03,0.2)