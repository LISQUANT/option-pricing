import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
print("Please imput values for the following metrics:")
option_type= str(input("type of option, call or put: "))
risk_free_rate=float(input("drift: "))
volatility= float(input("volatility: "))
days=int(input("days till expiration: "))
stock_price=float(input("stock price: "))
strike_price=float(input("strike price: "))
deltaT= days/252
def black_scholes(s0,K,T,r,sigma,option_type):
    d1=(np.log(s0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    if option_type=="call":
        price=s0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type=="put":
        price=K*np.exp(-r*T)*norm.cdf(-d2)-s0*norm.cdf(-d1)
    return price*100
def monte_carlo_pricing(s0,K,mu,sigma,deltaT,option_type):
    count=100000
    yi=np.random.normal(size=count)
    stock_var=(mu-((sigma**2)/2))*deltaT
    weiner_process=sigma*yi*np.sqrt(deltaT)
    s=s0*(np.exp(stock_var+weiner_process))
    if(option_type=="call"):
        payoff= np.maximum(s-K,0)
    elif(option_type=="put"):
        payoff= np.maximum(K-s,0)
    expected_value=payoff.mean()
    discount_value=np.exp(-mu*deltaT)
    return discount_value*expected_value
monte_carlo_price=monte_carlo_pricing(s0=stock_price,K=strike_price,mu=risk_free_rate,sigma=volatility,deltaT=deltaT,option_type=option_type)*100

print("Option value based on Monte Carlo:",monte_carlo_price)
bs_value=black_scholes(stock_price,strike_price,deltaT,risk_free_rate,volatility,option_type)
print("Option value based on Black Scholes:",bs_value)
def binomial_tree_pricing(s0,K,deltaT,volatility,r):
    step_count=10000
    dt=deltaT/step_count
    up_factor=math.exp(volatility*math.sqrt(dt))
    down_factor=1/up_factor
    risk_neutral_prob=(math.exp(r*dt)-down_factor)/(up_factor-down_factor)
    j=np.arange(step_count+1)
    stock_tree=s0*(up_factor**j)*(down_factor**(step_count-j))
    pay=np.maximum(stock_tree-K,0)
    for i in range(step_count-1,-1,-1):
        pay[:i+1]=np.exp(-r*dt)*(risk_neutral_prob*pay[1:i+2]+(1-risk_neutral_prob)*pay[:i+1])
    return pay[0]
bi_value=binomial_tree_pricing(s0=stock_price, K=strike_price, deltaT=deltaT, volatility=volatility, r=risk_free_rate)*100
print("Option value based on Binomial tree:",bi_value)