import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
print("Please imput values for the following metrics:")
option_type= str(input("type of option, call or put: "))
risk_free_rate=float(input("drift: "))
volatility= float(input("volatility: "))
days=int(input("days till expiration: "))
stock_price=float(input("stock price: "))
strike_price=float(input("strike price: "))
def black_scholes(s0,K,T,r,sigma,option_type):
    d1=(np.log(s0/K)+(r+0.5*sigma**2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    if option_type=="call":
        price=s0*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)
    elif option_type=="put":
        price=K*np.exp(-r*T)*norm.cdf(-d2)-s0*norm.cdf(-d1)
    return price*100
class Brownian():
    def __init__ (self,x0=0):
        self.x0=float(x0)
    def gen_normal(self,deltaT):
        yi=np.random.normal(size=count)
        return self.x0+yi*np.sqrt(deltaT)
    def stock_price(self, s0=70.67, mu=0.05, sigma=0.325, deltaT=1):
        stock_var=(mu-((sigma**2)/2))*deltaT
        weiner_process=sigma*self.gen_normal(deltaT)
        s=s0*(np.exp(stock_var+weiner_process))
        return s
b=Brownian()
deltaT= days/252
price_sum=0.0
count=100000
sim_price=b.stock_price(s0=stock_price, mu=risk_free_rate, sigma=volatility,deltaT=deltaT)
if(option_type=="call"):
    payoff= np.maximum(sim_price-strike_price,0)
elif(option_type=="put"):
        payoff= np.maximum(strike_price-sim_price,0)
expected_value=payoff.mean()
discount_value=np.exp(-risk_free_rate*deltaT)
option_value=discount_value*expected_value*100 #the standard contract includes 100 shares
print("Option value based on Monte Carlo:",option_value)
bs_value=black_scholes(stock_price,strike_price,deltaT,risk_free_rate,volatility,option_type)
print("Option value based on Black Scholes:",bs_value)