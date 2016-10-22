import matplotlib.pyplot as plt
import timeit
from decimal import *
import sys

#epsilon = 1e-16
epsilon_m = sys.float_info.epsilon
# python reports epsilon_m = 2.220446049250313e-16

def bsqrt(a, epsilon_m, epsilon=0):    
    xold = a
    xnew = float('nan')
    while True:
        xnew = 0.5 * (xold + a / xold)    
        print(xnew)
        if abs(xnew - xold) <= epsilon + 4.0*epsilon_m*abs(xnew):
            return xnew
        else:
            xold = xnew 

x = bsqrt(2.0, epsilon_m)
