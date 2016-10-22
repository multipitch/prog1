# sequence.py
#
# contains two functions that iterate over the following function:
#     x_n = 108 - [815 - 1500/x_(n-2)] / x_(n-1)
# the first function, floatSolve, uses floating point arithmetic
# the second function, decSolve uses specified-precision decimal arithmetic
# 
# additionally, results using the above functions are collected and graphed
# using timeit and matplotlib
#
# Author:  Sean Tully
# Date:    22 Oct 2016
# Rev:     1.0

import matplotlib.pyplot as plt
import timeit
from decimal import *

# declare parameters
maxiter = 1000
maxprec = 100
initials = [4., 4.25]



def floatSolve(initials, maxiter):
    '''
    Given two initial values, iterates over a the function
        x_n = 108 - [815 - 1500/x_(n-2)] / x_(n-1)
    until the result has converged or the maximum number of iterations has been
    exceeded.  Convergence is defined here as when two consecutive iterations
    produce the same numerical result.  The initial two values are counted as
    the first two iterations, so that the set of x values spans from x[0], the
    first initial value, to x[n] = x[n-1].
    The function uses python floating point arithmetic.

    Keyword Arguments:
    initials  (list) :  a list containing the two initial values as numbers
                        (any type of number that can be recast as a float)
    maxiter   (int)  :  the maximum number of iterations to perform

    Returns:
    x         (list) :  the set of results, including the initial two values
    iters     (int)  :  the number of iterations until convergence (a value of 
                        -1 indicates that convergence wasn't achieved)
    xn        (float):  the converged result (or last result, if convergence was
                        not achieved)
    '''   
    iters = -1                     # value indicates convergence not achieved
    x = map(float, initials)       # ensure initial values are of type 'float'
    for n in range(2, maxiter):                   # don't exceed iteration limit
        xn = 108. - (815. - 1500./x[n-2])/x[n-1]  # apply function
        x.append(xn)                              # append results
        if x[n] == x[n-1]:                        # test for convergence
            iters = n
            break
    return x, iters, xn



def decSolve(initials, maxiter, prec=getcontext().prec):
    '''
    Given two initial values, iterates over a the function
        x_n = 108 - [815 - 1500/x_(n-2)] / x_(n-1)
    until the result has converged or the maximum number of iterations has been
    exceeded.  Convergence is defined here as when two consecutive iterations
    produce the same numerical result.  The initial two values are counted as
    the first two iterations, so that the set of x values spans from x[0], the
    first initial value, to x[n] = x[n-1].
    The function uses python fixed-precision decimal arithmetic, where the
    required precision may optionally be specified as an input.

    Keyword Arguments:
    initials  (list) :  a list containing the two initial values as numbers
                        (any type of number that can be recast as a float)
    maxiter   (int)  :  the maximum number of iterations to perform

    Returns:
    x         (list) :  the set of results, including the initial two values
    iters     (int)  :  the number of iterations until convergence (a value of 
                        -1 indicates that convergence wasn't achieved)
    xn        (float):  the converged result (or last result, if convergence was
                        not achieved)
    '''  
    iters = -1                     # value indicates convergence not achieved
    getcontext().prec = prec       # set decimal precision
    x = map(Decimal, initials)     # ensure initial values are of type 'Decimal'
    for n in range(2, maxiter):                  # don't exceed iteration limit
        xn = Decimal(108) - (Decimal(815) - Decimal(1500)/x[n-2])/x[n-1]
        x.append(xn)                             # apply funcion, append results
        if Decimal.compare(x[n],x[n-1]) == 0:    # test for convergence
            iters = n
            break
    return x, iters, xn



# initialise lists for storing outputs of decSolve
# (for x values, no. of iteraions, results)
dx = []
dn = []
dr = []
# run floatSolve, store x values, no. of iterations, result
fx, fn, fr = floatSolve(initials, maxiter)
# run decSolve for a range of decimal precisions and store results
for prec in range(1, maxprec+1):
    spam, eggs, sausage = decSolve(initials, maxiter, prec)
    dx.append(spam)
    dn.append(eggs)
    dr.append(sausage)


'''
# timings
def timeitWrapperFloat():
    spam, eggs, sausage = floatSolve(initials, maxiter)

def timeitWrapperDec():
    spam, eggs, sausage = decSolve(initials, maxiter, prec)

setup_float='from __main__ import timeitWrapperFloat'
ft = timeit.timeit('timeitWrapperFloat()', setup_float, number = 100)
setup_dec='from __main__ import timeitWrapperDec'
dt = []
for prec in range(1, maxprec+1):
    dt.append(timeit.timeit('timeitWrapperDec()', setup_dec, number = 100))



# plots
# x[n] as a function of iterations for float and fixed-precision decimals
p = [1,2,5,10,15,20]                           # range of precisions to plot
s = ['.b-','vg-','*r-','+c-','xm-','1y-']      # styles to use
plt.plot(range(len(fx)), fx, 'ok-', label=r'$float$') # plot float results
for i in range(6):                                    # plot decimal results
    plt.plot(range(len(dx[p[i]-1])), dx[p[i]-1], s[i], label=(p[i]))
plt.xlabel(r'$n$',fontsize=16)                        # add labels
plt.ylabel(r'$x_n$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
plt.legend(title=r'$precision$',fontsize=12)          # add legend
plt.plot()                                            # create plot
plt.savefig("fig1.pdf", format="pdf")                 # export as pdf
plt.close()
# time plot
plt.plot([0,maxprec+1],[ft,ft], ',r-', label=r'$float$')
plt.plot(range(1,maxprec+1),dt, 'ob-', label=r'$decimal$')
plt.xlabel(r'$precision$',fontsize=16)                        # add labels
plt.ylabel(r'$time (s)$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
plt.legend(title=r'$type$',fontsize=12, loc=4)
plt.savefig("fig2.pdf", format="pdf")                 # export as pdf
plt.close()
'''


# find roots of the underlying equation and plot the equation
import numpy as np
p = np.poly1d([1,-108,815,-1500])
r = p.roots
x = np.linspace(-10,110,241)
y = p(x)
plt.plot(x,y)
plt.scatter(r, p(r))
plt.xlabel(r'$x$',fontsize=16)
plt.ylabel(r'$f(x)$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=10)
plt.savefig("fig3.pdf", format="pdf")
plt.close()
