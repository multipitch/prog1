# sequence.py
#
# contains two functions that iterate over the following function, when given
# two starting values, x_(n-1) and x_(n-2):
#     x_n = 108 - [815 - 1500/x_(n-2)] / x_(n-1)
# the first function, floatSolve, uses floating point arithmetic
# the second function, decSolve uses specified-precision decimal arithmetic
# 
# additionally, results using the above functions are collected and graphed
# using timeit and matplotlib
#
# Author:  Sean Tully
# Date:    23 Oct 2016
# Rev:     1.0

import matplotlib.pyplot as plt
import timeit
from decimal import *
import sys

# set maximum number of iterations
nmax = 1000
# note there will therefore be a maximum of (nmax + 1) results, i.e. the 
# original guesses are x_0 and x_1 and the maximum final solution is x_nmax


def floatSolve(x_0, x_1, nmax, eps=0):
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
    x_0        (number):  first initial value
    x_1        (number):  second initial value
    nmax       (int)   :  the maximum number of iterations to perform
    eps        (number):  user-specified tolerance

    Returns:
    results    (list)  :  the set of results after each iteration (including
                          initial guess) (list of floats)
    conv       (bool)  :  True if converged, False if not
    '''   
    eps_m = sys.float_info.epsilon  # get value for machine epsilon
    xoldold = float(x_0)            # initial value for x_(n-2), cast as float
    xold = float(x_1)               # initial value for x_(n-1), cast as float
    results = [xoldold, xold]       # record initial guesses (n = 0, n = 1)
    conv = False
    for n in range(2, nmax+1):                    # don't exceed iteration limit
        xnew = 108. - (815. - 1500./xoldold)/xold # apply function
        results.append(xnew)                      # append results
        if abs(xnew - xold) <= eps + 4.0*eps_m*abs(xnew): # test for convergence
            conv = True        # if convergence test met, set conv to true
            break              # and break out of iterations
        else:                  # if convergence test not met:
            xoldold = xold     # updated xold and xoldold
            xold = xnew
    return results, conv       # return results and conversion status


def decSolve(x_0, x_1, maxiter, prec=getcontext().prec):
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
    x_0        (number):  first initial value
    x_1        (number):  second initial value
    nmax       (int)   :  the maximum number of iterations to perform
    prec       (int)   :  Decimal precision (defaults to unchanged)

    Returns:
    results    (list)  :  the set of results after each iteration (including
                          initial guess) (list of Decimal objects)
    conv       (bool)  :  True if converged, False if not
    '''  
    getcontext().prec = prec   # set decimal precision
    xoldold = Decimal(x_0)     # initial value for x_(n-2), cast as Decimal
    xold = Decimal(x_1)        # initial value for x_(n-1), cast as Decimal
    results = [xoldold, xold]  # record initial guesses (n = 0, n = 1)
    conv = False
    for n in range(2, nmax+1):                    # don't exceed iteration limit
        xnew = Decimal(108) - (Decimal(815) - Decimal(1500)/xoldold)/xold
        results.append(xnew)                      # apply funcion, append results
        if Decimal.compare(xnew,xold) == 0:       # test for convergence
            conv = True        # if convergence test met, set conv to true
            break              # and break out of iterations
        else:                  # if convergence test not met:
            xoldold = xold     # updated xold and xoldold
            xold = xnew
    return results, conv       # return results and conversion status


# set initial values
x_0 = 4.
x_1 = 4.25

# run floating point solver
# fx = list of outputs after each iteration
# fconv: True if converged, False if not
fx, fconv = floatSolve(x_0, x_1, nmax)

# run fixed-precision decimal solver for a range of precisions
dx = []
dconv = []
maxprec = 100
for prec in range(1, maxprec+1):                # loop for a range of precisions
    spam, eggs = decSolve(x_0, x_1, nmax, prec) # run decimal solver
    dx.append(spam)
    dconv.append(eggs)

# timings
doTimings = False
doTimings = True
nt = 100 # number of timing iterations

def timeitWrapperFloat():
    spam, eggs = floatSolve(x_0, x_1, nmax)

def timeitWrapperDec():
    spam, eggs = decSolve(x_0, x_1, nmax, prec)

if doTimings == True:
    setup_float='from __main__ import timeitWrapperFloat'
    # ft = timing for nt float runs
    ft = timeit.timeit('timeitWrapperFloat()', setup_float, number=nt)
    setup_dec='from __main__ import timeitWrapperDec'
    # dt = list of (timings for nt decimal runs at various precisions)
    dt = []
    for prec in range(1, maxprec+1):
        dt.append(timeit.timeit('timeitWrapperDec()', setup_dec, number=nt))



# plots

# x[n] as a function of iterations for float and fixed-precision decimals
p = [1, 5, 28, 50, 75, 100]                    # range of precisions to plot
s = ['.b-','vg-','*r-','+c-','xm-','1y-']      # styles to use
plt.plot(range(len(fx)), fx, 'ok-', label=r'$float$') # plot float results
for i in range(len(p)):                               # plot decimal results
    plt.plot(range(len(dx[p[i]-1])), dx[p[i]-1], s[i], label=(p[i]))
plt.xlabel(r'$n$',fontsize=16)                        # add labels
plt.ylabel(r'$x_n$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
plt.legend(title=r'$precision$',fontsize=12, loc=4)          # add legend
plt.axis([0, 160, -20, 120])
plt.plot()                                            # create plot
plt.savefig("fig1.pdf", format="pdf")                 # export as pdf
plt.close('all')

# time plot (time as a function of precision)
if doTimings == True:
    plt.plot([0,maxprec+1],[ft,ft], ',r-', label=r'$float$')
    plt.plot(range(1,maxprec+1),dt, 'ob-', label=r'$decimal$')
    plt.xlabel(r'$precision$',fontsize=16)                    # add labels
    plt.ylabel(r'$time (s)$',fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
    plt.legend(title=r'$type$',fontsize=12, loc=4)
    plt.plot()
    plt.savefig("fig2.pdf", format="pdf")                     # export as pdf
    plt.close('all')



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
plt.plot()
plt.savefig("fig3.pdf", format="pdf")
plt.close('all')

