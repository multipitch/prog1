# squareroot.py
#
# contains two functions that iterate over the following function:
#     x_k = (1/2) * [ x_(k-1) + a / x_(k-1) ]
# the first function, fsqrt, uses floating point arithmetic
# the second function, dsqrt, uses specified-precision decimal arithmetic
# 
# additionally, results using the above functions are collected and graphed
# using matplotlib
#
# Author:  Sean Tully
# Date:    23 Oct 2016
# Rev:     1.0

import matplotlib.pyplot as plt
import timeit
from decimal import *
import sys
from math import log

# set maximum number of iterations
kmax = 100
# note there will therefore be a maximum of (kmax + 1) results, i.e. the 
# original guess is x_0 and the maximum final solution is x_kmax

def fsqrt(a, kmax, eps=0):    
    '''
    Finds square root of a number using Babylonian Method and floating point
    arithmetic

    Keyword Arguments:
    a        (number):  number for which square root is required
    kmax     (int)   :  maximum number of iterations
    eps      (number):  user-specified epsilon (default = 0)

    Returns:
    results  (list)  :  the set of results after each iteration (including
                        initial guess) (list of floats)
    conv     (bool)  :  True if converged, False if not
    ''' 
    eps_m = sys.float_info.epsilon  # get value for machine epsilon
    xold = float(a)            # take 'a' as first guess, cast as float
    xnew = float('nan')        # initialise xnew as float, value unimportant
    results = [xold]           # record first (k = 0) guess
    conv = False
    for k in range(1,kmax+1):  # ensure max no. of iterations isn't exceeded
        xnew = 0.5 * (xold + a / xold)                    # Babylonian method    
        results.append(xnew)                              # record new guess
        if abs(xnew - xold) <= eps + 4.0*eps_m*abs(xnew): # test for convergence
            conv = True        # if convergence test met, set conv to true
            break              # and break out of iterations
        else:                  # if convergence test not met:
            xold = xnew        # update xold and iterate
    return results, conv       # return results and conversion status


def dsqrt(a, kmax, prec=getcontext().prec):
    '''
    Finds square root of a number using Babylonian Method and fixed-precision
    decimal arithmetic

    Keyword Arguments:
    a        (number):  number for which square root is required
    kmax        (int):  maximum number of iterations
    prec        (int):  decimal precision (defaults to existing setting)

    Returns:
    results  (list)  :  the set of results after each iteration (including
                        initial guess) (list of Decimal objects)
    conv     (bool)  :  True if converged, False if not
    ''' 
    getcontext().prec = prec   # set precision of Decimal objects
    xold = Decimal(a)          # take 'a' as first guess, cast as Decimal
    xnew = Decimal('NaN')      # initialise xnew as Decimal, value unimportant
    results = [xold]           # record first guess
    conv = False
    for k in range(1,kmax+1):  # ensure max no. of iterations isn't exceeded
        xnew = (xold + a / xold) / Decimal(2)  # Babylonian method
        results.append(xnew)                   # record new guess
        if Decimal.compare(xnew,xold) == 0:    # test for convergence
            conv = True        # if convergence test met, set conv to true
            break              # and break out of iterations
        else:                  # if convergence test not met:
            xold = xnew        # update xold and iterate
    return results, conv       # return results and conversion status

    
# set a large value for 'a'
a = 268435456   # (2**14)**2
xknown = 16384  # 2**14

# run floating point solver
# fx = list of outputs after each iteration
# fconv: True if converged, False if not
fx, fconv = fsqrt(a, kmax)

# run decimal solver for a range of precisions
# dx = list of (list of outputs after each iteration) for range of precisions
# dconv = list of convergence test outputs for each precision
dx = []
dconv = []
p = [4,28,100,200,300,400]             # specify precisions to use in runs
for prec in p:                         # loop for a range of precisions
    spam, eggs = dsqrt(a, kmax, prec)  # run decimal solver
    dx.append(spam)
    dconv.append(eggs)

# plot convergence for floating point and various fixed-precision decimal runs
# (plot of results as a function of number of iterations for various precisions)
s = ['.b-','vg-','*r-','+c-','xm-','1y-']              # styles to use
plt.plot(range(len(fx)), fx, 'ok-', label=r'$float$')  # plot float results
for i in range(len(p)):                                # plot decimal results
    plt.plot(range(len(dx[i])), dx[i], s[i], label=(p[i]))
plt.xlabel(r'$k$',fontsize=16)                         # add labels
plt.ylabel(r'$x_k$',fontsize=16)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
plt.legend(title=r'$precision$',fontsize=12)  
plt.plot()                                             # create plot
plt.savefig("fig4.png", format="png")                  # export as pdf
plt.close('all')

# plot convergence for floating point and various fixed-precision decimal runs
# (plot of number of iterations to achieve convergence as a function of 
# precision)
kconvs = []
for x in dx:
    kconvs.append(len(x)-1)
#plt.plot(p, kconvs, 'ob-', label=r'$decimal$')
plt.plot([0,max(p)],[len(fx)-1,len(fx)-1], ',k--', label=r'$float$')
plt.scatter(p, kconvs, label=r'$decimal$')
plt.axis([0, max(p), 0, max(kconvs)+1])
plt.xlabel(r'$p$',fontsize=16)                         # add labels
plt.ylabel(r'$k$',fontsize=16)
plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
plt.legend(title=r'$type$',fontsize=12, loc=4)  
plt.plot()                                             # create plot
plt.savefig("fig5.png", format="png")                  # export as pdf
plt.close('all')

# calculate relative errors
fe = []
for i in range(len(fx)):
    fe.append(abs(fx[i] - xknown) / xknown)
de = []
for i in range(len(dx)):
    de.append([])
    for j in range(len(dx[i])):
        de[i].append(abs(dx[i][j] - xknown) / xknown)


# plot some relative errors
s = ['.b-','vg-','*r-','+c-','xm-','1y-']              # styles to use
plt.plot(range(len(fe)), fe, 'ok-', label=r'$float$')  # plot float results
for i in range(len(p)):                                # plot decimal results
    plt.plot(range(len(de[i])), de[i], s[i], label=(p[i]))
plt.xlabel(r'$k$',fontsize=16)                         # add labels
plt.ylabel('relative error',fontsize=16)
plt.yscale('log')
plt.tick_params(axis='both', which='major', labelsize=10) # size tick labels
plt.legend(title=r'$precision$',fontsize=12)  
plt.plot()                                             # create plot
plt.savefig("fig6.png", format="png")                  # export as pdf
plt.close('all')

