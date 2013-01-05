#!/usr/bin/env python


# right now this is just to test the .5 problem, so we'll go with full
# matrices.
 
 
import numpy
from numpy import *

import time, math
from matplotlib import pyplot


def row2col(row):
    """takes a row vector (a single array) and returns a column
    vector (a 2d array)"""
    return reshape(row, (len(row), 1))

# note that since we're not doing quasirandom at this stage, the
# inversion algorithm doesnt matter.  RandomArray seems to use
# Forsythe's algorithm
 
def stock(s0, r, sig, timesteps, T, numpaths):
    """generates gbm stocks with the parameters"""
    numassets, dt, sig2 = len(s0), T/timesteps, diagonal(sig)

    # this gives return[pathnum, timestep, numassetsension]
    returns = numpy.random.multivariate_normal(zeros(numassets), sig, \
                                            (numpaths, timesteps))

    premult = ones((numpaths, 1+timesteps, numassets))
    premult[:, 1:, :] = exp((r - sig2/2)*dt + sqrt(dt)*returns)

    return s0*multiply.accumulate(premult, 1)

def stock_pl(s0, r, sig, timesteps, T, numpaths):
    """generates gbm stocks from pseudorandom numbers and the linear
    discretization"""

    ret = normalpseudo(s0, r, sig, timesteps, T, numpaths)
    return stocklin(s0, r, sig, timesteps, T, numpaths, ret)

def stock_pb(s0, r, sig, timesteps, T, numpaths):
    """pseudorandom numbers with bb, for verification of the bb"""

    ret = normalpseudo(s0, r, sig, timesteps, T, numpaths)
    return stockbb(s0, r, sig, timesteps, T, numpaths, ret)



def normalpseudo(s0, r, sig, timesteps, T, numpaths):
    """creates a table of pseurorandom normals"""

    #RandomArray.seed(seedpair[0], seedpair[1])
    numassets, dt, sig2 = len(s0), T/timesteps, diagonal(sig)

    # this gives return[pathnum, timestep, numassetsension]
    return numpy.random.multivariate_normal(zeros(numassets), sig, \
                                            (numpaths, timesteps))



def stocklin(s0, r, sig, timesteps, T, numpaths, returns):
    """creates gbm stocks from the returns using the linear (ordinary)
    discretization"""
    numassets, dt, sig2 = len(s0), T/timesteps, diagonal(sig)

    premult = ones((numpaths, 1+timesteps, numassets) )
    premult[:, 1:, :] = exp((r - sig2/2.)*dt + sqrt(dt)*returns)

    return s0*multiply.accumulate(premult, 1)

def stockbb(s0, r, sig, timesteps, T, numpaths, returns):
    """creates stocks from the returns using the Brownian bridge
    (alternative) discretization"""
    numassets, dt, sig2 = len(s0), T/timesteps, diagonal(sig)

    # dim better be a power of 2
    assert(math.fabs(math.log(timesteps, 2.) - int(math.log(timesteps, 2.))) \
                                                < .00001)

    # setup the far point
     
    stock = zeros((numpaths, 1+timesteps, numassets))

    stock[:, 0] = s0
    stock[:, -1] = s0*exp((r - sig2/2.) * T + sqrt(T)*returns[:, 0])

    print "stockshape=", stock.shape, "ret.shape=", returns.shape

    bbtime = 1

    step = timesteps / 2
    while step:
        for t in range(step, timesteps, 2*step):

            stock[:, t] = sqrt(stock[:, t-step]*stock[:, t+step])*\
                exp(sqrt(dt * float(step)/2.) * returns[:, bbtime])
            bbtime += 1

        step /= 2


    assert(bbtime == timesteps)

    return stock





def minput(stock, strike):
    """computes the immediate exercise value for a minput for a group
    of stock trajectories"""


    # stock[path, time, assetnum]
     
     
    # timeasset means we map each path in the total vector, and then
    # we are looking at a single trajectory stock[time,asset]
    #
    # we map max onto this stock[time,asset], giving the largest
    # strike-stock
    #
    # finally limit this below with maximum, a ufunc which limits below

    return maximum(map(lambda timeasset: map(max, timeasset), strike-stock), 0.)

    #numpaths, numtimes = stock.shape[0], stock.shape[1] 
    #under = zeros((numpaths, numtimes) )
    #for p in range(numpaths):
        #for t in range(numtimes):
            #under[p, t] = max(0., max(strike-stock[p, t]))
    #return under


def lmc(stock, imm_payoff_fn, r, basisfns, T): 

    # first setup the cashflow and cashflowtimes according to the
    # last timestep (european)
    
    numpaths, dt, timesteps = len(stock), T/(stock.shape[1]-1), stock.shape[1]-1
    print "numpaths=%d, dt=%g" % (numpaths, dt)
    immediate = imm_payoff_fn(stock)
    #print "immediate=", immediate
    cashflow = array(immediate[:, -1])
    cashflowtime = T*ones(numpaths)

    numinmoneyvec = zeros(timesteps)

    #print cashflow
     
    for tx in arange(timesteps-1, -1, -1): 
        print "_"*30, "tx=", tx, "_"*30
        # first, we need to find the paths which are in the money
        
        inmoneyx = [px for px in range(numpaths) if immediate[px, tx] > 0.]
        print 'numinmoney', len(inmoneyx)
        numinmoneyvec[tx] = len(inmoneyx)
        if(len(inmoneyx) == 0): continue

        #inmoneyx = compress(choose(greater(immediate[:, tx]), (arange(numpaths), 0.)))

        in_stock = take(stock[:, tx, :], inmoneyx)
        in_imm = compress(greater(immediate[:, tx], 0.), immediate[:, tx])
        in_cash = take(cashflow, inmoneyx)
        in_cashtime = take(cashflowtime, inmoneyx)

        #print "inmoneystock=", in_stock

        # now use the basis functions to form the regression matrix
        # basis functions should be functions of the underlying and of
        # the immediate payoff
        basisdata = transpose([bf(in_stock, in_imm) for bf in basisfns])
         
        #print "basisdata=",basisdata

        # discount back the continuation values using ufuncs
         
        disccash = in_cash * exp(-r * (in_cashtime - tx*dt))

        #print "disccash=", disccash

        contcoeffs = numpy.linalg.lstsq(basisdata, disccash)[0]

        #rint "contcoeffs=", contcoeffs

        contvals = matrixmultiply(basisdata, contcoeffs)

        #rint "inmoneyimm=", in_imm
        #rint "contvals=", contvals

        #rint "greater=", choose(greater(contvals, in_imm), (in_imm,  in_cash))
    
        put(cashflow, inmoneyx, choose(greater(contvals, in_imm), \
                                    (in_imm,  in_cash)))
        #rint "cashflow=", cashflow

        put(cashflowtime, inmoneyx, choose(greater(contvals, in_imm), \
                                            (dt*tx, in_cashtime))) 
        #rint "cashflowtime=", cashflowtime
         
    mean = innerproduct(cashflow, exp(-r * cashflowtime))/float(numpaths)
    e2 = innerproduct(cashflow**2, exp(-2*r * cashflowtime))/float(numpaths)
    stddev = math.sqrt(e2-mean*mean)

    print "lsm=%g, stddev=%g" % (mean, stddev)

    return mean, stddev, numinmoneyvec




if __name__ == '__main__':
    # do some unit testing
    #do2 = 1
    timesteps, numassets, numpaths = 10, 1, 20000

    madan = (.0596, .1191,   100., .25)
    (r, sig, K, T) = madan
    S0 = 150.
     
    t0 = time.time()
 

    pl1=stock_pl(s0=[S0], r=r,  sig=array([[sig]]), \
        timesteps=2*timesteps, T=T, numpaths=numpaths)

    print "stocks generated" 
    #print s

    #print minput(s, 100.)
    basisfns1 = [lambda s, i: ones(len(s)), lambda s, i: s[:, 0], \
                lambda s, i: i,  lambda s, i: s[:, 0]*i]

    #basisfns = [lambda s, i: ones(len(s), ), lambda s, i: s[:, 0], \
    #           lambda s, i: s[:, 1],  lambda s, i: i, \
    #            lambda s, i: s[:, 1]*s[:, 0]]
#

    amervalpl1, stdpl1, numinmoneyvec = lmc(pl1, lambda s: minput(s, 100.), .06, basisfns1, .5)
    t1 = time.time()

    plot(numinmoneyvec)
    xlabel('timestep')
    ylabel('number of paths in money')
    title('Least-Squares Monte Carlo: Number of paths in money vs timestep')
    #output('lsminmoney.ps', 'postscript')


    print "pl1=%g" % (amervalpl1)
    print "elapsed=", t1 -t0
