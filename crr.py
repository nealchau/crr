from scipy import *
from matplotlib import pyplot
import math
import time


def N(x): return .5+.5*math.erf(x/sqrt(2))



def bsput(sig, T, r, S0, K):
    d1 = (log(S0/K)+(r+sig**2/2)*T)/(sig*sqrt(T))
    d2 = d1-sig*sqrt(T)
    return K*exp(-r*T)*N(-d2)-S0*N(-d1)




def europut(sig, steps, T, r, S0, K):

    deltat = T/float(steps)
    u = exp(sig*sqrt(deltat))
    d = exp(-sig*sqrt(deltat))
    p = (exp(r*deltat)-d)/(u-d)

    Smin = S0*d**steps
    Smax = S0*u**steps

    #print ("u=%g d=%g p=%g steps=%d Smin=%g Smax=%g") % (u,d,p,steps,Smin,Smax)

    optvalue = maximum(K-Smin*(u**(2.*arange(0.,steps+1))),zeros(steps+1))
    nextvalue = zeros(len(optvalue))

    for ti in range(steps-1, -1, -1):
        #print ("ti=%d")%(ti)
        #print optvalue
        nextvalue[:ti+1] = exp(-r*deltat)*(p*optvalue[1:ti+2]+(1-p)*optvalue[:ti+1])
        optvalue = array(nextvalue)

    return optvalue


def graphcrr(r, sig, K, T, S0, tsteplevels):
    bsval = bsput(sig,T,r,S0,K)
    err = zeros(len(tsteplevels))
    times = zeros(len(tsteplevels))

    for ti, Tsteps in enumerate(tsteplevels):
        t1 = time.time()
        putval = europut(sig, Tsteps, T, r, S0, K)
        t2 = time.time()
        err[ti] = abs(putval[0] - bsval)
        times[ti] = t2 - t1


    pyplot.plot(times, err)
 
if __name__ == '__main__':
    madan = (.0596, .1191,   100.,.25)
    (r, sig, K, T) = madan
    S0 = 100.
    tsteplevels = range(10,1000,50)

    pyplot.clf()

    graphcrr(r, sig, K, T, S0, tsteplevels)

    pyplot.xlabel("error versus time")
    pyplot.ylabel("absolute error")
    pyplot.title("execution time seconds")
    pyplot.yscale('log')
    pyplot.xscale('log')
    pyplot.show()


