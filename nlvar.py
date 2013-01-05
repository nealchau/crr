# $Id: nlvar.py,v 1.4 2009/02/19 19:01:40 sun Exp sun $

# ENTROPY?

from scipy import *
from pylab import *
from time import gmtime, strftime
import scipy.optimize, scipy.special, scipy.stats
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sigmarunner import *
import logging, csv


#matplotlib.use('PDF')
#import logging, Gnuplot, Gnuplot.funcutils
#import pychecker.checker

#def N(x): return (scipy.special.erf(x/sqrt(2.))[0]+1.)/2.
def N(x): return (scipy.special.erf(x/sqrt(2.))+1.)/2.

def bscall(sig, s0, K, r, T):
    d1 = (math.log(s0/K)+(r+sig*sig/2.)*T)/(sig*sqrt(T))
    d2 = d1 - sig*sqrt(T)
    return s0*N(d1)-K*exp(-r*T)*N(d2)

def impvolcall(callval, s0, K, r, T):
    if callval < s0-K*exp(-r*T)-.001: return -1
    optvaldiff = lambda s: fabs(callval-bscall(s, s0, K, r, T))
    return scipy.optimize.fmin(optvaldiff, [.2])
 
def bsput(sig, s0, K, r, T):
    d1 = (scipy.log(s0/K)+(r+sig*sig/2.)*T)/(sig*sqrt(T))
    d2 = d1 - sig*sqrt(T)
    log.log(5, "N(-d1=%g)=%g N(-d2=%g)=%g" % (-d1,N(-d1),-d2,N(-d2)))
    return -s0*N(-d1)+K*exp(-r*T)*N(-d2)

def impvolput(putval, s0, K, r, T):
    if putval < K*exp(-r*T)-s0-.001: return -1
    optvaldiff = lambda s: fabs(putval-bsput(s, s0, K, r, T))
    minsig = scipy.optimize.fminbound(optvaldiff, .000001, .5)
    return minsig



def normprobplot(x, w=-1.):
    """draw a normal probability plot for x"""

    n = len(x)
    zscores = stats.norm.ppf(arange(.5/n, 1., 1./n))
    assert(n == len(zscores))
    if(0<=w): plot(zscores, sort(x), label=str(w))
    else: plot(zscores, sort(x))

def nlvaralpha(n, numnoisetrd, alphavec, sig):

    ret = zeros((len(alphavec), numperiods))
    websig = zeros((len(alphavec), numperiods))
    ord = zeros((len(alphavec), numperiods, numnoisetrd))
    
    for alpha in alphavec:
        (ret[alpha], websig[alpha], ord[alpha]) = \
            nlvar(n, numnoisetrd, 0, 0, alpha, sig)

def nlvarlsv(n0, n1, numnoisetrd, alpha0, alpha1, sig):        
    suptitle("LSV Herding", fontweight='bold')
    subplot(2,2,1)
    ylabel("occurances")
    nlvar(n0, numnoisetrd, 0, alpha0, sig, False,False,'lsv')
    subplot(2,2,2)
    nlvar(n0, numnoisetrd, 0, alpha1, sig, False,False,'lsv')
    subplot(2,2,3)
    ylabel("occurances")
    xlabel("HM")
    nlvar(n1, numnoisetrd, 0, alpha0, sig, False,False,'lsv')
    subplot(2,2,4)
    xlabel("HM")
    nlvar(n1, numnoisetrd, 0, alpha1, sig, False,False,'lsv')

def nlvaracorr(n0, n1, numnoisetrd, alpha0, alpha1, sig):        
    suptitle("Autocorrelation", fontweight='bold')
    subplot(2,2,1)
    ylabel("autocorr")
    nlvar(n0, numnoisetrd, 0, alpha0, sig, False,False,'acorr')
    subplot(2,2,2)
    nlvar(n0, numnoisetrd, 0, alpha1, sig, False,False,'acorr')
    subplot(2,2,3)
    ylabel("autocorr")
    xlabel("lag")
    nlvar(n1, numnoisetrd, 0, alpha0, sig, False,False,'acorr')
    subplot(2,2,4)
    xlabel("lag")
    nlvar(n1, numnoisetrd, 0, alpha1, sig, False,False,'acorr')


def nlvarvolsurf4(nummcruns, n0, n1, numnoisetrd, alpha0, \
    alpha1, sig, K0, K1, numKsteps, numtermsteps):        
    nlvarvolsurf(nummcruns, n0, numnoisetrd, alpha0, sig,\
        K0, K1, numKsteps,  numtermsteps)
    nlvarvolsurf(nummcruns, n0, numnoisetrd, alpha1, sig,\
        K0, K1, numKsteps,  numtermsteps)
#    nlvarvolsurf(nummcruns, n1, numnoisetrd, alpha0, sig,\
#        K0, K1, numKsteps,  numtermsteps)
#    nlvarvolsurf(nummcruns, n1, numnoisetrd, alpha1, sig,\
#        K0, K1, numKsteps, numtermsteps)






def nlvarvolsurf():

    log.info("T=%g"%(T)) 
    
    deltaK = (K1-K0)/numKsteps
    deltaterm = numperiods/numtermsteps
    log.info('deltaterm=%g deltaK=%g numtermsteps=%d sig=%g'% (deltaterm,deltaK,numtermsteps,sig))
    #optval = zeros((numKsteps, numtermsteps))
    optval = [[sigmarunner() for i in range(numtermsteps)] \
         for j in range(numKsteps)]

    finstock = sigmarunner()
    termstock = [sigmarunner() for i in range(numtermsteps)]

    impvoll = zeros((numKsteps, numtermsteps))
    impvolm = zeros((numKsteps, numtermsteps))
    impvolh = zeros((numKsteps, numtermsteps))

    itm     = zeros((numKsteps, numtermsteps))
    
    for i in range(nummcruns):

        (ret, websignal, order) = nlvar(doplot='none')

# we'll leave s0=1, r=0 each simulated path will have
# sqrt(numsteps*numtraders)*sig standard deviation and no
# drift.  to make the impvol work, we need to subtract
# sig*sig/2. from each timestep, or sig^2*numsteps from the
# final point
        finstock.push(exp(sum(ret)))
        stockval = 1.
        #print "mu %g std %g" % (mstock, sqrt(sstock/i))
        for term in range(numtermsteps):
            termindex = (1+term)*deltaterm
            termT = (1+termindex)*deltaT
            disco = exp(-r*termT)
            stockval *= exp(sum(ret[term*deltaterm:termindex])) 
            termstock[term].push(stockval)
            for Ki in range(numKsteps):
                K = K0 + deltaK*Ki
                optval[Ki][term].push(disco*max(K-stockval,0.))
                if(stockval<K): itm[Ki, term] += 1

    log.info('finstock='+str(finstock))
    #log.info('optval.conflo95'+str([[sr.conflo95() for sr in sofk] for sofk in optval]))
    log.info('optval.mean'+str([[sr.mean for sr in sofk] for sofk in optval]))
    #log.info('optval.confhi95'+str([[sr.confhi95() for sr in sofk] for sofk in optval]))
    log.info('optval.stdev'+str([[sr.stdev() for sr in sofk] for sofk in optval]))

    for Ki in range(numKsteps):
        K = K0 + deltaK*Ki
        for term in range(numtermsteps):
            termT = (1+(1+term)*deltaterm)*deltaT
            impvoll[Ki,term] = impvolput(optval[Ki][term].conflo95(), 1., K, r, termT)
            impvolm[Ki,term] = impvolput(optval[Ki][term].mean, 1., K, r, termT)
            log.debug('impvolput(%g,1.,%g,%g,%g)=%g'%(optval[Ki][term].mean,K,r,termT,impvolm[Ki,term]))
            log.debug('termT=%g'%(termT))
            impvolh[Ki,term] = impvolput(optval[Ki][term].confhi95(), 1., K, r, termT)

    log.info('K='+str(arange(K0,K1,deltaK)))
    log.info('term='+str(arange(deltaterm, numperiods, deltaterm)))
    #log.info('impvoll='+str(impvoll))
    log.info('impvolm='+str(impvolm))
    #log.info('impvolh='+str(impvolh))

    zeroconf=impvolh-impvoll
    

    return (impvoll, impvolm, impvolh)



def plotimpsurf(nummcruns, n, numnoisetrd, alpha, sig, K0, \
    K1, numKsteps, numtermsteps):

    deltaK = (K1-K0)/numKsteps
    deltaterm = numperiods/numtermsteps
    (impvoll,impvolm,impvolh) = nlvarvolsurf(nummcruns, n, numnoisetrd, alpha, sig, K0, \
        K1, numKsteps, numtermsteps)

    log.info('impvol='+str(impvol))
    Kax = arange(K0, K1+.0001, deltaK+.0001)
    termax = arange(deltaterm,1+numperiods+.0001, deltaterm+.0001)
    doplot = True
    if len(Kax) > 2 and len(termax) > 2:
        fig = figure()
        ax = Axes3D(fig)
        Kax, termax = meshgrid(Kax,termax)
        ax.plot_surface(Kax, termax, impvol, rstride=1, cstride=1, cmap=cm.binary)
#        ax.plot_surface(Kax, termax, itm, rstride=1, cstride=1, cmap=cm.jet)

        ax.set_xlabel("$K$ strike")
        ax.set_ylabel("$t$ expiration")
        ax.set_zlabel("implied $\sigma$ ($n=%d\ \\alpha=%.2f$)"%(n,alpha))
   

def nlvar(impulse=False, doplot='all'):

    nlog = logging.getLogger('nlvar.nlvar')
    nlog.setLevel(logging.INFO)
    
    order       = zeros((n+numperiods, numtrdrs))
    ret         = zeros(n+numperiods)

#here sig is standard deviation
    gaussgen    = scipy.stats.norm((r-sig*sig/2.)*deltaT,sig*sqrt(deltaT))
    noise       = gaussgen.rvs((n+numperiods, numtrdrs))


    avgord      = zeros((n+numperiods,numtrdrs))
    avgret      = zeros((n+numperiods,numtrdrs))
    ttlret      = zeros((n+numperiods,numtrdrs))

    websignal   = zeros(n+numperiods)

# to avoid division by zero

    if impulse: 
        order[:n]       = 1.
        avgord[:n]      = 1.
        websignal[:n]   = 1.

# over the course of the simulation, we go numperiods
# timesteps, each time we're adding together
# \sqrt(numtraders)*indivsig, which is the effective vol
# rate at each timestep.  hence total vol over a simulation
# should be \sqrt(numtraders*numperiods)*sig

    order[:n]   = noise[:n]
    ret[:n]     = sum(order[:n], axis=1)
    cumord      = cumsum(order[:n], axis=1)

    avgord[0] = order[0]
    avgret[0] = order[0]*ret[0]
    for t in range(1,n):
        avgord[t] = cumord[t]/(t+1.)
        avgret[t] = avgret[t-1] + (1./(t+1.))*(order[t]*ret[t]-avgret[t-1])


    for t in range(n, n+numperiods):
        #nlog.debug('websig=%g alpha=%g'%(websignal[t-1], alpha))
        order[t]=alpha*websignal[t-1]+noise[t]

    # in the absence of alpha, at each timestep we are adding together
    # numtraders gaussians with  standard deviation sig,
    # giving total sigma per timestep = \sqrt(numtraders)*\indivsig
        ret[t] = sum(order[t])


# now we form the websignal

        # we assume the average order and return upto t-1 has been 0
        avgord[t] = avgord[t-1] + (1./n)*(order[t]-order[t-n])

        avgret[t] = avgret[t-1]+\
            (1./n)*(ret[t]*order[t]-ret[t-n]*order[t-n])

        dot_ordret = dot(avgord[t], avgret[t])

        #ttlret[t] = ttlret[t-1]+ret[t]*order[t]
#
# this has |p| < 1 which is what we want.
        websignal[t] = dot_ordret/(.1+sum(avgret[t]))
                        
        #nlog.log(5,"____________t=%d__________________" % (t))
        #nlog.log(5,"ORDER %s"%str(order[t]))
        #magord = sqrt(dot(avgord[t], avgord[t]))
        #magret = sqrt(dot(avgret[t], avgret[t]))
        #theta = arccos(dot_ordret/(magord*magret))
        #nlog.log(5,"ret %g theta %g magord %g magret %g"% (ret[t], 180*theta/pi, magord, magret))
        #nlog.log(5,"avgord %s"%str(avgord[t]))
        #nlog.log(5,"num %g denom %g websignal %g" % (dot(avgord[t], avgret[t]), sum(avgret[t]),\
        #        websignal[t]))
        #nlog.log(5,"ret %s" %str(ret))

    return (ret[n:], websignal[n:], order[n:])

def nlvarplot():
    if doplot == 'lsv':
        title("$\sigma$=%g n=%d $\\alpha$=%g"%(sig, n, alpha))
        buyers = sum(less_equal(-order, 0),1)
        #plot(buyers)

        var1 = sum([abs(i-numtrdrs/2.)*\
            comb(numtrdrs,i)/(2.**numtrdrs) \
            for i in range(numtrdrs+1)])
        log.debug("lsv var1=%g" % (var1))
        hist(abs(buyers-numtrdrs/2.)-var1,numtrdrs)

    elif doplot == 'acorr':
        title("$\sigma$=%g n=%d $\\alpha$=%g"%(sig, n, alpha))
        #acorr(ret, normed=True, usevlines=True)
        acorr(ret, maxlags=100)

    elif doplot == 'all' or doplot == 'fast':
        subplot(331)
        title("ret")
        plot(ret)
        plot(websignal)
        subplot(332)
        title("dist")
        hist(ret)
        subplot(333)
        title("autocorr")
        acorr(ret, normed=True, usevlines=True)
        subplot(334)
        title("norm pp")
        normprobplot(ret)

        price = cumsum(ret)

        subplot(335)
        title("price")
        plot(price)

        if doplot == 'all':
            subplot(336)
            title("ttlr")
            imshow(sort(ttlret), aspect='auto')
            
            subplot(337)
            title("ord")
            imshow(sort(order, axis=1), aspect='auto')

        subplot(338)
        #title("avgret")
        #imshow(sort(avgret, axis=1), aspect='auto')
        title("lsv")
        buyers = sum(less_equal(-order, 0),1)
        #plot(buyers)

        var1 = sum([abs(i-numtrdrs/2.)*\
            comb(numtrdrs,i)/(2.**numtrdrs) \
            for i in range(numtrdrs+1)])
        log.debug("lsv var1=%g" % (var1))
        hist(abs(buyers-numtrdrs/2.)-var1,numtrdrs)
        


        subplot(339)
        title("websignal")
        plot(websignal)

    return (ret, websignal, order)


if __name__ == '__main__':
    logging.basicConfig(level=10) 
    log = logging.getLogger('nlvar')
    log.setLevel(0)

    numinftrd     = 0
    nummomtrd     = 0

    volparams     = dict(numperiods=500, numnoisetrd=4, sig=.2, r=.01)
    regparams     = dict(numperiods=10000, numnoisetrd=100, sig=.3)

    currentparams = volparams

    numperiods    = currentparams['numperiods']
    numnoisetrd   = currentparams['numnoisetrd']

    numtrdrs      = numnoisetrd + numinftrd + nummomtrd

    sig           = currentparams['sig']
    r             = currentparams['r']

    alpha         = .7
    n             = 1
    T             = 1.*numperiods
    deltaT        = T/numperiods

    nummcruns     = 500
    K0            = .9
    K1            = 1.5
    numKsteps     = 3
    numtermsteps  = 3

    #nlvar(n, numnoisetrd, numinftrd, alpha, sig, impulse=False, \
    #    doprint=False, doplot='autocorr')

    #nlvarlsv(2, 10, 100, .01, .05, .1)
    #nlvaracorr(2, 10, 100, .01, .10, .1)
#    nlvarvolsurf4(nummcruns, 2, 10, numnoisetrd, .01, .3, \
#        sig, K0, K1, numKsteps, numtermsteps)
    #nlvarvolsurf(nummcruns, 2, numnoisetrd, .00, sig, K0, K1, numKsteps, numtermsteps)
    #nlvarvolsurf(nummcruns, 2, numnoisetrd, .00, sig, K0, K1, numKsteps, numtermsteps)
    #nlvarvolsurf()
    #plotimpsurf(nummcruns, 2, numnoisetrd, .00, sig, K0, K1, numKsteps, numtermsteps)
# we'd expect an implied volatility approximately
# \sqrt(numtrders)*sig here
    #nlvarvolsmile(nummcruns, n, numnoisetrd, alpha, sig, K0, K1, numKsteps, '', numtermsteps)

    (r, w, o) = nlvar()
    print r
    print w
    print o

    f = open("nlvar.csv","w")
    for i in range(len(r)): f.write(str(r[i])+"\n")
    f.close()

#    show()
