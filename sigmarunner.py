from math import *

class sigmarunner:
    def __init__(self):
        self.n = 0
        self.mean = 0.
        self.s = 0.
        self.mold = 0.
        self.sold = 0.

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.mold = x
            self.mean = x
        else:
            self.mean = self.mold + (x-self.mold)/(self.n+0.)
            self.s = self.sold + (x-self.mold)*(x-self.mean)

            self.mold = self.mean
            self.sold = self.s

    def variance(self):
        return self.n > 1 and (self.s/(self.n-1.)) or 0.

    def stdev(self):
        return sqrt(self.variance())

    def marginerror95(self):
        return 1.96*self.stdev()/sqrt(self.n)
    def conflo95(self):
        return self.mean-self.marginerror95()
    def confhi95(self):
        return self.mean+self.marginerror95()

    def __str__(self):
        return "n=%d mean=%g stdev=%g E=%g" % (self.n, self.mean, self.stdev(), self.marginerror95())

    #def __repr__(self):
        #return "n=%d mean=%g stdev=%g" % (self.n, self.mean, self.stdev())

if __name__=='__main__':
    s = sigmarunner()

    s.push(4)
    print(s)
    s.push(4)
    print(s)
    s.push(4)
    print(s)
    s.push(4)
    print(s)
    s.push(4)
    print(s)
    s.push(5)
    print(s)
    s.push(6)
    print(s)
