import numpy as np

euler = np.euler_gamma

def harmonic(N):
    '''
        Returns n-th harmonic nunmber; See Gravel (2014) et al.
    '''
    if(N < 20):
        return np.sum(1. / np.arange(1, N+1))
    else:
        return np.log(N) + euler + 1./(2*N) - 1./(12*N**2) + 1./(120*N**4)
    
def deltaf(M,N):
    '''
        Delta(M,N) := ()
    '''
    N_tot = M+N
    return harmonic(N_tot-1) - harmonic(N-1)

def missed(sfs_0, M, order):
    '''
        Input :
            sfs_0 array of lenght n+1; sfs[1] is  # things observed once
            M <int> extrapolation size
            order <int> jackknife order
        Output :
            missed <array of len M>
    '''
    n = len(sfs_0)-1
    delta = deltaf(M,n)
#     print(delta)
    if order == 1:
        missed = (-1.+n) / n * delta * sfs_0[1]
    elif order == 2:
        missed = (((1+2*(-2+n)*n)*delta)/(n*(-3+2*n))+((-2+n)*(-1+n)*delta**2)/(n*(-3+2*n)))*sfs_0[1]+(-((2*(-2+n)**2*delta)/((-1+n)*n*(-3+2*n)))-(2*(-2+n)**2*delta**2)/(n*(-3+2*n)))*sfs_0[2]
    elif order == 3:
        n = float(n)
        missed = (delta*(-127+628*n-1386*n**2+1722*n**3-1269*n**4+546*n**5-126*n**6+12*n**7+(-1+n)*(-10-87*n+310*n**2-372*n**3+210*n**4-57*n**5+6*n**6)*delta+2*(-2+n)**2*(-1+n)**3*(6-5*n+n**2)*delta**2)*sfs_0[1])/((-1+n)**2*n*(-5+2*n)*(-3+2*n)*(11-12*n+3*n**2))+(delta*(-2*(-248+402*n-61*n**2-249*n**3+195*n**4-57*n**5+6*n**6)-2*(-1+n)*(-560+1130*n-731*n**2+55*n**3+127*n**4-51*n**5+6*n**6)*delta+2*(-1+n)**2*(6-5*n+n**2)*(52-78*n+38*n**2-6*n**3)*delta**2)*sfs_0[2])/((-1+n)**2*n*(-5+2*n)*(-3+2*n)*(11-12*n+3*n**2))+(delta*(6*(3-2*n)**2*(-3+n)**3+6*(-3+n)**3*(-1+n)*(15-19*n+6*n**2)*delta+6*(-3+n)**2*(-1+n)**2*(-3+2*n)*(6-5*n+n**2)*delta**2)*sfs_0[3])/((-1+n)**2*n*(-5+2*n)*(-3+2*n)*(11-12*n+3*n**2))
    elif order == 4:
        n = float(n)
        missed = (delta*(2*(79138 - 768163*n + 3305209*n**2 - 8455556*n**3 + 14456396*n**4 - 17528889*n**5 + 15570611*n**6 - 10302336*n**7 + 5104884*n**8 - 1885839*n**9 + 511449*n**10 - 98781*n**11 + 12849*n**12 - 1008*n**13 + 36*n**14) + (-1 + n)*(-142984 + 984880*n - 3219708*n**2 + 6550921*n**3 - 9150649*n**4  + 9174455*n**5 - 6742096*n**6 + 3659027*n**7 - 1462263*n**8 + 424335*n**9  - 86856*n**10 + 11874*n**11 - 972*n**12 + 36*n**13)*delta + 2*(-1 + n)**2*(6 - 5*n + n**2)*(-2288 + 3592*n + 10090*n**2 - 36645*n**3 + 50160*n**4 - 39031*n**5 + 19017*n**6 - 5922*n**7 + 1147*n**8 - 126*n**9 + 6*n**10)*delta**2 + (-4 + n)*(-1 + n)**3*(6 - 5*n + n**2)**2*(184 - 606*n + 803*n**2 - 549*n**3  + 204*n**4 - 39*n**5 + 3*n**6)*delta**3)*sfs_0[1])/((-2 + n)*(-1 + n)**3*n *(-7 + 2*n)*(-5 + 2*n)*(-3 + 2*n)*(5 - 5*n + n**2)*(26 - 18*n + 3*n**2)*(11- 12*n + 3*n**2)) + (delta*(2*(806336 - 4114800*n + 9788448*n**2 - 14623884*n**3 + 15584978*n**4 - 12651005*n**5 + 8033192*n**6 - 3985285*n**7 + 1516278*n**8- 429717*n**9 + 87120*n**10 - 11877*n**11 + 972*n**12 - 36*n**13)- 2*(-1 + n)*(-1762976 + 8275744*n - 17540568*n**2 + 22472256*n**3 - 19868590*n**4 + 13296025*n**5 - 7229740*n**6 + 3298055*n**7 - 1236311*n**8 + 360392*n**9 - 76491*n**10 + 10944*n**11 - 936*n**12 + 36*n**13)*delta - 4*(-1 + n)**2*(6 - 5*n + n**2)*(-117832 + 426816*n - 645654*n**2 + 513018*n**3 - 207385*n**4 + 15319*n**5 + 25646*n**6 - 13195*n**7 + 3075*n**8 - 366*n**9 + 18*n**10)*delta**2 - 2*(-4 + n)*(-1 + n)**3*(6 - 5*n + n**2)**2*(3176 - 8768*n+ 9854*n**2 - 5764*n**3 + 1850*n**4 - 309*n**5 + 21*n**6)*delta**3)*sfs_0[2])/((-2 + n)*(-1 + n)**3*n*(-7 + 2*n)*(-5 + 2*n)*(-3 + 2*n)*(5 - 5*n + n**2)*(26 - 18*n + 3*n**2)*(11 - 12*n + 3*n**2)) + (delta*(12*(3 - 2*n)**2*(-3 + n) *(24678 - 73356*n + 88865*n**2 - 54881*n**3 + 16400*n**4 - 464*n**5 - 1289*n**6 + 419*n**7 - 57*n**8 + 3*n**9) + 6*(-1 + n)*(9 - 9*n + 2*n**2)*(-354348 + 1276800*n - 1961648*n**2 + 1662242*n**3 - 829199*n**4 + 230491*n**5- 21616*n**6 - 6925*n**7 + 2631*n**8 - 354*n**9 + 18*n**10)*delta +6*(-1 + n)**2*(-3 + 2*n)*(6 - 5*n + n**2)*(138996 - 425226*n + 547888*n**2 - 385039*n**3+ 158306*n**4 - 36911*n**5 + 3767*n**6 + 208*n**7 - 87*n**8 + 6*n**9)*delta**2
+ 6*(-4 + n)*(-1 + n)**3*(-3 + 2*n)*(6 - 5*n + n**2)**2*(-1494 + 2874*n - 2121*n**2 + 758*n**3 - 132*n**4 + 9*n**5)*delta**3)*sfs_0[3])/((-2 + n)*(-1 + n)**3*n*(-7 + 2*n)*(-5 + 2*n)*(-3 + 2*n)*(5 - 5*n + n**2)*(26 - 18*n + 3*n**2)*(11 - 12*n + 3*n**2)) + (delta*(-12*(3 - 2*n)**2*(-4 + n)**4*(-3 + n) *(11 - 12*n + 3*n**2)**2 - 12*(-4 + n)**4*(-1 + n)*(9 - 9*n + 2*n**2)*(-803 + 2196*n - 2363*n**2 + 1249*n**3 - 324*n**4 + 33*n**5)*delta- 24*(-4 + n)**4*(-1 + n)**2*(-3 + 2*n)*(6 - 5*n + n**2)*(143 - 299*n + 228*n**2 - 75*n**3 + 9*n**4)*delta**2 - 12*(-4 + n)**4*(-1 + n)**3*(-3 + 2*n)*(6 - 5*n + n**2)**2*(11 - 12*n + 3*n**2)*delta**3)*sfs_0[4])/((-2 + n)*(-1 + n)**3*n*(-7 + 2*n)*(-5 + 2*n)*(-3 + 2*n)*(5 - 5*n + n**2)*(26 - 18*n + 3*n**2)*(11 - 12*n + 3*n**2))
    return missed

def predict_jack(sfs, cts, M, order):
    '''
        Input :
            sfs array of lenght n+1; sfs[1] is  # things observed once
            M <int> extrapolation size
            order <int> jackknife order
        Output :
            missed <array of len M>
    '''
    N = len(sfs)
    sfs = np.concatenate([[0], sfs])
    preds = np.zeros(len(cts))
    preds[:N+1] = cts[:N+1]
    preds[N+1:] = cts[N+1] + np.asarray([missed(sfs, m, order) for m in range(1,M)])
    return preds