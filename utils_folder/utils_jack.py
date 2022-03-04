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

def missed_jack(N, M, sfs, order):
    '''
        Input :
            sfs array; sfs[0] is  # things observed once
            M <int> extrapolation size
            order <int> jackknife order
        Output :
            missed <array of len M>
    '''
    assert order in [1,2,3,4], 'Order not yet implemented'
    if N == 2:
        order = min(order, 3) # this is because of numerical stability; order 4 not stable for N = 1, 2.
    if N == 1:
        order = 1
    sfs_0 = np.concatenate([[0], sfs])
    lsfs_0 = len(sfs_0)
    if lsfs_0 < order+1:
        sfs_0 = np.concatenate([sfs_0, np.zeros(order+1-lsfs_0)])
    delta = deltaf(M,N)
    
    if order == 1:
        missed = (-1.+N) / N * delta * sfs_0[1]
    elif order == 2:
        missed = (((1+2*(-2+N)*N)*delta)/(N*(-3+2*N))+((-2+N)*(-1+N)*delta**2)/(N*(-3+2*N)))*sfs_0[1]+(-((2*(-2+N)**2*delta)/((-1+N)*N*(-3+2*N)))-(2*(-2+N)**2*delta**2)/(N*(-3+2*N)))*sfs_0[2]
    elif order == 3:
        N = float(N)
        missed = (delta*(-127+628*N-1386*N**2+1722*N**3-1269*N**4+546*N**5-126*N**6+12*N**7+(-1+N)*(-10-87*N+310*N**2-372*N**3+210*N**4-57*N**5+6*N**6)*delta+2*(-2+N)**2*(-1+N)**3*(6-5*N+N**2)*delta**2)*sfs_0[1])/((-1+N)**2*N*(-5+2*N)*(-3+2*N)*(11-12*N+3*N**2))+(delta*(-2*(-248+402*N-61*N**2-249*N**3+195*N**4-57*N**5+6*N**6)-2*(-1+N)*(-560+1130*N-731*N**2+55*N**3+127*N**4-51*N**5+6*N**6)*delta+2*(-1+N)**2*(6-5*N+N**2)*(52-78*N+38*N**2-6*N**3)*delta**2)*sfs_0[2])/((-1+N)**2*N*(-5+2*N)*(-3+2*N)*(11-12*N+3*N**2))+(delta*(6*(3-2*N)**2*(-3+N)**3+6*(-3+N)**3*(-1+N)*(15-19*N+6*N**2)*delta+6*(-3+N)**2*(-1+N)**2*(-3+2*N)*(6-5*N+N**2)*delta**2)*sfs_0[3])/((-1+N)**2*N*(-5+2*N)*(-3+2*N)*(11-12*N+3*N**2))
    elif order == 4:
        N = float(N)
        missed = (delta*(2*(79138 - 768163*N + 3305209*N**2 - 8455556*N**3 + 14456396*N**4 - 17528889*N**5 + 15570611*N**6 - 10302336*N**7 + 5104884*N**8 - 1885839*N**9 + 511449*N**10 - 98781*N**11 + 12849*N**12 - 1008*N**13 + 36*N**14) + (-1 + N)*(-142984 + 984880*N - 3219708*N**2 + 6550921*N**3 - 9150649*N**4  + 9174455*N**5 - 6742096*N**6 + 3659027*N**7 - 1462263*N**8 + 424335*N**9  - 86856*N**10 + 11874*N**11 - 972*N**12 + 36*N**13)*delta + 2*(-1 + N)**2*(6 - 5*N + N**2)*(-2288 + 3592*N + 10090*N**2 - 36645*N**3 + 50160*N**4 - 39031*N**5 + 19017*N**6 - 5922*N**7 + 1147*N**8 - 126*N**9 + 6*N**10)*delta**2 + (-4 + N)*(-1 + N)**3*(6 - 5*N + N**2)**2*(184 - 606*N + 803*N**2 - 549*N**3  + 204*N**4 - 39*N**5 + 3*N**6)*delta**3)*sfs_0[1])/((-2 + N)*(-1 + N)**3*N *(-7 + 2*N)*(-5 + 2*N)*(-3 + 2*N)*(5 - 5*N + N**2)*(26 - 18*N + 3*N**2)*(11- 12*N + 3*N**2)) + (delta*(2*(806336 - 4114800*N + 9788448*N**2 - 14623884*N**3 + 15584978*N**4 - 12651005*N**5 + 8033192*N**6 - 3985285*N**7 + 1516278*N**8- 429717*N**9 + 87120*N**10 - 11877*N**11 + 972*N**12 - 36*N**13)- 2*(-1 + N)*(-1762976 + 8275744*N - 17540568*N**2 + 22472256*N**3 - 19868590*N**4 + 13296025*N**5 - 7229740*N**6 + 3298055*N**7 - 1236311*N**8 + 360392*N**9 - 76491*N**10 + 10944*N**11 - 936*N**12 + 36*N**13)*delta - 4*(-1 + N)**2*(6 - 5*N + N**2)*(-117832 + 426816*N - 645654*N**2 + 513018*N**3 - 207385*N**4 + 15319*N**5 + 25646*N**6 - 13195*N**7 + 3075*N**8 - 366*N**9 + 18*N**10)*delta**2 - 2*(-4 + N)*(-1 + N)**3*(6 - 5*N + N**2)**2*(3176 - 8768*N+ 9854*N**2 - 5764*N**3 + 1850*N**4 - 309*N**5 + 21*N**6)*delta**3)*sfs_0[2])/((-2 + N)*(-1 + N)**3*N*(-7 + 2*N)*(-5 + 2*N)*(-3 + 2*N)*(5 - 5*N + N**2)*(26 - 18*N + 3*N**2)*(11 - 12*N + 3*N**2)) + (delta*(12*(3 - 2*N)**2*(-3 + N) *(24678 - 73356*N + 88865*N**2 - 54881*N**3 + 16400*N**4 - 464*N**5 - 1289*N**6 + 419*N**7 - 57*N**8 + 3*N**9) + 6*(-1 + N)*(9 - 9*N + 2*N**2)*(-354348 + 1276800*N - 1961648*N**2 + 1662242*N**3 - 829199*N**4 + 230491*N**5- 21616*N**6 - 6925*N**7 + 2631*N**8 - 354*N**9 + 18*N**10)*delta +6*(-1 + N)**2*(-3 + 2*N)*(6 - 5*N + N**2)*(138996 - 425226*N + 547888*N**2 - 385039*N**3+ 158306*N**4 - 36911*N**5 + 3767*N**6 + 208*N**7 - 87*N**8 + 6*N**9)*delta**2
+ 6*(-4 + N)*(-1 + N)**3*(-3 + 2*N)*(6 - 5*N + N**2)**2*(-1494 + 2874*N - 2121*N**2 + 758*N**3 - 132*N**4 + 9*N**5)*delta**3)*sfs_0[3])/((-2 + N)*(-1 + N)**3*N*(-7 + 2*N)*(-5 + 2*N)*(-3 + 2*N)*(5 - 5*N + N**2)*(26 - 18*N + 3*N**2)*(11 - 12*N + 3*N**2)) + (delta*(-12*(3 - 2*N)**2*(-4 + N)**4*(-3 + N) *(11 - 12*N + 3*N**2)**2 - 12*(-4 + N)**4*(-1 + N)*(9 - 9*N + 2*N**2)*(-803 + 2196*N - 2363*N**2 + 1249*N**3 - 324*N**4 + 33*N**5)*delta- 24*(-4 + N)**4*(-1 + N)**2*(-3 + 2*N)*(6 - 5*N + N**2)*(143 - 299*N + 228*N**2 - 75*N**3 + 9*N**4)*delta**2 - 12*(-4 + N)**4*(-1 + N)**3*(-3 + 2*N)*(6 - 5*N + N**2)**2*(11 - 12*N + 3*N**2)*delta**3)*sfs_0[4])/((-2 + N)*(-1 + N)**3*N*(-7 + 2*N)*(-5 + 2*N)*(-3 + 2*N)*(5 - 5*N + N**2)*(26 - 18*N + 3*N**2)*(11 - 12*N + 3*N**2))
    return missed

def predict_jack(N, M, sfs, cts, order):
    '''
        Input :
            sfs array; sfs[0] is  # things observed once
            M <int> extrapolation size
            order <int> jackknife order

    '''
    assert len(sfs)<N+1, 'Too many entries in the sfs; 0-th entry should be # things observed once; last entry # things observed N times'
    preds = np.zeros(N+M+1)
    if len(sfs)<order:
        sfs = np.concatenate([sfs, np.zeros(order - len(sfs))])
    preds[:N+1] = cts[:N+1]
    preds[N+1:] = cts[N] + np.asarray([missed_jack(N, m, sfs, order) for m in range(1,M+1)])
    return preds
