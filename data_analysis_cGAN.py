import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import fsolve
import csv

myfile = open('accelerated_analysis_results.csv', 'w')
writer = csv.writer(myfile)
writer.writerow(["m_sq", "lmbd", "alpha", "mpi_from_fpi2", "mpi_from_fpi_err2"])

def get_jackknife_blocks(data,block_size,f=lambda x:x):
    N = int(len(data)/block_size)
    data_mean = np.mean(data,axis=0)*N*block_size
    block_avgs = []
    for i in range(N):
        block_av=np.copy(data_mean)
        for j in range(block_size):
            block_av -= data[i*block_size+j]
        block_av /= (N-1)*block_size
        block_avgs.append(f(block_av))
    return block_avgs

def get_errors_from_blocks(est_value,blocks):
    N = len(blocks)
    err = 0
    bias = 0
    for i in range(N):
        err = np.add(err, (N-1)/N*np.power(np.subtract(est_value,blocks[i]),2))
        bias = np.add(bias,np.divide(blocks[i],N))
    err = np.power(err,0.5)
    return [np.add(est_value,np.multiply(N-1,np.subtract(est_value,bias))), err]

def jackknife(data,f=np.mean):
    N = len(data)
    err = 0
    bias = 0
    data_f = f(data)
    for i in range(N):
        d = data.pop(i)
        omit_f = f(data)
        err = np.add(err, (N-1)/N*np.subtract(data_f,omit_f)**2)
        bias = np.add(bias,np.divide(omit_f,N))
        data.insert(i, d)
    err = np.power(err,0.5)
    return [np.add(data_f,np.multiply(N-1,np.subtract(data_f,bias))), err]

def jackknife2(data1,data2,f):
    N = len(data1)
    err = 0
    bias = 0
    data_f = f(data1,data2)
    for i in range(N):
        d1 = data1.pop(i)
        d2 = data2.pop(i)
        omit_f = f(data1,data2)
        err = np.add(err, (N-1)/N*np.subtract(data_f,omit_f)**2)
        bias = np.add(bias,np.divide(omit_f,N))
        data1.insert(i, d1)
        data2.insert(i, d2)
    err = np.power(err,0.5)
    return [np.add(data_f,np.multiply(N-1,np.subtract(data_f,bias))), err]

def get_obs_avg(data, cutoff=0, corr_dist=1):
    avg = np.mean([data[i] for i in range(cutoff, len(data), corr_dist)])
    err = np.std([data[i] for i in range(cutoff, len(data), corr_dist)])/((len(data)-cutoff)/corr_dist)**0.5
    return [avg, err]

def get_obs_avg_jackknife(data, cutoff=0, block_size=1):
    return jackknife([data[i] for i in range(cutoff, len(data))], block_size, np.mean)

def get_correlator_avgs(corrs, cutoff=0, corr_dist=1):
    avgs = [np.mean([corrs[i][j] for i in range(cutoff,len(corrs),corr_dist)]) for j in range(len(corrs[0]))]
    errs = [np.std([corrs[i][j] for i in range(cutoff,len(corrs),corr_dist)])/((len(corrs)-cutoff)/corr_dist)**0.5 for j in range(len(corrs[0]))]
    # Because of periodic boundary conditions, the last correlator should be equal to the first
    avgs.append(avgs[0])
    errs.append(errs[0])
    return [avgs, errs]

def get_correlator_avgs_jackknife(corrs, cutoff=0, corr_dist=1):
    avgs = []
    errs = []
    for i in range(len(corrs[0])):
        a, b = jackknife([corrs[j][i] for j in range(cutoff, len(corrs), corr_dist)], np.mean)
        avgs.append(a)
        errs.append(b)
    # Because of periodic boundary conditions, the last correlator should be equal to the first
    avgs.append(avgs[0])
    errs.append(errs[0])
    return [avgs, errs]

cosh_model = lambda nt,A0,m_pi : [A0*np.cosh((Nt/2.0-nt[i])*m_pi) for i in range(len(nt))]

def find_mass_from_fit(corr_avgs, fit_range_start, mass_guess=0.3):
    Nt = len(corr_avgs) - 1
    nt = range(fit_range_start,int(Nt)+1-fit_range_start)
    A0_guess = .1#np.abs(corr_avgs[int(Nt/2)])
    a = corr_avgs[fit_range_start:int(Nt)+1-fit_range_start]
    pi_opt, pi_cov = curve_fit(cosh_model, nt, corr_avgs[fit_range_start:int(Nt)+1-fit_range_start], 
                               p0=[A0_guess,mass_guess])
    return [np.abs(pi_opt[1]), pi_opt[0]]

def autocorr(data):
    N = len(data)
    mean = np.mean(data)
    variance = np.var(data)
    data = np.subtract(data,mean)
    r = np.correlate(data, data, mode = 'full')[-N:]
    assert np.allclose(r, np.array([np.sum(np.multiply(data[:N-k],data[-(N-k):])) for k in range(N)]))
    result = r/(variance*(np.arange(N, 0, -1)))
    return result

def fpi_jacknife2(pi_corrs):
    pi_corr_avgs = [np.mean([pi_corrs[i][j] for i in range(len(pi_corrs))]) for j in range(len(pi_corrs[0]))]
    return fpi_fit2(pi_corr_avgs, 0)

def jacknife(data,f):
    N = len(data)
    err = 0
    bias = 0
    data_f = f(data)
    for i in range(N):
        d = data.pop(i)
        omit_f = f(data)
        err = np.add(err, (N-1)/N*np.subtract(data_f,omit_f)**2)
        bias = np.add(bias,np.divide(omit_f,N))
        data.insert(i, d)
    err = np.power(err,0.5)
    return [np.add(data_f,np.multiply(N-1,np.subtract(data_f,bias))), err]



# Extent of the lattice
Nx = 8
Nt = 16
# Spacial volume of the lattice
Vx = Nx**3
# The parameters in the action

# The number of trajectories until thermalization
cutoff = 0
# The size of the blocks needed to get uncorrelated block averages
block_size = 20
parameters = [[m,l,a] for m in [-20550.0] for l in [100000.0] for a in [0.0015, 0.0016]]
for [m_sq, lmbd, alpha] in parameters:
    print("msq: "+str(m_sq)+", lmbd: "+str(lmbd)+", alpha: "+str(alpha))
    phi_list = [[float(0) for i in range(4)] for j in range(400)]
    timeslices = [[[float(0) for i in range(4)] for j in range(16)] for k in range(400)]
    # Load the data
    print("Loading Configurations...")
    for round in range(400):
        lattice = np.load(f"cGAN_pions_{round}_8x16_msq_{m_sq}_lmbd_{lmbd}_alph_{alpha}.npy")
        for m in range(4):
            for i in range(8):
                for j in range(8):
                    for k in range(8):
                        for l in range(16):
                            phi_list[round][m]
                            lattice[m][i][j][k][l]
                            phi_list[round][m] = phi_list[round][m] + lattice[m][i][j][k][l]
        for m in range(4):
            phi_list[round][m] =phi_list[round][m]/(8*8*8*16)
        
        for m in range(4):
            for i in range(8):
                for j in range(8):
                    for k in range(8):
                        for l in range(16):
                            timeslices[round][l][m] = timeslices[round][l][m] + lattice[m][i][j][k][l]
        for m in range(4):
            for l in range(16):
                timeslices[round][l][m] =timeslices[round][l][m]/(8*8*8)
        if(round % 100 == 0 and round != 0):
            print(" loaded "+str(round)+" configurations")
        
    # Plot an observable

    def pion_correlator(tslices,Vx,Nt,delta_t):
        rtn = 0
        for t0 in range(Nt):
            rtn += (tslices[t0%Nt][1]*tslices[(t0+delta_t)%Nt][1] + 
                    tslices[t0%Nt][2]*tslices[(t0+delta_t)%Nt][2] + 
                    tslices[t0%Nt][3]*tslices[(t0+delta_t)%Nt][3])/3.0
        return rtn/Nt
    end = len(timeslices)
    pi_corrs = [[pion_correlator(timeslices[i],Vx,Nt,dt) for dt in range(Nt)] for i in range(cutoff,end)]
    pi_corr_avgs = np.mean(pi_corrs,axis=0)
    f = lambda data : find_mass_from_fit(data, 0, 1.0)

    end = len(phi_list)
    vev_sigma_blocks = get_jackknife_blocks([phi_list[i][0] for i in range(cutoff,end)], block_size)
    vev_sigma, vev_sigma_err = get_errors_from_blocks(np.mean(vev_sigma_blocks),vev_sigma_blocks)

    def pion_correlator(tslices,Vx,Nt,delta_t):
        rtn = 0
        for t0 in range(Nt):
            rtn += (tslices[t0%Nt][1]*tslices[(t0+delta_t)%Nt][1] + 
                    tslices[t0%Nt][2]*tslices[(t0+delta_t)%Nt][2] + 
                    tslices[t0%Nt][3]*tslices[(t0+delta_t)%Nt][3])/3.0
        return rtn/Nt
    end = len(timeslices)
    #pi_corrs = [[pion_correlator(timeslices[i],Vx,Nt,dt) for dt in range(Nt)] for i in range(cutoff,end)]
    #pi_corr_avgs = np.mean(pi_corrs,axis=0)
    def sigma_correlator(tslices,Vx,Nt,vev_sigma,delta_t):
        rtn = 0
        for t0 in range(Nt):
            rtn += (tslices[t0%Nt][0]/Vx-vev_sigma)*(tslices[(t0+delta_t)%Nt][0]/Vx-vev_sigma)
        return rtn/Nt

    end=len(timeslices)
    s_corrs = [[sigma_correlator(timeslices[i],Vx,Nt,vev_sigma,dt) for dt in range(Nt)] for i in range(cutoff,end)]
    s_corr_avgs = np.mean(s_corrs,axis=0)

    f = lambda data : find_mass_from_fit(data, 0, 1.0)
    end = len(s_corrs)


    end = len(timeslices)


    def f(A_corr_avgs):
        return A_corr_avgs[1:int(Nt)]


    fpi_model2 = lambda nt,A0,m_pi,alpha : [-A0/alpha*(m_pi*Vx*np.exp(-m_pi*Nt/2))**0.5*(np.sinh((Nt/2.0-nt[i]-1/2)*m_pi)-
                                                                                        np.sinh((Nt/2.0-nt[i]+1/2)*m_pi))/
                                        (np.cosh((Nt/2.0-nt[i])*m_pi))**0.5 for i in range(len(nt))]

    def fpi_fit2(pi_corr_avgs, fit_range_start=0, A0_guess=2.0, mpi_guess=0.2):
        Nt = len(pi_corr_avgs) - 1
        nt = range(fit_range_start,int(Nt)+1-fit_range_start)
        for i in range(len(pi_corr_avgs)):
            if pi_corr_avgs[i]<0:
                pi_corr_avgs[i]=0
        a = np.power(pi_corr_avgs[fit_range_start:int(Nt)+1-fit_range_start], 0.5)
        a = np.nan_to_num(a)
        pi_opt, pi_cov = curve_fit(lambda nt,A0,pi_mass : fpi_model2(nt,A0,pi_mass,alpha), nt, 
                                a, p0=[A0_guess, mpi_guess])
        return pi_opt

    try:
        fpi_blocks2 = get_jackknife_blocks(pi_corrs, block_size, fpi_fit2)
        [[fpi2, mpi_from_fpi2],[fpi_err2,mpi_from_fpi_err2]] = get_errors_from_blocks(fpi_fit2(pi_corr_avgs), fpi_blocks2)
    except:
        [[fpi2, mpi_from_fpi2],[fpi_err2,mpi_from_fpi_err2]] = [["error","error"],["error","error"]]

    print(f"blocked m_pi is {mpi_from_fpi2}/a +- {mpi_from_fpi_err2}/a")

    print()

    pi_corr_avg_blocks = get_jackknife_blocks(pi_corrs,block_size)
    pi_corr_avgs_bc, pi_corr_errs = get_errors_from_blocks(pi_corr_avgs, pi_corr_avg_blocks)
    s_corr_avg_blocks = get_jackknife_blocks(s_corrs,block_size)
    s_corr_avgs_bc, s_corr_errs = get_errors_from_blocks(s_corr_avgs, s_corr_avg_blocks)

    writer.writerow([m_sq, lmbd, alpha, mpi_from_fpi2, mpi_from_fpi_err2])

myfile.close()