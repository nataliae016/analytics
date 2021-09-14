import numpy as np
import matplotlib.pyplot as plt
import math
import random as rnd
import scipy.sparse as sparse
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment




J=2
#mu=2.5*J #trivial
mu=0.5*J #topological

delta=J



gamma_l=0.2 
gamma_g=0.2



mu_0=100

n=2
L=2*n 
D=2**n 



def n_to_bin(s, L):
    a=list(reversed(bin(s)[2:].zfill(L)))
    for i in range(len(a)):
       a[i]=int(a[i])
    a=np.asarray(a)
    return a

def bin_to_n(n):
    a=list(reversed(n))
    for i in range(len(a)):
        a[i]=str(a[i])
    return int(''.join(a), 2)

def c_i(i, s, L):
    alpha=n_to_bin(s, L)
    if alpha[i]==0:
        return 0
    else:
        phase=(-1)**(np.sum(alpha[:i]))
        new_state=s+(-1**alpha[i])*2**i
        return phase, new_state

def cdagger_i(i, s, L):
    alpha=n_to_bin(s, L)
    if alpha[i]==1:
        return 0
    else:
        phase=(-1)**(np.sum(alpha[:i]))
        new_state=s+((-1)**alpha[i])*2**i
        return phase, new_state

def bath_matrix(l, n, L):
    M=np.zeros((L, L), dtype=complex)
    for j in range(L):
        for k in range(L):
            for m in range(n):
                M[j,k]+=l[m, j]*np.conjugate(l[m, k])
    return M

def psi_init(N, L, D):
    psi=np.zeros((N, 2**(L)), dtype=complex)
    psi[0, 0]=1/D
    for j in range(1, 2**L):
        power=0
        j_bin=n_to_bin(j, L)
        n_w=np.sum(j_bin)
        for i in range(1, n_w):
            power+=i
        r=rnd.uniform(0, 1)
        if r<=1:
            if power%2==0:
                psi[0, j]=rnd.uniform(-1/(D**3-D**2), 1/(D**3-D**2))
            else:
                psi[0, j]=1j*rnd.uniform(-1/(D**3-D**2), 1/(D**3-D**2))
        else:
            pass
    return psi

def psi_init_ground(n, N, L):
    psi=np.zeros((N, 2**(L)), dtype=complex)
    psi_product=np.asarray([1/2, 0, 0, 1j/2])
    for j in range(1, n):
        psi_product=np.kron(np.asarray([1/2, 0, 0, 1j/2]), psi_product)
    psi[0]=psi_product
    return psi


def psi_init_thermal(n, N, L, beta, mu_0):
    psi=np.zeros((N, 2**(L)), dtype=complex) 
    Z=(1/2)*np.exp(-(1/2)*beta*mu_0*n)*(np.exp(beta*mu_0)+1)*np.trace(np.identity(2)) #partition function
    psi_0=np.exp(-(1/2)*beta*mu_0*n)*np.asarray([(1/2)*(np.exp(beta*mu_0)+1), 0, 0, -1j/2*(np.exp(beta*mu_0)-1)])
    psi_0=psi_0/Z
    psi_product=psi_0
    for j in range(1, n):
        psi_product=np.kron(psi_0, psi_product)
    psi[0]=psi_product
    return psi




sigma_plus=np.asarray([[0, 1], [0, 0]])
sigma_minus=np.asarray([[0, 0], [1, 0]])


def fermions(n, D):
    f=np.zeros((n, D, D), dtype=complex)
    fdagger=np.zeros((n, D, D), dtype=complex)
    I=np.identity(2)

    for j in range(n):
        if j==0:
            f_j=sigma_minus
            fdagger_j=sigma_plus
        else:
            f_j=I
            fdagger_j=I
        for i in range(1, n):
            if i==j:
                f_j=np.kron(f_j, sigma_minus)
                fdagger_j=np.kron(fdagger_j, sigma_plus)
            else:
                f_j=np.kron(f_j, I)
                fdagger_j=np.kron(fdagger_j, I)
        f[j]=f_j
        fdagger[j]=fdagger_j

    a=np.zeros((n, D, D), dtype=complex)
    adagger=np.zeros((n, D, D), dtype=complex)

    for j in range(n):
        N_f=np.zeros((D, D), dtype=complex)
        for k in range(0, j):
            N_f+=np.matmul(fdagger[k], f[k])
        a[j]=np.matmul(expm(1j*np.pi*N_f), f[j])
        adagger[j]=np.matmul((expm(-1j*np.pi*N_f)), fdagger[j])
    return a, adagger

def majoranas(a, adagger, n, L, D):
    w=np.zeros((L, D, D), dtype=complex)
    for j in range(n):
        w[2*j]=a[j]+adagger[j]
        w[2*j+1]=1j*(a[j]-adagger[j])
    return w


def psi_to_rho(psi, L, D, w):
    sum=np.zeros((D, D), dtype=complex)
    for l in range (2**L):
        alpha=n_to_bin(l, L)
        product=np.linalg.matrix_power(w[L-1],alpha[L-1])
        if psi[l]==0:
            continue
        for k in range((L-2), -1, -1):
            product=np.matmul(np.linalg.matrix_power(w[k],alpha[k]), product)
        sum+=psi[l]*product
    return sum


def time_evolution(n, L, D, s, N_final):
    step=0.001
    steps=np.arange(0, N_final, step) 
    N=len(steps) 
    A=np.zeros((L, L), dtype=complex) 
    l=np.zeros((n, L), dtype=complex) 



    for i in range(n):
        epsilon=rnd.uniform(-s, s) #disorder
        A[2*i+1, 2*i]=-(mu+epsilon)
        A[2*i, 2*i+1]=mu+epsilon
    for i in range(n-1):
        A[2*i+1, 2*i+2]=-J-delta
        A[2*i, 2*i+3]=J-delta
        A[2*i+3, 2*i]=-J+delta
        A[2*i+2, 2*i+1]=J+delta


    for i in range(n):
        l[i, 2*i]=0.5*(np.sqrt(gamma_l)+np.sqrt(gamma_g))
        l[i, 2*i+1]=0.5*1j*(-np.sqrt(gamma_l)+np.sqrt(gamma_g))



    M=np.matmul(np.transpose(l), np.conjugate(l)) 
    M_R=np.real(M) 
    M_I=np.imag(M) 

    psi=psi_init_ground(n, N, L)
    col_ind=[] 
    row_ind=[] 
    data=[] 


    for alpha in range(0, (2**L)):
        for j in range(0, L):
            for k in range(0, L):
                try:
                    phase01, beta_01=c_i(k, alpha,L)
                    phase1, beta_1=cdagger_i(j, beta_01, L)
                    value1=phase1*phase01*A[j, k]
                    if value1!=0:
                        row_ind.append(beta_1)
                        col_ind.append(alpha)
                        data.append(value1)
                    else:
                        pass
                except:
                    pass

    L_0=sparse.coo_matrix((data, (row_ind, col_ind)), shape=(2**L, 2**L))
    L_0=L_0.tocsr()

    #Dissipator part

    col_ind=[] 
    row_ind=[] 
    data=[] 

    for alpha in range(0, (2**L)):
        numb=np.sum(n_to_bin(alpha, L))
        if numb%2==0:
            even=True
        else:
            even=False
        for j in range(0, L):
            for k in range(0, L):
                try:
                    if even:
                        phase01, beta_01=cdagger_i(k, alpha,L)
                        phase1, beta_1=cdagger_i(j, beta_01, L)
                    else:
                        phase01, beta_01=c_i(k, alpha,L)
                        phase1, beta_1=c_i(j, beta_01, L)
                    value1=4*1j*phase1*phase01*M_I[j, k]
                    if value1!=0:
                        row_ind.append(beta_1)
                        col_ind.append(alpha)
                        data.append(value1)
                    else:
                        pass
                except:
                    pass
                try:
                    if even:
                        phase02, beta_02=c_i(k, alpha,L)
                        phase2, beta_2=cdagger_i(j, beta_02, L)
                    else:
                        phase02, beta_02=cdagger_i(k, alpha,L)
                        phase2, beta_2=c_i(j, beta_02, L)
                    value2=-4*phase2*phase02*M_R[j, k]
                    if value2!=0:
                        row_ind.append(beta_2)
                        col_ind.append(alpha)
                        data.append(value2)
                    else:
                        pass
                except:
                    pass

    Dissipator=sparse.coo_matrix((data, (row_ind, col_ind)), shape=(2**L, 2**L))
    Dissipator=Dissipator.tocsr()

    Lindbladian=L_0+Dissipator 




    a, adagger=fermions(n, D)
    w=majoranas(a, adagger, n, L, D)
    rho0=psi_to_rho(psi[0], L, D, w)
    for i in range(0, N-1):
        psi[i+1]=psi[i]+step*sparse.csr_matrix.dot(Lindbladian, psi[i])
    return psi

def reduced_rho(psi):
    sum=np.zeros((D_half, D_half), dtype=complex)
    TrR=np.trace(np.identity(D_half))
    for l in range (2**L_half):
        alpha=np.zeros((L), dtype=int)
        alpha[:L_half]=n_to_bin(l, L_half)
        index=bin_to_n(alpha)
        alpha=n_to_bin(l, L_half)
        product=np.linalg.matrix_power(reduced_w[L_half-1],alpha[L_half-1])
        for k in range((L_half-2), -1, -1):
            product=np.matmul(np.linalg.matrix_power(reduced_w[k],alpha[k]), product)
        sum+=psi[index]*TrR*product
    return sum


n_half=int(n/2) #half of fermion chain
L_half=int(2*n_half) #half of corresponding Majorana chain

D_half=2**n_half #Hilbert space dimension of half of the chain

a, adagger=fermions(n_half, D_half)
reduced_w=majoranas(a, adagger, n_half, L_half, D_half)




N_iter=50
t_max_f=5
t_max=np.arange(0.01, t_max_f, 0.01)
s_list=np.asarray([1, 3, 5, 10])
file1=open('Data_topological_N=2.txt', 'w')
for s in s_list:
    minimum_gap=np.full((len(t_max), N_iter), 1000, dtype=float)
    for iter in range(N_iter):
        psi=time_evolution(n, L, D, s, t_max_f)
        spectrum=np.zeros((len(psi), 2**n_half))
        eig0, eig_vec0=np.linalg.eigh(reduced_rho(psi[0]))
        assignment0=np.arange(0, 2**n_half, 1)
        spectrum[0]=eig0
        for i in range(1, len(psi)):
            rho_L=reduced_rho(psi[i])
            eig, eig_vec=np.linalg.eigh(rho_L)
            overlap=np.abs(np.matmul(np.transpose(np.conj(eig_vec0)), eig_vec))
            assignment=linear_sum_assignment(-overlap)[1]
            for j in range(len(assignment)):
                eig0[j]=eig[assignment[j]]
                eig_vec0[:, j]=eig_vec[:, assignment[j]]
                spectrum[i]=eig0
        for k in range(len(t_max)):
            minimum_gap_i=1000
            for i in range(0, int(t_max[k]/0.001)):
                gap=abs(spectrum[i, -1]-spectrum[i, -2])
                if gap<minimum_gap_i:
                    minimum_gap_i=gap
                else:
                    pass
            minimum_gap[k, iter]=minimum_gap_i
    file1.write('s:'+"\n")
    file1.write(str(s)+"\n")
    file1.write('minimum_gap_data'+ '\n')
    file1.write(str(minimum_gap.tolist())+'\n')

file1.close()



