import numpy as np
import matplotlib.pyplot as plt
import math
import random as rnd
import json

J=2
#mu=2.5*J #trivial
mu=0.5*J #topological
n=2

delta=J

gamma_l=0.2 
gamma_g=0.2

def read_data(file_name):
    f1=open(file_name, 'r')
    lines=f1.readlines()
    f1.close()
    data=[]
    for i in range(1, len(lines), 2):
        data.append(json.loads(lines[i]))
    data=np.asarray(data)
    return data

def calculate_mean(minimum_gap, N, len_s):
    mean=np.zeros((len_s, N))
    for j in range(len_s):
        for k in range(N):
            mean[j, k]=np.mean(minimum_gap[j, k])
    return mean

def calculate_variance(minimum_gap, N, len_s):
    variance=np.zeros((len_s, N))
    for j in range(len_s):
        for k in range(N):
            variance[j, k]=np.var(minimum_gap[j, k])
    return variance

def calculate_typical_value(minimum_gap, N, len_s):
    typical_value=np.zeros((len_s, N))
    N_iter=50
    for i in range(len_s):
        for k in range(N):
            mean_log=0
            for j in range(N_iter):
                mean_log+=np.log(minimum_gap[i,k,j])/N_iter
            typical_value[i, k]=np.exp(mean_log)
    return typical_value

minimum_gap_N2_topological=read_data('Data_topological_N=2.txt')
minimum_gap_N2_trivial=np.asarray(read_data('Data_trivial_N=2.txt'))
    
minimum_gap_N4_topological=np.asarray(read_data('Data_topological_N=4.txt'))
minimum_gap_N4_trivial=np.asarray(read_data('Data_trivial_N=4.txt'))

minimum_gap_N6_topological=np.asarray(read_data('Data_topological_N=6.txt'))
minimum_gap_N6_trivial=np.asarray(read_data('Data_trivial_N=6.txt'))

s=np.asarray([1, 3, 5, 10])
t_max_f=5
t_max=np.arange(0.01, t_max_f, 0.01)

mean_N2_topological=calculate_mean(minimum_gap_N2_topological, len(t_max), len(s))
mean_N4_topological=calculate_mean(minimum_gap_N4_topological, len(t_max), len(s))
mean_N6_topological=calculate_mean(minimum_gap_N6_topological, len(t_max), len(s))

variance_N2_topological=calculate_variance(minimum_gap_N2_topological, len(t_max), len(s))
variance_N4_topological=calculate_variance(minimum_gap_N4_topological, len(t_max), len(s))
variance_N6_topological=calculate_variance(minimum_gap_N6_topological, len(t_max), len(s))

typical_value_N2_topological=calculate_typical_value(minimum_gap_N2_topological, len(t_max), len(s))
typical_value_N4_topological=calculate_typical_value(minimum_gap_N4_topological, len(t_max), len(s))
typical_value_N6_topological=calculate_typical_value(minimum_gap_N6_topological, len(t_max), len(s))

mean_N2_trivial=calculate_mean(minimum_gap_N2_trivial, len(t_max), len(s))
mean_N4_trivial=calculate_mean(minimum_gap_N4_trivial, len(t_max), len(s))
mean_N6_trivial=calculate_mean(minimum_gap_N6_trivial, len(t_max), len(s))

variance_N2_trivial=calculate_variance(minimum_gap_N2_trivial, len(t_max), len(s))
variance_N4_trivial=calculate_variance(minimum_gap_N4_trivial, len(t_max), len(s))
variance_N6_trivial=calculate_variance(minimum_gap_N6_trivial, len(t_max), len(s))

typical_value_N2_trivial=calculate_typical_value(minimum_gap_N2_trivial, len(t_max), len(s))
typical_value_N4_trivial=calculate_typical_value(minimum_gap_N4_trivial, len(t_max), len(s))
typical_value_N6_trivial=calculate_typical_value(minimum_gap_N6_trivial, len(t_max), len(s))


colors=['b', 'g', 'r', 'm']

#mean_topological:
for i in range(len(s)):
    plt.plot(t_max, mean_N2_topological[i], label='s='+str(s[i])+'; N=2', color=colors[i])
    plt.plot(t_max, mean_N4_topological[i], label='s='+str(s[i])+'; N=4', linestyle='--', color=colors[i])
    plt.plot(t_max, mean_N6_topological[i], label='s='+str(s[i])+'; N=6', linestyle=':', color=colors[i])

#variance_topological
#for i in range(len(s)):
#    plt.plot(t_max, variance_N2_topological[i], label='s='+str(s[i])+'; N=2', color=colors[i])
#    plt.plot(t_max, variance_N4_topological[i], label='s='+str(s[i])+'; N=4', linestyle='--', color=colors[i])
#    plt.plot(t_max, variance_N6_topological[i], label='s='+str(s[i])+'; N=6', linestyle=':', color=colors[i])

#typical_value_topological
#for i in range(len(s)):
#    plt.plot(t_max, typical_value_N2_topological[i], label='s='+str(s[i])+'; N=2', color=colors[i])
#    plt.plot(t_max, typical_value_N4_topological[i], label='s='+str(s[i])+'; N=4', linestyle='--', color=colors[i])
#    plt.plot(t_max, typical_value_N6_topological[i], label='s='+str(s[i])+'; N=6', linestyle=':', color=colors[i])

#mean_trivial:
#for i in range(len(s)):
#    plt.plot(t_max, mean_N2_trivial[i], label='s='+str(s[i])+'; N=2', color=colors[i])
#    plt.plot(t_max, mean_N4_trivial[i], label='s='+str(s[i])+'; N=4', linestyle='--', color=colors[i])
#    plt.plot(t_max, mean_N6_trivial[i], label='s='+str(s[i])+'; N=6', linestyle=':', color=colors[i])

#variance_trivial
#for i in range(len(s)):
#    plt.plot(t_max, variance_N2_trivial[i], label='s='+str(s[i])+'; N=2', color=colors[i])
#    plt.plot(t_max, variance_N4_trivial[i], label='s='+str(s[i])+'; N=4', linestyle='--', color=colors[i])
#    plt.plot(t_max, variance_N6_trivial[i], label='s='+str(s[i])+'; N=6', linestyle=':', color=colors[i])

#typical_value_trivial
#for i in range(len(s)):
#    plt.plot(t_max, typical_value_N2_trivial[i], label='s='+str(s[i])+'; N=2', color=colors[i])
#    plt.plot(t_max, typical_value_N4_trivial[i], label='s='+str(s[i])+'; N=4', linestyle='--', color=colors[i])
#    plt.plot(t_max, typical_value_N6_trivial[i], label='s='+str(s[i])+'; N=6', linestyle=':', color=colors[i])

#mean and typival value, topological
#for i in range(len(s)):
#    plt.plot(t_max, mean_N6_topological[i], label='s='+str(s[i])+r'; $\mu$', color=colors[i])
#    plt.plot(t_max, typical_value_N6_topological[i], label='s='+str(s[i])+r'; $exp(<log(E)>)$', linestyle='--', color=colors[i])

#mean and typical value, trivial
#for i in range(len(s)):
#    plt.plot(t_max, mean_N6_trivial[i], label='s='+str(s[i])+r'; $\mu$', color=colors[i])
#    plt.plot(t_max, typical_value_N6_trivial[i], label='s='+str(s[i])+r'; $exp(<log(E)>)$', linestyle='--', color=colors[i])


zeros=np.zeros(len(t_max))
plt.plot(t_max, zeros, color='k')
plt.xlabel(r'$t_{max}$', fontsize=18)

plt.ylabel(r'$\tau$', fontsize=18) #mean
#plt.ylabel(r'$\sigma^2$', fontsize=18) #variance
#plt.ylabel(r'$exp(<log(E)>)$', fontsize=18) #typical value
#text_mean:
plt.text(1, 0.9, r'$\mu=$'+str(mu)+r'$; \Delta=$'+str(delta)+ r'$; J=$'+str(J)+r'$; \gamma_g=$'+str(gamma_g)+r'$; \gamma_l=$'+str(gamma_l), fontsize=18)
#text variance:
#plt.text(1, 0.06, r'$\mu=$'+str(mu)+r'$; \Delta=$'+str(delta)+ r'$; J=$'+str(J)+r'$; \gamma_g=$'+str(gamma_g)+r'$; \gamma_l=$'+str(gamma_l), fontsize=18)
#text_typical value:
#plt.text(1, 0.9, r'$\mu=$'+str(mu)+r'$; \Delta=$'+str(delta)+ r'$; J=$'+str(J)+r'$; \gamma_g=$'+str(gamma_g)+r'$; \gamma_l=$'+str(gamma_l), fontsize=18)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(loc='upper right', fontsize=16)
plt.show()


