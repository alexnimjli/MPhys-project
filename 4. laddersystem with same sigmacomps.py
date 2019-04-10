import pylab
import scipy
import numpy

from scipy import linalg
from numpy import matrix
from numpy import linalg
import time

def heatcapacity(E_Bvallist, no_eigvals, kb, T):

    Z = 0
    E_av = 0
    E2_av = 0

    for i in range(no_eigvals):
        Z += numpy.exp(-(E_Bvallist[i])/ (kb * T))
        E_av += E_Bvallist[i] * numpy.exp(-(E_Bvallist[i])/ (kb * T))
        E2_av += E_Bvallist[i] * E_Bvallist[i] * numpy.exp(-(E_Bvallist[i])/ (kb * T))
    C = (1/(kb*T*T)) * ((E2_av / Z) - (E_av / Z)*(E_av / Z))
    U = E_av/Z
    return (U, C)

start_time = time.time()

atoms = 2
chains = 1
atoms_p_chain = atoms/chains

I22 = numpy.eye(2)
sigmazlist = []
sigmaxlist = []
sigmaylist = []
sigmaz = numpy.array([[0.5, 0], [0, -0.5]])
sigmax = numpy.array([[0, 0.5], [0.5, 0]])
sigmay = numpy.array([[0, complex(0, -0.5)], [complex(0, 0.5), 0]])

######################
# Here we construct the Pauli spin matrices for n-atom system

for i in range(atoms):
    sigma_nz = 1
    sigma_nx = 1
    sigma_ny = 1
    for j in range(atoms):
        if j == i:
            sigma_nz = numpy.kron(sigma_nz, sigmaz)
            sigma_nx = numpy.kron(sigma_nx, sigmax)
            sigma_ny = numpy.kron(sigma_ny, sigmay)
        else:
            sigma_nz = numpy.kron(sigma_nz, I22)
            sigma_nx = numpy.kron(sigma_nx, I22)
            sigma_ny = numpy.kron(sigma_ny, I22)
    sigmazlist.append(sigma_nz)
    sigmaylist.append(sigma_ny)
    sigmaxlist.append(sigma_nx)

################################

no_eigvals = 2 ** atoms
J_Szcomp = 0
J_Sycomp = 0
J_Sxcomp = 0
zcomptot = 0

J = 1.38e-23 * 5.0  # very small coupling energy?
J_chains = 1.38e-23 * 0.5

# this is for the interactions between each atom in the chain
for k in range(chains):
    for i in range(k*atoms_p_chain,(k+1)*(atoms_p_chain)-1):
        J_Szcomp = J_Szcomp + J * numpy.dot(sigmazlist[i], sigmazlist[i + 1])
        J_Sycomp = J_Sycomp + J * numpy.dot(sigmaylist[i], sigmaylist[i + 1])
        J_Sxcomp = J_Sxcomp + J * numpy.dot(sigmaxlist[i], sigmaxlist[i + 1])
# this is for the interaction between neighbouring chains
for w in range(atoms-atoms_p_chain):
    J_Szcomp = J_Szcomp + J_chains * numpy.dot(sigmazlist[w], sigmazlist[w + atoms_p_chain])
    J_Sycomp = J_Sycomp + J_chains * numpy.dot(sigmaylist[w], sigmaylist[w + atoms_p_chain])
    J_Sxcomp = J_Sxcomp + J_chains * numpy.dot(sigmaxlist[w], sigmaxlist[w + atoms_p_chain])
for i in range(atoms):
    zcomptot = zcomptot + sigmazlist[i]
J_S = J_Szcomp + J_Sycomp + J_Sxcomp

B_axis = []
E_vallist = []
for i in range(no_eigvals):
    E_vallist.append([])

B_field = 10.0
B_interval = 3
T_interval = 200
g = 2.0
mu_b = 9.275e-24

for i in range(B_interval):
    B = (B_field * i) / T_interval
    # H_matrix = - chains * g * mu_b * B * zcomptot + J_S
    H_matrix = - g * mu_b * B * zcomptot + J_S
    Heig_vals, Heig_vecs = numpy.linalg.eig(H_matrix)
    Heig_vals_sorted = numpy.sort(Heig_vals)
    B_axis.append(B)
    for j in range(no_eigvals):
        E_vallist[j].append(Heig_vals_sorted[j])


# for i in range(no_eigvals):
#     pylab.plot(B_axis, E_vallist[i])
#
# pylab.xlabel('B-Field')
# pylab.ylabel('Energy')
#
# pylab.show()


kb = 1.38e-23
T_max = 3.0
# T_max = 25.0


Zpf_T = []
T_axis = []
for i in range(T_interval):
    Zpf_T.append([])

for w in range(T_interval):
    T_temp = (T_max * (w)) / T_interval
    if T_temp == 0:
        T_temp = 0.0001
    for i in range(B_interval): #this gets all the energies for a given B and T
        expoE_T = 0
        for j in range(no_eigvals):
            expoE_T += numpy.exp(-(E_vallist[j][i])/ (kb * T_temp))
        Zpf_T[w].append(expoE_T)
    T_axis.append(T_temp)


free_energyT = []
free_e = 0
for i in range(T_interval):
    free_energyT.append([])

for w in range(T_interval):
    T_temp = (T_max * w) / T_interval
    for i in range(B_interval):
        if w == 0:
            T_temp = 0.0001
        free_e = - kb * T_temp * numpy.log(Zpf_T[w][i])
        free_energyT[w].append(free_e)

m_momentT = []

for w in range(T_interval):
    m_moment = -numpy.gradient(free_energyT[w], B_field / T_interval)
    m_momentT.append(m_moment)

mu_0 = 1.256e-6

chi_T = []
chi = []
for i in range(T_interval):
    chi_T.append([])

for w in range(T_interval):
    for i in range(B_interval):
            chi_store = (m_momentT[w][i] * mu_0) / B_axis[i]
            chi_T[w].append(chi_store)

chi_constB = []
for i in range(B_interval):
    chi_constB.append([])

#this makes chi a list with constant B and varying temp by basically transposing the array
for w in range(B_interval):
    for j in range(T_interval):
        chi_constB[w].append(chi_T[j][w])

chi_g2mub2n = []
kbT_J_axis = []
for i in range(B_interval):
    chi_g2mub2n.append([])

for i in range(T_interval):
    kbT_J_axis.append([])

for w in range(T_interval):
    kbT_J_axis[w] = (kb * T_axis[w])/abs(J)
    for j in range(B_interval):
        chi_g2mub2n[j].append(chi_constB[j][w]/(g*g*mu_b*mu_b*atoms))


print "My program took", time.time() - start_time, "to run"

# pylab.plot(T_axis, chi_constB[1])
pylab.plot(kbT_J_axis, chi_g2mub2n[1])

pylab.xlabel('temp')
pylab.ylabel('chi')
pylab.show()

# T_file = open("more_T_axis.txt","w") #opens file with name of "kbT_J_axis_data.txt"
# for i in range(T_interval):
#     T_file.write("%f\n" % T_axis[i])
# T_file.close()
#
if J < 0:
    sign = "-"
else:
    sign = "+"
if J_chains < 0:
    csign = "-"
else:
    csign = "+"
m = J/J_chains
#
# chi_file = open("more_Jchi_%s%sJ_%sJ_%satom_%schains_%satomsperchains.txt" % (sign, m, csign, atoms, chains, atoms_p_chain) , 'w')
# for i in range(T_interval):
#     chi_file.write("%f\n" % (chi_constB[1][i]*1e29))
# chi_file.close()





# r_T_file = open("kbT_J_axis.txt","w") #opens file with name of "kbT_J_axis_data.txt"
# for i in range(T_interval):
#     r_T_file.write("%f\n" % kbT_J_axis[i])
# r_T_file.close()
#
# chi_file = open("reduced_Jchi_%s%sJ_%sJ_%satom_%schains_%satomsperchains.txt" % (sign, m, csign, atoms, chains, atoms_p_chain) , 'w')
# for i in range(T_interval):
#     chi_file.write("%f\n" % (chi_g2mub2n[1][i]))
# chi_file.close()




























# E_Bvallist = []
# for i in range(no_eigvals):
#         E_Bvallist.append(E_vallist[i][1])
#
# C = []
# U = []
# for w in range(T_interval):
#     T_temp = (T_max * (w)) / T_interval
#     if T_temp == 0:
#         T_temp = 0.0001
#     internal_e, heatcap = heatcapacity(E_Bvallist, no_eigvals, kb, T_temp)
#     U.append(internal_e)
#     C.append(heatcap)

# pylab.plot(T_axis, C)
# pylab.plot(T_axis, U)
# pylab.show()