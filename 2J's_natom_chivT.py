import pylab
import scipy
import numpy

from scipy import linalg
from numpy import matrix
from numpy import linalg

I22 = numpy.eye(2)

# atoms = int(raw_input("how many atoms are in your system? "))
atoms = 3

sigmazlist = []
sigmaxlist = []
sigmaylist = []
sigmaz = numpy.array([[0.5, 0], [0, -0.5]])
sigmax = numpy.array([[0, 0.5], [0.5, 0]])
sigmay = numpy.array([[0, complex(0, -0.5)], [complex(0, 0.5), 0]])

######################
# Here we construct the Pauli spin matrices for n-atom system

for i in range(atoms):
    if i == 0:
        sigma_nz = sigmaz
        sigma_nx = sigmax
        sigma_ny = sigmay
        for j in range(atoms - 1):
            sigma_nz = numpy.kron(sigma_nz, I22)
            sigma_nx = numpy.kron(sigma_nx, I22)
            sigma_ny = numpy.kron(sigma_ny, I22)

        sigmazlist.append(sigma_nz)
        sigmaylist.append(sigma_ny)
        sigmaxlist.append(sigma_nx)

    elif i == atoms - 1:
        sigma_nz = I22
        sigma_nx = I22
        sigma_ny = I22
        # -2 here because we are looking at the number of I .krons to take before finally sigma
        for j in range(atoms - 2):
            sigma_nz = numpy.kron(sigma_nz, I22)
            sigma_nx = numpy.kron(sigma_nx, I22)
            sigma_ny = numpy.kron(sigma_ny, I22)
        sigma_nz = numpy.kron(sigma_nz, sigmaz)
        sigma_nx = numpy.kron(sigma_nx, sigmax)
        sigma_ny = numpy.kron(sigma_ny, sigmay)

        sigmazlist.append(sigma_nz)
        sigmaylist.append(sigma_ny)
        sigmaxlist.append(sigma_nx)

    else:
        sigma_nz = I22
        sigma_nx = I22
        sigma_ny = I22
        I_before = i - 1
        I_after = atoms - i - 1
        for j in range(I_before):
            sigma_nz = numpy.kron(sigma_nz, I22)
            sigma_nx = numpy.kron(sigma_nx, I22)
            sigma_ny = numpy.kron(sigma_ny, I22)
        sigma_nz = numpy.kron(sigma_nz, sigmaz)
        sigma_nx = numpy.kron(sigma_nx, sigmax)
        sigma_ny = numpy.kron(sigma_ny, sigmay)
        for k in range(I_after):
            sigma_nz = numpy.kron(sigma_nz, I22)
            sigma_nx = numpy.kron(sigma_nx, I22)
            sigma_ny = numpy.kron(sigma_ny, I22)

        sigmazlist.append(sigma_nz)
        sigmaylist.append(sigma_ny)
        sigmaxlist.append(sigma_nx)

################################

######## Now lets find the exchange hamiltonian with all the components we've found
no_eigvals = 2 ** atoms
Szcomp = 0
Sycomp = 0
Sxcomp = 0
zcomptot = 0

# here i make the components each pauli spin matrix dotted together
# for this I am dealing with a circular configuration
for i in range(atoms):
    if i == atoms - 1:
        Szcomp = Szcomp + numpy.dot(sigmazlist[i], sigmazlist[0])
        Sycomp = Sycomp + numpy.dot(sigmaylist[i], sigmaylist[0])
        Sxcomp = Sxcomp + numpy.dot(sigmaxlist[i], sigmaxlist[0])
    else:
        Szcomp = Szcomp + numpy.dot(sigmazlist[i], sigmazlist[i + 1])
        Sycomp = Sycomp + numpy.dot(sigmaylist[i], sigmaylist[i + 1])
        Sxcomp = Sxcomp + numpy.dot(sigmaxlist[i], sigmaxlist[i + 1])
    # ztot is the zeeman effect component, so addition of all the z pauli matrices
    zcomptot = zcomptot + sigmazlist[i]

# S is the exchange matrix
S = Szcomp + Sycomp + Sxcomp

Seig_vals, Seig_vecs = numpy.linalg.eig(S)

# print S
# print Seig_vals
# print zcomptot

g = 2.0
mu_b = 9.275e-24
J = 1.38e-23 * 5  # very small coupling energy?

# now i will make a lists in lists so we can have each of the energy values and how it
# varies with the magnetic field stored
B_axis = []
E_vallist = []
for i in range(no_eigvals):
    E_vallist.append([])

B_field = 10.0 #make sure you use 5.0 not 5 so that this is a double rather than an integer so the loops will give B into decimals rather than integers
interval = 100

for i in range(interval):
    B = (B_field * i) / interval
    H_matrix = - g * mu_b * B * zcomptot + J * S
    Heig_vals, Heig_vecs = numpy.linalg.eig(H_matrix)
    Heig_vals_sorted = numpy.sort(Heig_vals)
    B_axis.append(B)
    # this here will put all the energies in the right lists
    for j in range(no_eigvals):
        E_vallist[j].append(Heig_vals_sorted[j])


# print E_vallist

# for i in range(no_eigvals):
#     pylab.scatter(Xaxis, E_vallist[i], s=0.001)
#     pylab.plot(Xaxis, E_vallist[i])
# print "There should be ", no_eigvals, "energy lines"
# print Heig_vals
#
# pylab.xlabel('B-Field')
# pylab.ylabel('Energy')
#
# pylab.show()
########################
# ok, now we find the partition function, free energy and m

kb = 1.38e-23
T_max = 25.0
# we want to find the partition function at each given B
# so we must add all the exp(E) at each given B value
# we have made 100 B points to evaluate. so there must be 100

Zpf_T = []
T_axis = []

for i in range(interval):
    Zpf_T.append([])

for w in range(interval):
    T_temp = (T_max * (w)) / interval
    if T_temp == 0:
        T_temp = 0.0001
    for i in range(interval): #this gets all the energies for a given B and T
        expoE_T = 0
        for j in range(no_eigvals):
            expoE_T += numpy.exp(-(E_vallist[j][i])/ (kb * T_temp))
        Zpf_T[w].append(expoE_T)
    T_axis.append(T_temp)

# print Zpf_T

free_energyT = []
free_e = 0

for i in range(interval):
    free_energyT.append([])

for w in range(interval):
    T_temp = (T_max * w) / interval
    for i in range(interval):
        if w == 0:
            T_temp = 0.0001
        free_e = - kb * T_temp * numpy.log(Zpf_T[w][i])
        free_energyT[w].append(free_e)

# print "freeenergyT: ", free_energyT

m_momentT = []

for w in range(interval):
    m_moment = -numpy.gradient(free_energyT[w], B_field / interval)
    m_momentT.append(m_moment)


# print "m_momentT: ", m_momentT
# print m_momentT[1][1]

mu_0 = 1.256e-6

chi_T = []

for i in range(interval):
    chi_T.append([])

for w in range(interval):
    for i in range(interval):
            chi_store = (m_momentT[w][i] * mu_0) / B_axis[i]
            chi_T[w].append(chi_store)
#
# for w in range(interval):
#             chi_store = -numpy.gradient(m_momentT[w], B_field/(interval * mu_0))
#             chi_T.append(chi_store)

# print "chi_T: ", chi_T

chi_constB = []
for i in range(interval):
    chi_constB.append([])

#this makes chi a list with constant B and varying temp by basically transposing the array
for w in range(interval):
    for j in range(interval):
        chi_constB[w].append(chi_T[j][w])

store = 0

# chi_constB_inv = []
# for i in range(interval):
#     chi_constB_inv.append([])
#
# for w in range(interval):
#     for j in range(interval):
#         store = 1.0/chi_constB[j][w]
#         chi_constB_inv.append(store)

pylab.plot(T_axis, chi_constB[1])

pylab.xlabel('temp')
pylab.ylabel('chi')


# -----------------------------------------
# okay here we do the repeat but with a different -ve J value so we can superimpose and see the difference

J2 = -1.38e-23 * 5

B_axis2 = []
E_vallist2 = []
for i in range(no_eigvals):
    E_vallist2.append([])

for i in range(interval):
    B = (B_field * i) / interval
    H_matrix = - g * mu_b * B * zcomptot + J2 * S
    Heig_vals, Heig_vecs = numpy.linalg.eig(H_matrix)
    Heig_vals_sorted = numpy.sort(Heig_vals)
    B_axis2.append(B)
    # this here will put all the energies in the right lists
    for j in range(no_eigvals):
        E_vallist2[j].append(Heig_vals_sorted[j])

Zpf_T2 = []
T_axis2 = []

for i in range(interval):
    Zpf_T2.append([])

for w in range(interval):
    T_temp = (T_max * (w)) / interval
    if T_temp == 0:
        T_temp = 0.0001
        for i in range(interval):
            expoE_T = 0
            for j in range(no_eigvals):
                expoE_T += numpy.exp(-(E_vallist2[j][i])/ (kb * T_temp))
            Zpf_T2[w].append(expoE_T)
        T_axis2.append(T_temp)
    else:
        for i in range(interval): #this gets all the energies for a given B and T
            expoE_T = 0
            for j in range(no_eigvals):
                expoE_T += numpy.exp(-(E_vallist2[j][i])/ (kb * T_temp))
            Zpf_T2[w].append(expoE_T)
        T_axis2.append(T_temp)


free_energyT2 = []
free_e2 = 0

for i in range(interval):
    free_energyT2.append([])

for w in range(interval):
    T_temp = (T_max * w) / interval
    for i in range(interval):
        if w == 0:
            T_temp = 0.0001
            free_e2 = - kb * T_temp * numpy.log(Zpf_T2[w][i])
            free_energyT2[w].append(free_e2)
        else:
            free_e2 = - kb * T_temp * numpy.log(Zpf_T2[w][i])
            free_energyT2[w].append(free_e2)

m_momentT2 = []

for w in range(interval):
    m_moment = -numpy.gradient(free_energyT2[w], B_field / interval)
    m_momentT2.append(m_moment)

mu_0 = 1.256e-6

chi_T2 = []

for i in range(interval):
    chi_T2.append([])

for w in range(interval):
    for i in range(interval):
        if i == 0:
            chi_store = 0
            chi_T2[w].append(chi_store)
        else:
            chi_store = (m_momentT2[w][i] * mu_0) / B_axis[i]
            chi_T2[w].append(chi_store)

chi_constB2 = []
for i in range(interval):
    chi_constB2.append([])

#this makes chi a list with constant B and varying temp by basically transposing the array
for w in range(interval):
    for j in range(interval):
        chi_constB2[w].append(chi_T2[j][w])

# chi_constB2_inv = []
# for i in range(interval):
#     chi_constB2_inv.append([])
#
# for w in range(interval):
#     for j in range(interval):
#         store = 1/chi_constB2[j][w]
#         chi_constB2_inv.append(store)


pylab.plot(T_axis2, chi_constB2[1])

pylab.show()

