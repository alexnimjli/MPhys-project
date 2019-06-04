import pylab
import scipy
import numpy

from scipy import linalg
from numpy import matrix
from numpy import linalg

I22 = numpy.eye(2)

atoms = 5
gamma = 1

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

######## Now lets find the exchange hamiltonian with all the components we've found
no_eigvals = 2 ** atoms
SdotSzcomp = 0
SdotSycomp = 0
SdotSxcomp = 0
Szcomptot = 0

# here i make the components each pauli spin matrix dotted together
for i in range(atoms):
    if i == atoms - 1:
        SdotSzcomp = SdotSzcomp + numpy.dot(sigmazlist[i], sigmazlist[0])
        SdotSycomp = SdotSycomp + numpy.dot(sigmaylist[i], sigmaylist[0])
        SdotSxcomp = SdotSxcomp + numpy.dot(sigmaxlist[i], sigmaxlist[0])
    else:
        SdotSzcomp = SdotSzcomp + numpy.dot(sigmazlist[i], sigmazlist[i + 1])
        SdotSycomp = SdotSycomp + numpy.dot(sigmaylist[i], sigmaylist[i + 1])
        SdotSxcomp = SdotSxcomp + numpy.dot(sigmaxlist[i], sigmaxlist[i + 1])
    # ztot is the zeeman effect component, so addition of all the z pauli matrices
    Szcomptot = Szcomptot + sigmazlist[i]

# S is the exchange matrix
SdotS = SdotSzcomp + gamma * SdotSycomp + gamma * SdotSxcomp

Seig_vals, Seig_vecs = numpy.linalg.eig(SdotS)

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
    H_matrix = - g * mu_b * B * Szcomptot + J * SdotS
    Heig_vals, Heig_vecs = numpy.linalg.eig(H_matrix)
    Heig_vals_sorted = numpy.sort(Heig_vals)
    B_axis.append(B)
    # this here will put all the energies in the right lists
    for j in range(no_eigvals):
        E_vallist[j].append(Heig_vals_sorted[j])


for i in range(no_eigvals):
#     pylab.scatter(Xaxis, E_vallist[i], s=0.001)
    pylab.plot(B_axis, E_vallist[i])

pylab.xlabel('$B(T)$', fontsize = "20")
pylab.ylabel('$E$ $(J)$', fontsize = "20")
#
pylab.show()
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

free_energyT = []
for i in range(interval):
    free_energyT.append([])
for w in range(interval):
    T_temp = (T_max * w) / interval
    for i in range(interval):
        if w == 0:
            T_temp = 0.0001
        free_e = - kb * T_temp * numpy.log(Zpf_T[w][i])
        free_energyT[w].append(free_e)

m_momentT = []
for w in range(interval):
    m_moment = -numpy.gradient(free_energyT[w], B_field / interval)
    m_momentT.append(m_moment)

mu_0 = 1.256e-6

chi_T = []
chi = []
for i in range(interval):
    chi_T.append([])
for w in range(interval):
    for i in range(interval):
        chi_store = (m_momentT[w][i] * mu_0) / B_axis[i]
        chi_T[w].append(chi_store)

chi_constB = []
for i in range(interval):
    chi_constB.append([])
#this makes chi a list with constant B and varying temp by basically transposing the array
for w in range(interval):
    for j in range(interval):
        chi_constB[w].append(chi_T[j][w])

#lets change the units of the temp axis and chi axis

chi_g2mub2n = []
kbT_J_axis = []
for i in range(interval):
    chi_g2mub2n.append([])
    kbT_J_axis.append([])

for w in range(interval):
    kbT_J_axis[w] = (kb * T_axis[w])/abs(J)
    for j in range(interval):
        chi_g2mub2n[w].append(chi_constB[w][j]/(g*g*mu_b*mu_b*atoms))

# pylab.plot(T_axis, chi_constB[1])
# pylab.plot(kbT_J_axis, chi_g2mub2n[1])

# pylab.plot(B_axis, m_momentT[1])

# pylab.xlabel('temp')
# pylab.ylabel('chi')
# pylab.show()





# T_file = open("kbT_J.txt","w") #opens file with name of "kbT_J_axis_data.txt"
# for i in range(interval):
#     T_file.write("%f\n" % kbT_J_axis[i])
# T_file.close()
#
if J < 0:
    sign = "-"
else:
    sign = "+"
# chi_file = open("chi_%sJ_%satom.txt" % (sign, atoms) , 'w')
# for i in range(interval):
#     chi_file.write("%f\n" % chi_g2mub2n[1][i])
# chi_file.close()

# m_m_file = open("moment_%sJ_%satom.txt" % (sign, atoms) , 'w')
# for i in range(interval):
#     m_m_file.write("%f\n" % m_momentT[1][i])
# m_m_file.close()


