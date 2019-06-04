import pylab
import scipy
import numpy
import time

from scipy import linalg
from numpy import matrix
from numpy import linalg

start_time = time.time()

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

atoms = 8
gamma = 1.0

sigmazlist = []
sigmaxlist = []
sigmaylist = []
I22 = numpy.eye(2)
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

sigma_time = time.time()
print "Time to get all sigmas: ", sigma_time - start_time
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
T_interval = 200
B_interval = 3

for i in range(B_interval):
    B = (B_field * i) / T_interval
    H_matrix = - g * mu_b * B * Szcomptot + J * SdotS
    Heig_vals, Heig_vecs = numpy.linalg.eig(H_matrix)
    Heig_vals_sorted = numpy.sort(Heig_vals)
    B_axis.append(B)
    for j in range(no_eigvals):
        E_vallist[j].append(Heig_vals_sorted[j])

kb = 1.38e-23
T_max = 6.25

energy_time = time.time()
print "Time to get all energies: ", energy_time - start_time

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

z_time = time.time()
print "Time to get calc Z's: ", z_time - start_time

free_energyT = []
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

#lets change the units of the temp axis and chi axis

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

pylab.plot(kbT_J_axis, chi_g2mub2n[1])

T_file = open("kbT_J_8chi.txt","w") #opens file with name of "kbT_J_axis_data.txt"
for i in range(T_interval):
    T_file.write("%f\n" % kbT_J_axis[i])
T_file.close()

if J < 0:
    sign = "-"
else:
    sign = "+"



E_Bvallist = []
for i in range(no_eigvals):
        E_Bvallist.append(E_vallist[i][1])

C = []
C_Nk = []
for w in range(T_interval):
    T_temp = (T_max * (w)) / T_interval
    if T_temp == 0:
        T_temp = 0.0001
    internal_e, heatcap = heatcapacity(E_Bvallist, no_eigvals, kb, T_temp)
    C.append(heatcap)
    C_Nk.append(heatcap/(atoms*kb))


# pylab.plot(kbT_J_axis, C_Nk)
#
# C_file = open("C_Nk_%sJ_%satom_%sgamma.txt" % (sign, atoms, gamma) , 'w')
# for i in range(T_interval):
#     C_file.write("%f\n" % C_Nk[i])
# C_file.close()

print "My program took", time.time() - start_time, "to run"

pylab.show()
