import pylab
import scipy
import numpy
import time

from scipy import linalg
from numpy import matrix
from numpy import linalg
start_time = time.time()


# here is the lanzcos function
def lanzcos(H_matrix, atoms, steps):

    dim = 2**atoms
    T_matrix = numpy.zeros(shape=(steps,steps))
    # psi = numpy.random.rand(dim, 1)*2 -1
    psi = numpy.random.rand(dim, 1)
    v_j = 0
    v_j_1 = 0
    for j in range(steps):
        if j == 0:
            beta_j = numpy.linalg.norm(psi)
            v_j = psi / beta_j
            r = v_j
            T_matrix[j,j] = numpy.dot( numpy.transpose(v_j), numpy.dot(H_matrix, v_j))[0][0]
            psi = numpy.dot( H_matrix, v_j) - T_matrix[j,j] * v_j
            v_j_1 = v_j
        else:
            beta_j = numpy.linalg.norm(psi)
            v_j = psi / beta_j
            T_matrix[j, j-1] = beta_j
            T_matrix[j-1, j] = T_matrix[j, j-1]
            T_matrix[j,j] = numpy.dot( numpy.transpose(v_j), numpy.dot(H_matrix, v_j))[0][0]
            psi = numpy.dot( H_matrix, v_j) - T_matrix[j,j] * v_j - beta_j * v_j_1
            v_j_1 = v_j
    return (T_matrix, r)

atoms = 8
B_field = 10.0
interval = 100

# this is the number of lanzcos steps
l_steps = 25

# these lists will store the different sigmas for each atom
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

################################


sigmatime = time.time()
print "Time to construct all sigmas: ", sigmatime - start_time

# Now lets find the coupling term of the hamiltonian
no_eigvals = 2 ** atoms
Szcomp = 0
Sycomp = 0
Sxcomp = 0
zcomptot = 0

# here i make the components each pauli spin matrix dotted together
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
SdotS = Szcomp + Sycomp + Sxcomp

g = 2.0
mu_b = 9.275e-24
J = 1.38e-23 * 5

# now i will make a lists in lists so we can have each of the energy values and how it
# varies with the magnetic field stored
B_axis = []
random_veclist = []
E_vallist = []
E_veclist = []
E_vallist_sorted = []
for i in range(l_steps):
    E_vallist.append([])
    E_veclist.append([])


# this will solve for the hamiltonians at different B-fields
for i in range(interval):
    B = (B_field * i) / interval
    H_matrix = - g * mu_b * B * zcomptot + J * SdotS

    # this calls the lanczos function
    T_matrix, r = lanzcos(H_matrix, atoms, l_steps)
    Teig_vals, Teig_vecs = numpy.linalg.eig(T_matrix)
    Teig_vals_sorted = numpy.sort(Teig_vals)
    B_axis.append(B)
    random_veclist.append(r)
    for j in range(l_steps):
        E_vallist[j].append(Teig_vals[j])
        E_veclist[j].append(Teig_vecs[:,j])


energy_time = time.time()
print "Time to get all energies: ", energy_time - start_time


#  ok, now we find the partition function, free energy and m

kb = 1.38e-23
T_max = 25.0

# a series of (100...) (010...) (0010...) vectors
listr = []
for i in range(l_steps):
    r = numpy.zeros(shape=(l_steps,1))
    r[i,0] = 1.0
    listr.append(r)

Zpf_T = []
T_axis = []
for i in range(interval):
    Zpf_T.append([])

for w in range(interval):
    T_temp = (T_max * (w)) / interval
    if T_temp == 0:
        T_temp = 0.0001
    for i in range(interval):
        expoE_T = 0
        for j in range(l_steps):
            for k in range(l_steps):
                expoE_T += (numpy.absolute((numpy.dot(numpy.transpose(listr[k]), E_veclist[j][i])[0])))**2 * numpy.exp(-(E_vallist[j][i])/ (kb * T_temp))
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

for w in range(interval):
    for j in range(interval):
        chi_constB[w].append(chi_T[j][w])


# changing the units of chi and T into Bonner + Fisher units
chi_g2mub2n = []
kbT_J_axis = []
for i in range(interval):
    chi_g2mub2n.append([])
    kbT_J_axis.append([])
for w in range(interval):
    kbT_J_axis[w] = (kb * T_axis[w])/abs(J)
    for j in range(interval):
        chi_g2mub2n[w].append(chi_constB[w][j]/(g*g*mu_b*mu_b*atoms))

pylab.plot(kbT_J_axis, chi_g2mub2n[1])

print "My program took", time.time() - start_time, "to run"

# pylab.xlabel('temp')
# pylab.ylabel('chi')
pylab.show()






