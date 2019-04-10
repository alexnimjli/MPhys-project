import pylab
import numpy
from numpy import matrix
from numpy import linalg
import time

start_time = time.time()
I22 = numpy.eye(2)

#.kron is the kronecker product, which is basically the outer
#product operation for matrices instead of vectors

sigmaz = numpy.array([[1,0], [0,-1]])
sigmaz_1 = numpy.kron(sigmaz,I22)
sigmaz_2 = numpy.kron(I22,sigmaz)
print "sigmaz_1:\n", sigmaz_1
print "sigmaz_2:\n", sigmaz_2

sigmax = numpy.array([[0,1], [1,0]])
sigmax_1 = numpy.kron(sigmax,I22)
sigmax_2 = numpy.kron(I22,sigmax)
print sigmax_1
print sigmax_2

sigmay = numpy.array([[0,complex(0,-1)], [complex(0,1),0]])
sigmay_1 = numpy.kron(sigmay,I22)
sigmay_2 = numpy.kron(I22,sigmay)
print sigmay_1
print sigmay_2

zcomp = numpy.dot(sigmaz_1,sigmaz_2)
xcomp = numpy.dot(sigmax_1,sigmax_2)
ycomp = numpy.dot(sigmay_1,sigmay_2)

print "zcomp\n", zcomp
print "ycomp\n", ycomp
print "xcomp\n", xcomp

S = zcomp + ycomp + xcomp

print S
 
eig_vals, eig_vecs = numpy.linalg.eig(S)

print "the eigenvalues are:\n", eig_vals
print "the eigenvectors are:\n", eig_vecs

for i in range(len(eig_vals)):
	print "eigenvalue no.", i ," is ", eig_vals[i]

#here I have B field change from 0-5T

g = 2
mu_b = 9.275e-24
J = 1e-24

e0 = []
e1 = []
e2 = []
e3 = []
Xaxis = []



B_field = 0.05

for i in range(500):
	B = (B_field*i)/100
	H_matrix = mu_b*g*B*0.5*(sigmaz_1+sigmaz_2)+J*S
	Heig_vals, Heig_vecs = numpy.linalg.eig(H_matrix)
	e0.append(Heig_vals[0])
	e1.append(Heig_vals[1])
	e2.append(Heig_vals[2])
	e3.append(Heig_vals[3])
	Xaxis.append(B)
		
pylab.plot(Xaxis,e0)
pylab.plot(Xaxis,e1)
pylab.plot(Xaxis,e2)
pylab.plot(Xaxis,e3)

pylab.xlabel('B-Field')
pylab.ylabel('Energy')

print "My program took", time.time() - start_time, "to run"

pylab.show()
