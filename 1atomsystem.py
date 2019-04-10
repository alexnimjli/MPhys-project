import pylab
import numpy
from numpy import matrix
from numpy import linalg

sigmaz= numpy.array([[1,0], [0,-1]])
A = numpy.array([[2,3],[3,5]])

I33 = numpy.eye(3)
print "Identity 3x3:\n", I33

eig_vals, eig_vecs = numpy.linalg.eig(sigmaz)

print "the eigenvalues are:\n", eig_vals
print "the eigenvectors are:\n", eig_vecs

print numpy.outer(sigmaz,A)

print "the number of eigen values are: ",  len(eig_vals)

for i in range(len(eig_vals)):
	print "eigenvalue no.", i ," is ", eig_vals[i]

#here I have B field change from 0-5T

g = 2
mu_b = 9.275e-24
			
for i in range(len(eig_vals)):
	B = numpy.linspace(0,5,100)
	H = B*eig_vals[i]*g*mu_b*0.5
	pylab.plot(B,H)
#pylab.plot(x,y) is the order of the variables

#pylab.xlabel("Magnetic Field / B")
#pylab.ylabel("Energy / J")
#pylab.title("Energy of system due to varying B field")
#pylab.show()

print eig_vecs[1]
