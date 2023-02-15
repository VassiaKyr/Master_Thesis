from __future__ import print_function
import math
import numpy as np
import pickle
import sys
from params import *
from numba import njit

#f.write('\n')
def read_kernels(filename):
    read_kernels = []
    f = open(filename,'rb')
    i = 0
    while True:
        try:
            ds = pickle.load(f)
            read_kernels.append(ds)
            print('\r',end='')
            print(i,end='')
            sys.stdout.flush()
            i += 1
        except Exception as e:
            print('')
            print(str(e))
            break
    f.close()
    print("")
    kernels = np.array(read_kernels, dtype=np.float32)
    #print("readkernels",len(kernels),len(kernels[0]))
    return kernels



@njit("float64(float64)")
def py2round(x):
    if x >= 0.0:
        return math.floor(x + 0.5)
    else:
        return math.ceil(x - 0.5)


@njit("float64(float64,float64,float64)")
def periodicDistance(x1, x2, length):
    delta = x1-x2   
    return delta - py2round(delta/length)*length

    
def unitary(vector):
    return vector / np.linalg.norm(vector)
    
@njit("float64[:](float64[:],float64[:])")
def periodicDistance3(v1, v2):
    dx = periodicDistance(v1[0], v2[0], L[0])
    dy = periodicDistance(v1[1], v2[1], L[1])
    dz = periodicDistance(v1[2], v2[2], L[2])
    x = np.array([math.sqrt(dx*dx + dy*dy + dz*dz), dx, dy, dz])
    return x
    
# v1 - v2
def periodicDiff2(v1,v2,box_size):
    dx = periodicDistance(v1[0], v2[0], box_size[0])
    dy = periodicDistance(v1[1], v2[1], box_size[1])
    dz = periodicDistance(v1[2], v2[2], box_size[2])
    return np.array([dx,dy,dz])


@njit("float64[:](int64,int64,float64[:,:])")
def calc_distance_1d(rootAtomIndex, neiAtomIndex, coords):
    rootPos = coords[rootAtomIndex,:]
    neiPos = coords[neiAtomIndex,:]
    return periodicDistance3(rootPos, neiPos)

      
        
@njit()      
def grad_D( Dist, g):
    g = g * (1.0/Dist)
    return -g

@njit()
def forces_pdf(force1, force2, X):
    D = X[0]
    sub_coords = np.array([X[1], X[2], X[3]]) 
    #forces = np.add(force1,force2)
    forces = np.subtract(force1,force2)
    e_ij = np.multiply(np.absolute(sub_coords), 1/D)
    F = 0
    for i in range(len(e_ij)):
        F += e_ij[i]*forces[i]
    return np.multiply(F,e_ij)
    
    
    
    
    
