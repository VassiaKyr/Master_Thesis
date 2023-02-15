import numpy as np
from auxil import calc_distance_1d,  unitary, grad_D
from params import *
import math
from operator import add
from numba import jit, njit, int64, float64  
from numba.experimental import jitclass

spec = [
    ('all_atom_coords',float64[:,:] ),
    ('rootAtomIndex',int64),
    ('number_of_basis', int64),
    ('basis_points',float64[:]),
    ('K',float64[:,:]),
    ('delta',float64),
    ('theta',float64)
]

@jitclass(spec) 
class Kernels:
    def __init__(self, rootAtomIndex, all_atom_coords, number_of_basis, basis_points, delta, theta):
        self.all_atom_coords = all_atom_coords
        self.rootAtomIndex = rootAtomIndex
        self.number_of_basis = number_of_basis
        self.basis_points = basis_points
        self.K =  np.zeros((nats*3, number_of_basis))
        self.delta = delta
        self.theta = theta
        
        
        
    
    def kernels_constr(self):
        for i in range(self.rootAtomIndex+1,len(self.all_atom_coords[:,0])):
            gradD = np.zeros(nats*3)
            X = calc_distance_1d(self.rootAtomIndex, i, self.all_atom_coords)
            d = X[0] 
            g = np.array([X[1], X[2], X[3]])
            grad = grad_D(d, g)
            gradD[self.rootAtomIndex*3:self.rootAtomIndex*3+3] = -grad[:]
            gradD[i*3:i*3+3] = grad[:]
            
            
            if method=='kernels':
                
                for j in range(self.number_of_basis):
                    dist = (d - self.basis_points[j])**2
                    k = (self.delta**2)*np.exp(-dist/(2*self.theta**2))
                    KK = gradD * k
                    self.K[:,j] = np.add(KK,self.K[:,j])
            
            
            elif method=='splines':
                dx = self.basis_points[1]-self.basis_points[0]
                for j in range(0,self.number_of_basis):
                    k = 0.0
                    
                    if d<= self.basis_points[j] and d>self.basis_points[j-1] and j>=1:
                        k = (d-self.basis_points[j-1])*(1./dx)
                        
                    elif d> self.basis_points[j] and d<=self.basis_points[j+1] and j<self.number_of_basis:
                        k = (self.basis_points[j+1]-d)*(1./dx)
                        
                    KK = gradD * k
                    self.K[:,j] = np.add(KK,self.K[:,j])
            
            
            elif method == 'LJ':
                
                if d<=rcut:
                    dvx = 6.0*d**(-7)
                    dvy = -12.0*d**(-13)
                    KK1 = dvx*gradD
                    KK2 = dvy*gradD
                    self.K[:,0] = np.add(KK1, self.K[:,0])
                    self.K[:,1] = np.add(KK2, self.K[:,1])
                    
                
        
        return (self.K)
      
     
            
    
    
    
        
        
        
    
        
    