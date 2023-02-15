import numpy as np
from auxil import calc_distance_1d,  unitary, grad_Disc_1, grad_Disc_2, grad_Disc_3
from params import *
import math
from operator import add
from numba import jit, njit, int64, float64  
from numba.experimental import jitclass

spec = [
    ('all_atom_coords',float64[:,:] ),
    ('rootAtomIndex',int64),
    ('number_of_basis', int64),
    ('basis_points',float64[:,:]),
    ('K',float64[:,:]),
    ('delta',float64),
    ('theta',float64[:])
]

@jitclass(spec) 
class Kernels_Tr:
    def __init__(self, rootAtomIndex, all_atom_coords, number_of_basis, basis_points, delta, theta):
        self.all_atom_coords = all_atom_coords
        self.rootAtomIndex = rootAtomIndex
        self.number_of_basis = number_of_basis
        self.basis_points = basis_points
        self.K =  np.zeros((nats*3, len(basis_points[0])*len(basis_points[1])*len(basis_points[2])))
        self.delta = delta
        self.theta = theta
        
        
        
    
    def kernels_constr(self):
        for i in range(self.rootAtomIndex+1,len(self.all_atom_coords[:,0])):
            gradD1 = np.zeros(nats*3)
            gradD2 = np.zeros(nats*3)
            gradD3 = np.zeros(nats*3)
            
            
            X1 = calc_distance_1d(self.rootAtomIndex, i, self.all_atom_coords)
            d1 = X1[0] 
            g1 = np.array([X1[1], X1[2], X1[3]])
            
           
            
            for j in range(i+1, len(self.all_atom_coords[:,0])):
                X2 = calc_distance_1d(self.rootAtomIndex, j, self.all_atom_coords)
                d2 = X2[0] 
                g2 = np.array([X2[1], X2[2], X2[3]])
                
                X3 = calc_distance_1d(j, i, self.all_atom_coords)
                d3 = X3[0] 
                g3 = np.array([X3[1], X3[2], X3[3]])
                
                
                grad11 = grad_Disc_1(d1,d2, g1,g2, 1)
                grad12 = grad_Disc_1(d1,d2, g1,g2, 2)
                grad13 = grad_Disc_1(d1,d2, g1,g2, 3)
                
                grad21 = grad_Disc_2(d2,d3, g2,g3, 1)
                grad22 = grad_Disc_2(d2,d3, g2,g3, 2)
                grad23 = grad_Disc_2(d2,d3, g2,g3, 3)
                
                grad3 = grad_Disc_3(d3, g3, 1)
                
                gradD1[self.rootAtomIndex*3:self.rootAtomIndex*3+3] = grad11[:]
                gradD1[i*3:i*3+3] = grad12[:]
                gradD1[j*3:j*3+3] = grad13[:]
                
                gradD2[self.rootAtomIndex*3:self.rootAtomIndex*3+3] = grad21[:]
                gradD2[i*3:i*3+3] = grad22[:]
                gradD2[j*3:j*3+3] = grad23[:]
                
                
                gradD3[i*3:i*3+3] = grad3[:]
                gradD3[j*3:j*3+3] = -grad3[:] 
                
                
                
                l = 0
                for l1 in range(len(self.basis_points[0])):
                    dist1 = (d1 - self.basis_points[0][l1])
                    
                    for l2 in range(len(self.basis_points[1])):
                        dist2 = (d2 - self.basis_points[1][l2])
                            
                        for l3 in range(len(self.basis_points[2])):
                            dist3 = (d3 - self.basis_points[2][l3])
                            
                            k = (self.delta**2) * np.exp(-(dist1**2) / (2*self.theta[0]**2)- (dist2**2) / (2*self.theta[1]**2)- (dist3**2) / (2*self.theta[2]**2))
                            
                            KK1 = (dist1 * gradD1 * k)/(self.theta[0]**2)
                            KK2 = (dist2 * gradD2 * k)/(self.theta[1]**2)
                            KK3 = (dist3 * gradD3 * k)/(self.theta[2]**2)
                            KK = - KK1 - KK2 - KK3
                            self.K[:,l] = np.add(KK,self.K[:,l])
                        
                            l+=1
            
                    
        return (self.K)
      
     
            
    
    
    
        
        
        
    
        
    