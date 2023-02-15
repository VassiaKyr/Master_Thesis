from __future__ import print_function
from params import *
import pandas as pd
import  prepare 
from auxil import *
import numpy as np
import math
import random

def train_test_split(number_of_basis,train_percent):
    if add_trimer:
        number_of_basis =  number_of_basis**3
    
    random.seed(10) ##choose  "specific random" variables
    kernel_matrix = read_kernels(Kernels_fname)
    forces = pd.read_csv(cg_forces_fname,delimiter='\s+', header=None).values
    
    
    if rescale:
        forces = forces/rescale_param
        
        
    kernel_matrix = kernel_matrix.reshape(-1, number_of_basis)   
    nconfsKer = kernel_matrix.shape[0]//(nats*3)
    nconfsf = forces.shape[0] // nats
    nconfs = min(nconfsKer, nconfsf)
    print("Loaded", nconfs, "configurations")
   
    
    kernel_matrix = kernel_matrix[:nats*nconfs*3,:] 
    forces = forces[:nats*nconfs,:]
    
    nconfs_test = int(math.floor(nconfs * train_percent))
    confs = list(range(nconfs))
    random.shuffle(confs)
    test_confs = confs[0:nconfs_test]
    train_confs = confs[nconfs_test:]
    nconfs_test = len(test_confs)
    nconfs_train = len(train_confs)

    print("Number of train confs:", nconfs_train)
    print("Number of test confs:", nconfs_test)
    
    
    train_ker = np.zeros((nconfs_train*nats*3, number_of_basis))
    train_for = np.zeros((nconfs_train*nats, 3))
    test_ker = np.zeros((nconfs_test*nats*3, number_of_basis))
    test_for = np.zeros((nconfs_test*nats, 3))
    
    idest = 0
    for isrc in train_confs:
        train_ker[idest*nats*3:(idest+1)*nats*3, :] = kernel_matrix[isrc*nats*3:(isrc+1)*nats*3, :]
        train_for[idest*nats:(idest+1)*nats,:] = forces[isrc*nats:(isrc+1)*nats,:]
        idest += 1
    
    idest = 0
    for isrc in test_confs:
        test_ker[idest*nats*3:(idest+1)*nats*3,:] = kernel_matrix[isrc*nats*3:(isrc+1)*nats*3,:]
        test_for[idest*nats:(idest+1)*nats,:] = forces[isrc*nats:(isrc+1)*nats,:]
        idest += 1
    return train_ker, train_for, test_ker, test_for
    
    
    
#if __name__ == '__main__':
def sep_train_test(number_of_basis,train_test_s):
    train_ker, train_for, test_ker, test_for = train_test_split(number_of_basis,train_test_s)
    np.savetxt(train_Kernels_fname, train_ker)
    np.savetxt(train_forces_fname, train_for)
    np.savetxt(test_Kernels_fname, test_ker)
    np.savetxt(test_forces_fname, test_for)

