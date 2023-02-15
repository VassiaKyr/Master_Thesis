import pandas as pd
from Kernels import Kernels
from Kernels_Tr import Kernels_Tr
import numpy as np
import pickle
from params import *
from operator import add

###------ check if the particles are unwrapped or wrapped
def check_unwrapped():
    t=False
    if (max(all_coords[:,0])>all_boxes[0,0]) or (min(all_coords[:,0])<0):
        print(max(all_coords[:,0]),min(all_coords[:,0]),7)
        t=True
    if (max(all_coords[:,1])>all_boxes[0,1]) or (min(all_coords[:,1])<0) :
        print(max(all_coords[:,1]),min(all_coords[:,1]),-1)
        t=True
    if (max(all_coords[:,2])>all_boxes[0,2]) or (min(all_coords[:,2])<0):
        print(max(all_coords[:,2]),min(all_coords[:,2]),0)
        t=True
    if t:
        print("Unwrapped particles")
    else:
        print("Wrapped particles")
    
    
#if __name__ == '__main__':
def prepare(max_frames,number_of_basis,delta,theta):   

    all_coords = pd.read_csv(cg_coords_fname, delimiter='\s+', header=None).values
    nconfs = all_coords.shape[0] // nats
    print("Read", nconfs, "configurations")
    all_forces = pd.read_csv(cg_forces_fname, delimiter='\s+', header=None).values
    nconfsf = all_forces.shape[0] // nats
    if nconfsf!= nconfs:
        raise Exception('bad number of boxes')

    if max_frames < nconfs:
        all_coords = all_coords[0:max_frames * nats]
        nconfs = all_coords.shape[0] // nats
        print("Truncated to {} frames".format(nconfs))
    
    #check_unwrapped()
        
    #---------- create the basis points
    if  method=='splines' :
        basis_points = np.array(np.linspace(3.25,rcut,number_of_basis))
        print(number_of_basis," basis_points: ",basis_points )
    elif method=='kernels': 
        if  not add_trimer:
            basis_points = np.array(np.linspace(3.25,rcut,number_of_basis))
            print(number_of_basis," basis_points: ",basis_points )
        else:
            e = 0.25
            b1 = np.array(np.linspace(2*min_distanc-e, 2*rcut, number_of_basis))
            b2 = np.array(np.linspace(0.0 ,(rcut-min_distanc)**2, number_of_basis))
            b3 = np.array(np.linspace(min_distanc-e, rcut, number_of_basis))
            basis_points = np.stack((b1,b2,b3))
    else:
        basis_points = np.zeros(10)
    
    ##---------prepare the matrix (kernels, splines, LJ)-------------
    f = open(Kernels_fname, 'wb')
    for iconf in range(nconfs):
        
        print("Processing configuration", iconf)
        coords = all_coords[iconf*nats:(iconf+1)*nats,:]
        i = 0
        for rootAtomIndex in range(nats-2):
            
            if add_trimer:
                d2 = Kernels_Tr(rootAtomIndex, coords, number_of_basis, basis_points, delta,theta)
            else:
                d2 = Kernels(rootAtomIndex, coords, number_of_basis, basis_points, delta,theta)
            K = d2.kernels_constr()
            if i == 0:
                Kernel_matrix = K
                
            else:
                Kernel_matrix = np.add(K, Kernel_matrix)
                
            i+=1 
        pickle.dump(Kernel_matrix, f)
    f.close()
            
       

        
        

