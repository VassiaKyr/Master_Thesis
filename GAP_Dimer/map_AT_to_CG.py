from params import *
import numpy as np
import pandas as pd
import pickle


coords_AT = 'out_xxp.dat'
forces_AT = 'out_ff_nb.dat'
map_coords = 'CG_coords.txt'
map_forces = 'CG_forces.txt'

####-------------params------------------
m_C = 12
m_H = 1
m_tot = m_C + 4*m_H
N_at = 2560


all_coords = pd.read_csv(coords_AT, delimiter='\s+', header=None, skip_blank_lines=True).values
nconfs = all_coords.shape[0] // N_at
all_forces = pd.read_csv(forces_AT, delimiter='\s+', header=None, skip_blank_lines=True).values
nconfs1 = all_forces.shape[0] // N_at
if nconfs == nconfs1:
    print("Read", nconfs, "configurations")


####---------map positions----------------
number_of_CG_part = int(nconfs*N_at/5)
coords_CG = np.zeros((number_of_CG_part, 3))
k = 0
for i in range(0,nconfs*N_at,5):
    coords_CG[k,:] = (all_coords[i,:]*m_C + all_coords[i+1,:]*m_H+all_coords[i+2,:]*m_H+all_coords[i+3,:]*m_H+all_coords[i+4,:]*m_H)/m_tot
    k+=1
np.savetxt(map_coords,coords_CG)


####---------map forces----------------
F_CG = np.zeros((number_of_CG_part, 3))
k = 0
for i in range(0,nconfs*N_at,5):
    F_CG[k,:] = all_forces[i,:]+ all_forces[i+1,:]+all_forces[i+2,:]+all_forces[i+3,:]+all_forces[i+4,:]
    k+=1
np.savetxt(map_forces,F_CG)
