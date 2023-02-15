import numpy as np

nats = 512
max_frames_l = [5] #max frames= 7400
rcut = 12.0
L = np.array([33.4033, 33.4227, 31.9355])

### choose for method: kernels, LJ or splines
method = "kernels"



min_distanc = 3.2782924061163327

## choose the parameters 
number_of_basis_l = [15, 20, 48] # for LJ we have 2 number_of_basis for dimer
number_of_basis_l_tr = [ 5 ] # for trimer (kernels only)
delta_l = [1.0]#, 2.0, 6.0] 
theta_l = [2.5, 3.5, 5.0]  #If we have trimer then theta has 3 elements
theta_l_tr = [np.array([2.5, 3.5, 5.0])]
lambdaa_l = [0.0, 0.1, 0.01, 0.001]
train_test_s = 0.2

create_dots = True
create_scatter = True
create_pdf = True
add_trimer = True   # only for kernels
regular_matrix = [True, False]

if method == 'kernels':
    Kernels_fname =  'Kernels.dat'
elif  method == 'splines':
    Kernels_fname = 'Splines.dat'
elif method == 'LJ':
    Kernels_fname = 'LJ.dat'

cg_coords_fname = 'CG_coords.txt' 
cg_forces_fname = 'CG_forces.txt'
cg_boxes_fname = 'box.txt'

train_forces_fname = 'train_forces.txt'
test_forces_fname = 'test_forces.txt'
train_Kernels_fname = 'train_Kernels.dat'
test_Kernels_fname = 'test_Kernels.dat'

###------True if we want to rescale the forces from -1 to 1
rescale = False
rescale_param = 1.0
if rescale:
    forces = pd.read_csv(cg_forces_fname,delimiter='\s+', header=None).values
    rescale_param = np.abs(forces.max())