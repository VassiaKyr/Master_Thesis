# convert F to U given only the F dataset and the correspoding x dataset

import pandas as pd
import numpy as np
import os.path

def conv_FtoU(all):
    if not os.path.exists('data_Up'):
        os.mkdir('data_Up')   
    if not os.path.exists('data_Up/reg_matrx'):
        os.mkdir('data_Up/reg_matrx')
    if not os.path.exists('data_Up/reg_iden'):
        os.mkdir('data_Up/reg_iden')
    
    if all:#read all F files and convert the forces to Potential
        f_w = pd.read_csv('0Wasserstein_distance_new.txt', delimiter='\s+', header=None).values
            
        list_params = ['method', 'number of samples', 'number of basis points','Regular_param', 'delta', 'theta','lambdaa', 'train','test']
        
        
        for i in range(len(f_w)):
            
            a = (f_w[i][0]).split(',')
            if  a[3] == str(True):            
                file_start = 'data_Fp/reg_matrx/forces'
                file_start_new = 'data_Up/reg_matrx/potential'
                
            elif a[3] == str(False):
                file_start = 'data_Fp/reg_iden/forces'
                file_start_new = 'data_Up/reg_iden/potential'
               
                
            file_name_new = str(file_start_new)+str(a[0])+'_'+str(a[1])+'_'+str(a[2])+'_('+str(a[4])+','+str(a[5])+','+str(a[6])+')resc_False_.txt'
            file_name = str(file_start)+str(a[0])+'_'+str(a[1])+'_'+str(a[2])+'_('+str(a[4])+','+str(a[5])+','+str(a[6])+')resc_False_.txt'
                
            if  os.path.isfile(file_name_new):
                continue
                
            file_f = pd.read_csv(file_name, delimiter='\s+', header=None).values
            c = file_f[:,0]
            f = file_f[:,1]
            points =  len(f)
            
            U = np.zeros(points)
            ##create the main method
            for j in range(points-1,0, -1):
                dr = c[j] - c[j-1]
                cur_int = 0.5 * (f[j] + f[j-1])* dr
                U[j-1] = U[j] +cur_int
            
            #write the U file
            file = open(file_name_new, 'w')
            for j in range(points):
                file.write(''.join('{} {}'.format(c[j],U[j])))
                file.write('\n')
            file.close()
             
    else:
        #give a file
        file_name = 'data_Fp/reg_matrx/forceskernels_4000_48_(6.0,2.5,0.1)resc_False_.txt'
        file_f = pd.read_csv(file_name, delimiter='\s+', header=None).values
        c = file_f[:,0]
        f = file_f[:,1]
        points =  len(f)
        
        
        #give the name of U file
        file_name_new = 'data_Up/reg_matrx/potentialkernels_4000_48_(6.0,2.5,0.1)resc_False_.txt'
        
        U = np.zeros(points)
        ##create the main method
        for j in range(points-1,0, -1):
            dr = c[j] - c[j-1]
            cur_int = 0.5 * (f[j] + f[j-1])* dr
            U[j-1] = U[j] +cur_int
        #write the U file
        file = open(file_name_new, 'w')
        for i in range(points):
            file.write(''.join('{} {}'.format(c[i],U[i])))
            file.write('\n')
        file.close()

all = False #parameter if we want to convert all the files
        
conv_FtoU(all)