import numpy as np
import sys
import os.path
from auxil import read_kernels
import pandas as pd
import math
from matplotlib import pyplot as plt


def Find_best_chi_wass(frames):
    f_w = pd.read_csv('0Wasserstein_distance_new.txt', delimiter='\s+', header=None).values
    f_r = pd.read_csv('0Chi-Square.txt', delimiter='\s+', header=None).values

    train = list_params.index('train')
    test = list_params.index('test')
    print(int(len(f_w)/2.))
    
    Wass_Tr = np.zeros((int(len(f_w)/2.),len(list_params)))
    Wass_Fa = np.zeros((int(len(f_w)/2.),len(list_params)))
    Res_Tr = np.zeros((int(len(f_w)/2.),len(list_params)))
    Res_Fa = np.zeros((int(len(f_w)/2.),len(list_params)))
    
    Best_models = np.ones((2,4,4))
    m,n = 0,0
    for i in range(len(f_w)):
        a = (f_w[i][0]).split(',')
        b = (f_r[i][0]).split(',')
        if a[0]=='kernels':
            a[0]=3
            b[0]=3
        elif a[0]=='splines':
            a[0]=2
            b[0]=2
        elif a[0]=='LJ':
            a[0]=1
            b[0]=1
        if a[3] == 'True':
            a[3] = 1
            b[3] = 1
            Wass_Tr[m][:] = list(map(float,a)) 
            Res_Tr[m][:] = list(map(float,b)) 
            m+=1
        else:
            a[3] = 0
            b[3] = 0
            Wass_Fa[n][:] = list(map(float,a)) 
            Res_Fa[n][:] = list(map(float,b)) 
            n+=1
        
    f = open("plot_compare/best_options_new.txt", "w")    
    W_R = list(np.hstack((Wass_Fa,Res_Fa)))
    W_R.sort(key = lambda x: x[train])
    W_tr = W_R[:][:]
    W_R.sort(key = lambda x: x[test] )
    W_te = W_R[:][:]
    W_R.sort(key = lambda x: (x[train+len(list_params)]) )
    R_tr = W_R[:][:]
    W_R.sort(key = lambda x: (x[test+len(list_params)]) )
    R_te = W_R[:][:]
    
    print('Model with a Regularization parameter')
    print("Best train Wasserstein_distance:",W_tr[0][train],"(number of basis points, delta, theta,lambdaa)=",int(W_tr[0][2]), " {:.2f}".format(W_tr[0][4])," {:.2f}".format(W_tr[0][5]), " {:.2f}".format(W_tr[0][6]) )
    print("Best test Wasserstein_distance:",W_te[0][test],"(number of basis points, delta, theta,lambdaa)=",int(W_te[0][2]), " {:.2f}".format(W_te[0][4])," {:.2f}".format(W_te[0][5]), " {:.2f}".format(W_te[0][6]) )
    print("Best train Chi-square:",R_tr[0][train+len(list_params)],"(number of basis points, delta, theta,lambdaa)=",int(R_tr[0][2]), " {:.2f}".format(R_tr[0][4])," {:.2f}".format(R_tr[0][5]), " {:.2f}".format(R_tr[0][6]) )
    print("Best test Chi-square:",R_te[0][test+len(list_params)],"(number of basis points, delta, theta,lambdaa)=",int(R_te[0][2]), " {:.2f}".format(R_te[0][4])," {:.2f}".format(R_te[0][5]), " {:.2f}".format(R_te[0][6]) )
    print()
    
    f.write('Model with a Regularization parameter')
    f.write('\n')
    f.write("Best train Wasserstein_distance:"+str(W_tr[0][train])+"(number of basis points, delta, theta,lambdaa)= "+str(int(W_tr[0][2]))+ " {:.2f}".format(W_tr[0][4])+" {:.2f}".format(W_tr[0][5])+ " {:.3f}".format(W_tr[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[0][0], W_tr[0][1], W_tr[0][2], W_tr[0][3], W_tr[0][4],W_tr[0][5], W_tr[0][6],W_tr[0][7],W_tr[0][8], W_tr[0][-2], W_tr[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[1][0], W_tr[1][1], W_tr[1][2], W_tr[1][3], W_tr[1][4],W_tr[1][5], W_tr[1][6],W_tr[1][7],W_tr[1][8], W_tr[1][-2], W_tr[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[2][0], W_tr[2][1], W_tr[2][2], W_tr[2][3], W_tr[2][4],W_tr[2][5], W_tr[2][6],W_tr[2][7],W_tr[2][8], W_tr[2][-2], W_tr[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[3][0], W_tr[3][1], W_tr[3][2], W_tr[3][3], W_tr[3][4],W_tr[3][5], W_tr[3][6],W_tr[3][7],W_tr[3][8], W_tr[3][-2], W_tr[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[4][0], W_tr[4][1], W_tr[4][2], W_tr[4][3], W_tr[4][4],W_tr[4][5], W_tr[4][6],W_tr[4][7],W_tr[4][8], W_tr[4][-2], W_tr[4][-1] )))
    f.write('\n')
    Best_models[0,0, :] = np.array([ W_tr[0][2], W_tr[0][4],W_tr[0][5], W_tr[0][6]])
    
    f.write("Best test Wasserstein_distance:"+str(W_te[0][test])+"(number of basis points, delta, theta,lambdaa)= "+str(int(W_te[0][2]))+ " {:.2f}".format(W_te[0][4])+" {:.2f}".format(W_te[0][5])+ " {:.3f}".format(W_te[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[0][0], W_te[0][1], W_te[0][2], W_te[0][3], W_te[0][4],W_te[0][5], W_te[0][6],W_te[0][7], W_te[0][8], W_te[0][-2], W_te[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[1][0], W_te[1][1], W_te[1][2], W_te[1][3], W_te[1][4],W_te[1][5], W_te[1][6],W_te[1][7],W_te[1][8], W_te[1][-2], W_te[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[2][0], W_te[2][1], W_te[2][2], W_te[2][3], W_te[2][4],W_te[2][5], W_te[2][6],W_te[2][7],W_te[2][8], W_te[2][-2], W_te[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[3][0], W_te[3][1], W_te[3][2], W_te[3][3], W_te[3][4],W_te[3][5], W_te[3][6],W_te[3][7],W_te[3][8], W_te[3][-2], W_te[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[4][0], W_te[4][1], W_te[4][2], W_te[4][3], W_te[4][4],W_te[4][5], W_te[4][6],W_te[4][7],W_te[4][8], W_te[4][-2], W_te[4][-1] )))
    f.write('\n') 
    Best_models[0][1][:] = np.array([ W_te[0][2], W_te[0][4],W_te[0][5], W_te[0][6]])
    
    f.write("Best train Chi-square:"+str(R_tr[0][train+len(list_params)])+"(number of basis points, delta, theta,lambdaa)= "+str(int(R_tr[0][2]))+ " {:.2f}".format(R_tr[0][4])+" {:.2f}".format(R_tr[0][5])+ " {:.3f}".format(R_tr[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[0][0], R_tr[0][1], R_tr[0][2], R_tr[0][3], R_tr[0][4],R_tr[0][5], R_tr[0][6],R_tr[0][7], R_tr[0][8], R_tr[0][-2], R_tr[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[1][0], R_tr[1][1], R_tr[1][2], R_tr[1][3], R_tr[1][4],R_tr[1][5], R_tr[1][6],R_tr[1][7], R_tr[1][8], R_tr[1][-2], R_tr[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[2][0], R_tr[2][1], R_tr[2][2], R_tr[2][3], R_tr[2][4],R_tr[2][5], R_tr[2][6],R_tr[2][7], R_tr[2][8],R_tr[2][-2], R_tr[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[3][0], R_tr[3][1], R_tr[3][2], R_tr[3][3], R_tr[3][4],R_tr[3][5], R_tr[3][6],R_tr[3][7], R_tr[3][8], R_tr[3][-2], R_tr[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[4][0], R_tr[4][1], R_tr[4][2], R_tr[4][3], R_tr[4][4],R_tr[4][5], R_tr[4][6],R_tr[4][7], R_tr[4][8],R_tr[4][-2], R_tr[4][-1] )))
    f.write('\n')
    Best_models[0][2][:] = np.array([ R_tr[0][2], R_tr[0][4],R_tr[0][5], R_tr[0][6]])
    
    f.write("Best test Chi-square:"+str(R_te[0][test+len(list_params)])+"(number of basis points, delta, theta,lambdaa)= "+str(int(R_te[0][2]))+ " {:.2f}".format(R_te[0][4])+" {:.2f}".format(R_te[0][5])+ " {:.3f}".format(R_te[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[0][0], R_te[0][1], R_te[0][2], R_te[0][3], R_te[0][4],R_te[0][5], R_te[0][6],R_te[0][7], R_te[0][8], R_te[0][-2], R_te[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[1][0], R_te[1][1], R_te[1][2], R_te[1][3], R_te[1][4],R_te[1][5], R_te[1][6],R_te[1][7], R_te[1][8], R_te[1][-2], R_te[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[2][0], R_te[2][1], R_te[2][2], R_te[2][3], R_te[2][4],R_te[2][5], R_te[2][6],R_te[2][7], R_te[2][8], R_te[2][-2], R_te[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[3][0], R_te[3][1], R_te[3][2], R_te[3][3], R_te[3][4],R_te[3][5], R_te[3][6],R_te[3][7], R_te[3][8], R_te[3][-2], R_te[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[4][0], R_te[4][1], R_te[4][2], R_te[4][3], R_te[4][4],R_te[4][5], R_te[4][6],R_te[4][7], R_te[4][8], R_te[4][-2], R_te[4][-1] )))
    f.write('\n')       
    f.write('\n') 
    Best_models[0][3][:] = np.array([ R_te[0][2], R_te[0][4],R_te[0][5], R_te[0][6]])
    frames = 4000
    make_plot_of_best_f(W_tr,W_te,R_tr,R_te, False, frames)
    make_plot_of_best_u(W_tr,W_te,R_tr,R_te, False, frames)
    
    W_R = list(np.hstack((Wass_Tr,Res_Tr)))
    W_R.sort(key = lambda x: x[train])
    W_tr = W_R[:][:]
    W_R.sort(key = lambda x: x[test] )
    W_te = W_R[:][:]
    W_R.sort(key = lambda x: (x[train+len(list_params)]) )
    R_tr = W_R[:][:]
    W_R.sort(key = lambda x: (x[test+len(list_params)]) )
    R_te = W_R[:][:]
    print('Model with a Regularization matrix')
    print("Best train Wasserstein_distance:",W_tr[0][train],"(number of basis points, delta, theta,lambdaa)=",int(W_tr[0][2]), " {:.2f}".format(W_tr[0][4])," {:.2f}".format(W_tr[0][5]), " {:.2f}".format(W_tr[0][6]) )
    print("Best test Wasserstein_distance:",W_te[0][test],"(number of basis points, delta, theta,lambdaa)=",int(W_te[0][2]), " {:.2f}".format(W_te[0][4])," {:.2f}".format(W_te[0][5]), " {:.2f}".format(W_te[0][6]) )
    print("Best train Chi-square:",R_tr[0][train+len(list_params)],"(number of basis points, delta, theta,lambdaa)=",int(R_tr[0][2]), " {:.2f}".format(R_tr[0][4])," {:.2f}".format(R_tr[0][5]), " {:.2f}".format(R_tr[0][6]) )
    print("Best test Chi-square:",R_te[0][test+len(list_params)],"(number of basis points, delta, theta,lambdaa)=",int(R_te[0][2]), " {:.2f}".format(R_te[0][4])," {:.2f}".format(R_te[0][5]), " {:.2f}".format(R_te[0][6]) )
    
    
    f.write('Model with a Regularization matrix')
    f.write('\n')
    f.write("Best train Wasserstein_distance:"+str(W_tr[0][train])+"(number of basis points, delta, theta,lambdaa)= "+str(int(W_tr[0][2]))+ " {:.2f}".format(W_tr[0][4])+" {:.2f}".format(W_tr[0][5])+ " {:.3f}".format(W_tr[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[0][0], W_tr[0][1], W_tr[0][2], W_tr[0][3], W_tr[0][4],W_tr[0][5], W_tr[0][6],W_tr[0][7],W_tr[0][8], W_tr[0][-2], W_tr[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[1][0], W_tr[1][1], W_tr[1][2], W_tr[1][3], W_tr[1][4],W_tr[1][5], W_tr[1][6],W_tr[1][7],W_tr[1][8], W_tr[1][-2], W_tr[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[2][0], W_tr[2][1], W_tr[2][2], W_tr[2][3], W_tr[2][4],W_tr[2][5], W_tr[2][6],W_tr[2][7],W_tr[2][8], W_tr[2][-2], W_tr[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[3][0], W_tr[3][1], W_tr[3][2], W_tr[3][3], W_tr[3][4],W_tr[3][5], W_tr[3][6],W_tr[3][7],W_tr[3][8], W_tr[3][-2], W_tr[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_tr[4][0], W_tr[4][1], W_tr[4][2], W_tr[4][3], W_tr[4][4],W_tr[4][5], W_tr[4][6],W_tr[4][7],W_tr[4][8], W_tr[4][-2], W_tr[4][-1] )))
    f.write('\n')
    Best_models[1][0][:] = np.array([ W_tr[0][2], W_tr[0][4],W_tr[0][5], W_tr[0][6]])
    
    f.write("Best test Wasserstein_distance:"+str(W_te[0][test])+"(number of basis points, delta, theta,lambdaa)= "+str(int(W_te[0][2]))+ " {:.2f}".format(W_te[0][4])+" {:.2f}".format(W_te[0][5])+ " {:.3f}".format(W_te[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[0][0], W_te[0][1], W_te[0][2], W_te[0][3], W_te[0][4],W_te[0][5], W_te[0][6],W_te[0][7], W_te[0][8], W_te[0][-2], W_te[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[1][0], W_te[1][1], W_te[1][2], W_te[1][3], W_te[1][4],W_te[1][5], W_te[1][6],W_te[1][7],W_te[1][8], W_te[1][-2], W_te[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[2][0], W_te[2][1], W_te[2][2], W_te[2][3], W_te[2][4],W_te[2][5], W_te[2][6],W_te[2][7],W_te[2][8], W_te[2][-2], W_te[2][-1] )))
    f.write('\n') 
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[3][0], W_te[3][1], W_te[3][2], W_te[3][3], W_te[3][4],W_te[3][5], W_te[3][6],W_te[3][7],W_te[3][8], W_te[3][-2], W_te[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(W_te[4][0], W_te[4][1], W_te[4][2], W_te[4][3], W_te[4][4],W_te[4][5], W_te[4][6],W_te[4][7],W_te[4][8], W_te[4][-2], W_te[4][-1] )))
    f.write('\n') 
    Best_models[1][1][:] = np.array([ W_te[0][2], W_te[0][4],W_te[0][5], W_te[0][6]])
    
    f.write("Best train Chi-square:"+str(R_tr[0][train+len(list_params)])+"(number of basis points, delta, theta,lambdaa)= "+str(int(R_tr[0][2]))+ " {:.2f}".format(R_tr[0][4])+" {:.2f}".format(R_tr[0][5])+ " {:.3f}".format(R_tr[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[0][0], R_tr[0][1], R_tr[0][2], R_tr[0][3], R_tr[0][4],R_tr[0][5], R_tr[0][6],R_tr[0][7], R_tr[0][8], R_tr[0][-2], R_tr[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[1][0], R_tr[1][1], R_tr[1][2], R_tr[1][3], R_tr[1][4],R_tr[1][5], R_tr[1][6],R_tr[1][7], R_tr[1][8], R_tr[1][-2], R_tr[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[2][0], R_tr[2][1], R_tr[2][2], R_tr[2][3], R_tr[2][4],R_tr[2][5], R_tr[2][6],R_tr[2][7], R_tr[2][8], R_tr[2][-2], R_tr[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[3][0], R_tr[3][1], R_tr[3][2], R_tr[3][3], R_tr[3][4],R_tr[3][5], R_tr[3][6],R_tr[3][7], R_tr[3][8], R_tr[3][-2], R_tr[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_tr[4][0], R_tr[4][1], R_tr[4][2], R_tr[4][3], R_tr[4][4],R_tr[4][5], R_tr[4][6],R_tr[4][7], R_tr[4][8], R_tr[4][-2], R_tr[4][-1] )))
    f.write('\n')
    Best_models[1][2][:] = np.array([ R_tr[0][2], R_tr[0][4],R_tr[0][5], R_tr[0][6]])
    
    f.write("Best test Chi-square:"+str(R_te[0][test+len(list_params)])+"(number of basis points, delta, theta,lambdaa)= "+str(int(R_te[0][2]))+ " {:.2f}".format(R_te[0][4])+" {:.2f}".format(R_te[0][5])+ " {:.3f}".format(R_te[0][6]) )
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[0][0], R_te[0][1], R_te[0][2], R_te[0][3], R_te[0][4],R_te[0][5], R_te[0][6],R_te[0][7], R_te[0][8], R_te[0][-2], R_te[0][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[1][0], R_te[1][1], R_te[1][2], R_te[1][3], R_te[1][4],R_te[1][5], R_te[1][6],R_te[1][7], R_te[1][8], R_te[1][-2], R_te[1][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[2][0], R_te[2][1], R_te[2][2], R_te[2][3], R_te[2][4],R_te[2][5], R_te[2][6],R_te[2][7], R_te[2][8], R_te[2][-2], R_te[2][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[3][0], R_te[3][1], R_te[3][2], R_te[3][3], R_te[2][4],R_te[3][5], R_te[3][6],R_te[3][7], R_te[3][8], R_te[3][-2], R_te[3][-1] )))
    f.write('\n')
    f.write(''.join('{} {} {} {} {} {} {} {} {} {} {}'.format(R_te[4][0], R_te[4][1], R_te[4][2], R_te[4][3], R_te[2][4],R_te[4][5], R_te[4][6],R_te[4][7], R_te[4][8], R_te[4][-2], R_te[4][-1] )))
    f.write('\n')
    f.write('\n') 
    Best_models[0][3][:] = np.array([ R_te[0][2], R_te[0][4],R_te[0][5], R_te[0][6]])
      
    
    f.close()
    make_plot_of_best_f(W_tr,W_te,R_tr,R_te, True, frames)
    make_plot_of_best_u(W_tr,W_te,R_tr,R_te, True, frames)
    
    return Best_models
    
def make_plot_of_best_f(W_tr,W_te,R_tr,R_te, Reg, frames):
    if Reg == True:
        file = 'data_Fp/reg_matrx/forceskernels_'
    else:
        file = 'data_Fp/reg_iden/forceskernels_'
        
    file_W_tr = pd.read_csv(str(file)+str(frames)+'_'+str(int(W_tr[0][2]))+'_('+str(W_tr[0][4])+','+str(W_tr[0][5])+','+str(W_tr[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    W_tr_f = file_W_tr[:,1]
    W_tr_c = file_W_tr[:,0]
    
    '''file_W_te = pd.read_csv(str(file)+str(frames)+'_'+str(int(W_te[0][2]))+'_('+str(W_te[0][4])+','+str(W_te[0][5])+','+str(W_te[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    W_te_f = file_W_te[:,1]
    W_te_c = file_W_te[:,0]'''
    
    file_R_tr = pd.read_csv(str(file)+str(frames)+'_'+str(int(R_tr[0][2]))+'_('+str(R_tr[0][4])+','+str(R_tr[0][5])+','+str(R_tr[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    R_tr_f = file_R_tr[:,1]
    R_tr_c = file_R_tr[:,0]
    
    '''file_R_te = pd.read_csv(str(file)+str(frames)+'_'+str(int(R_te[0][2]))+'_('+str(R_te[0][4])+','+str(R_te[0][5])+','+str(R_te[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    R_te_f = file_R_te[:,1]
    R_te_c = file_R_te[:,0]'''
    if plot_splines:
        Splines = pd.read_csv('data_Fp/reg_iden/forcessplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
        
    if plot_LJ:
        LJ = pd.read_csv('data_Fp/reg_iden/forcesLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
    
    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel(r'F(r)(kcal/mol/$\AA$)')
     
    
    plt.plot(W_tr_c, W_tr_f, color='g',linestyle = 'dashed', label='Best Wasserstein distance model: '+str(int(W_te[0][2]))+'_('+"{:.1f}".format(W_te[0][4])+','+"{:.1f}".format(W_te[0][5])+','+str(W_te[0][6])+')')# "{:.1f}".format(W_te[0][5])+')')
    plt.plot(R_tr_c, R_tr_f, color='b',linestyle = 'dashed', label='Best Chi-square model: '+str(int(R_te[0][2]))+'_('+"{:.1f}".format(R_te[0][4])+','+"{:.1f}".format(R_te[0][5])+','+str(R_te[0][6])+')')#"{:.1f}".format(R_te[0][5])+')')
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.title(r"Compare the Pair Forces of Best Models (Kernels(basis points($\delta,\theta,\lambda$)))")
    plt.legend()
    plt.savefig('plot_compare/'+'_'+str(frames)+str(Reg)+'_compareForce_Best'+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show() 
    plt.close()
    

def make_plot_of_best_u(W_tr,W_te,R_tr,R_te, Reg, frames):
    if Reg == True:
        file = 'data_Up/reg_matrx/potentialkernels_'
    else:
        file = 'data_Up/reg_iden/potentialkernels_'
        
    file_W_tr = pd.read_csv(str(file)+str(frames)+'_'+str(int(W_tr[0][2]))+'_('+str(W_tr[0][4])+','+str(W_tr[0][5])+','+str(W_tr[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    W_tr_f = file_W_tr[:,1]
    W_tr_c = file_W_tr[:,0]
    
    '''file_W_te = pd.read_csv(str(file)+str(frames)+'_'+str(int(W_te[0][2]))+'_('+str(W_te[0][4])+','+str(W_te[0][5])+','+str(W_te[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    W_te_f = file_W_te[:,1]
    W_te_c = file_W_te[:,0]'''
    
    file_R_tr = pd.read_csv(str(file)+str(frames)+'_'+str(int(R_tr[0][2]))+'_('+str(R_tr[0][4])+','+str(R_tr[0][5])+','+str(R_tr[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    R_tr_f = file_R_tr[:,1]
    R_tr_c = file_R_tr[:,0]
    
    '''file_R_te = pd.read_csv(str(file)+str(frames)+'_'+str(int(R_te[0][2]))+'_('+str(R_te[0][4])+','+str(R_te[0][5])+','+str(R_te[0][6])+')resc_False_.txt', delimiter='\s+', header=None).values
    R_te_f = file_R_te[:,1]
    R_te_c = file_R_te[:,0]'''
    if plot_splines:
        Splines = pd.read_csv('data_Up/reg_iden/potentialsplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
        
    if plot_LJ:
        LJ = pd.read_csv('data_Up/reg_iden/potentialLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
    
    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel('U(r)(kcal/mol)')
     
    
    plt.plot(W_tr_c, W_tr_f, color='g',linestyle = 'dashed', label='Best Wasserstein distance model: '+str(int(W_te[0][2]))+'_('+"{:.1f}".format(W_te[0][4])+','+"{:.1f}".format(W_te[0][5])+','+str(W_te[0][6])+')')# "{:.1f}".format(W_te[0][5])+')')
    plt.plot(R_tr_c, R_tr_f, color='b',linestyle = 'dashed', label='Best Chi-square model: '+str(int(R_te[0][2]))+'_('+"{:.1f}".format(R_te[0][4])+','+"{:.1f}".format(R_te[0][5])+','+str(R_te[0][6])+')')#"{:.1f}".format(R_te[0][5])+')')
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.title(r"Compare the Pair Potential of Best Models (Kernels(basis points($\delta,\theta,\lambda$)))")
    plt.legend()
    plt.savefig('plot_compare/'+'_'+str(frames)+str(Reg)+'_comparePotential_Best'+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show() 
    plt.close()
    
def Compare_best_models_f(Best_models):
    file_I = 'data_Fp/reg_iden/forceskernels_'+str(frames)+'_'
    file_K = 'data_Fp/reg_matrx/forceskernels_'+str(frames)+'_'
    
    if plot_splines:
        Splines = pd.read_csv('data_Fp/reg_iden/forcessplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
    if plot_LJ:
        LJ = pd.read_csv('data_Fp/reg_iden/forcesLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
        
    Models_I = Best_models[0]
    Models_K = Best_models[1]


    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel(r'F(r)(kcal/mol/$\AA$)')
    
    colors = ['b','m']
    labels_I = [r'Best Was.Dist model with $\lambda$:', r'Best Chi-square model with $\lambda$:']
    for i in range(int(len(Models_I)/2)):
        file_W_te = pd.read_csv(str(file_I)+str(int(Models_I[i*2][0]))+'_('+str(Models_I[i*2][1])+','+str(Models_I[i*2][2])+','+str(Models_I[i*2][3])+')resc_False_.txt', delimiter='\s+', header=None).values
        W_te_f = file_W_te[:,1]
        W_te_c = file_W_te[:,0]
        plt.plot(W_te_c, W_te_f, color=colors[i],linestyle = 'dashed', label=labels_I[i]+str(int(Models_I[i*2][0]))+'_('+str(Models_I[i*2][1])+','+str(Models_I[i*2][2])+','+str(Models_I[i*2][3])+')')
    
    colors = ['y','g']
    labels_K = [r'Best Was.Dist model with $K_{mn}$:',r'Best Chi-square model with $K_{mn}$:']
    for i in range(int(len(Models_K)/2)):
        file_W_te = pd.read_csv(str(file_K)+str(int(Models_K[i*2][0]))+'_('+str(Models_K[i*2][1])+','+str(Models_K[i*2][2])+','+str(Models_K[i*2][3])+')resc_False_.txt', delimiter='\s+', header=None).values
        W_te_f = file_W_te[:,1]
        W_te_c = file_W_te[:,0]
        plt.plot(W_te_c, W_te_f, color=colors[i],linestyle = 'dashed', label=labels_K[i]+str(int(Models_K[i*2][0]))+'_('+str(Models_K[i*2][1])+','+str(Models_K[i*2][2])+','+str(Models_K[i*2][3])+')')
    
    
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.title(r"Compare the Pair Forces of Best Models (Kernels(basis points($\delta,\theta,\lambda$)))")
    plt.legend()
    plt.savefig('plot_compare/all_cases_compareForce_Best'+'_'+str(frames)+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show() 
    plt.close()


def Compare_best_models_u(Best_models):
    file_I = 'data_Up/reg_iden/potentialkernels_'+str(frames)+'_'
    file_K = 'data_Up/reg_matrx/potentialkernels_'+str(frames)+'_'
    
    if plot_splines:
        Splines = pd.read_csv('data_Up/reg_iden/potentialsplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
    if plot_LJ:
        LJ = pd.read_csv('data_Up/reg_iden/potentialLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
        
    Models_I = Best_models[0]
    Models_K = Best_models[1]


    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel('U(r)(kcal/mol)')
    
    colors = ['b','m']
    labels_I = [r'Best Was.Dist model with $\lambda$:', r'Best Chi-square model with $\lambda$:']
    for i in range(int(len(Models_I)/2)):
        file_W_te = pd.read_csv(str(file_I)+str(int(Models_I[i*2][0]))+'_('+str(Models_I[i*2][1])+','+str(Models_I[i*2][2])+','+str(Models_I[i*2][3])+')resc_False_.txt', delimiter='\s+', header=None).values
        W_te_f = file_W_te[:,1]
        W_te_c = file_W_te[:,0]
        plt.plot(W_te_c, W_te_f, color=colors[i],linestyle = 'dashed', label=labels_I[i]+str(int(Models_I[i*2][0]))+'_('+str(Models_I[i*2][1])+','+str(Models_I[i*2][2])+','+str(Models_I[i*2][3])+')')
    
    colors = ['y','g']
    labels_K = [r'Best Was.Dist model with $K_{mn}$:',r'Best Chi-square model with $K_{mn}$:']
    for i in range(int(len(Models_K)/2)):
        file_W_te = pd.read_csv(str(file_K)+str(int(Models_K[i*2][0]))+'_('+str(Models_K[i*2][1])+','+str(Models_K[i*2][2])+','+str(Models_K[i*2][3])+')resc_False_.txt', delimiter='\s+', header=None).values
        W_te_f = file_W_te[:,1]
        W_te_c = file_W_te[:,0]
        plt.plot(W_te_c, W_te_f, color=colors[i],linestyle = 'dashed', label=labels_K[i]+str(int(Models_K[i*2][0]))+'_('+str(Models_K[i*2][1])+','+str(Models_K[i*2][2])+','+str(Models_K[i*2][3])+')')
    
    
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.title(r"Compare the Pair Potential of Best Models (Kernels(basis points($\delta,\theta,\lambda$)))")
    plt.legend()
    plt.savefig('plot_compare/all_cases_comparePotential_Best'+'_'+str(frames)+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show() 
    plt.close()



def make_plots_basis_points(Basis, best_res_tr, reg, measure ):
    data = np.zeros((300, 2*len(Basis)))
    if reg == 0:
        Reg = False
        file = 'data_Fp/reg_iden/forceskernels_'+str(frames)+'_'
        
    else:
        Reg = True
        file = 'data_Fp/reg_matrx/forceskernels_'+str(frames)+'_'
    for i in range(len(Basis)):
            
        file_W_tr = pd.read_csv(str(file)+str(Basis[i])+'_('+str(best_res_tr[0])+','+str(best_res_tr[1])+','+str(best_res_tr[2])+')resc_False_.txt', delimiter='\s+', header=None).values
        data[:,2*i] = file_W_tr[:,0]
        data[:,2*i+1] = file_W_tr[:,1]
    
    if plot_splines:
        Splines = pd.read_csv('data_Fp/reg_iden/forcessplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
    if plot_LJ:
        LJ = pd.read_csv('data_Fp/reg_iden/forcesLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
    colors = ['r', 'g', 'y', 'k', 'c', 'b']
    for i in range(len(Basis)):
        plt.plot(data[:,2*i], data[:,2*i+1], color=colors[i],linestyle = 'dashed', label='number of basis points: '+str(Basis[i]))
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel(r'F(r)(kcal/mol/$\AA$)')
    plt.title(r"Compare the Pair Forces of Best Models (Kernels($\delta,\theta,\lambda$)= ("+str(best_res_tr[0])+' ,'+str(best_res_tr[1])+' ,'+str(best_res_tr[2])+'))')
    plt.legend()
    plt.savefig('plot_compare/'+str(Reg)+'compareForce_points'+str(measure)+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')

    plt.show() 
    plt.close()
    return 0

def make_plots_delta(delta_l, best_res_tr, reg, measure):
    if reg == 0:
        Reg = False
        file = 'data_Fp/reg_iden/forceskernels_'+str(frames)+'_'
    else:
        Reg = True
        file = 'data_Fp/reg_matrx/forceskernels_'+str(frames)+'_'
    data = np.zeros((300,  2*len(delta_l)))
    for i in range(len(delta_l)):
        file_W_tr = pd.read_csv(str(file)+str(best_res_tr[0])+'_('+str(delta_l[i])+','+str(best_res_tr[1])+','+str(best_res_tr[2])+')resc_False_.txt', delimiter='\s+', header=None).values
        data[:,2*i] = file_W_tr[:,0]
        data[:,2*i+1] = file_W_tr[:,1]
    if plot_splines:
        Splines = pd.read_csv('data_Fp/reg_iden/forcessplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
    if plot_LJ:     
        LJ = pd.read_csv('data_Fp/reg_iden/forcesLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
    colors = ['r', 'g', 'y', 'k', 'c', 'b']
    for i in range(len(delta_l)):
        plt.plot(data[:,2*i], data[:,2*i+1], color=colors[i],linestyle = 'dashed', label=r'$\delta$: '+str(delta_l[i]))
    
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel(r'F(r)(kcal/mol/$\AA$)')
    plt.title(r"Compare the Pair Forces of Best Models (Kernels(number of basis points,$\theta,\lambda$)= ("+str(best_res_tr[0])+' ,'+str(best_res_tr[1])+' ,'+str(best_res_tr[2])+'))')
    plt.legend()
    plt.savefig('plot_compare/'+str(Reg)+'compareForce_delta'+str(measure)+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')

    plt.show() 
    plt.close()
    return 0

def make_plots_theta(theta,best_res_tr, reg, measure):
    if reg == 0:
        Reg = False
        file = 'data_Fp/reg_iden/forceskernels_'+str(frames)+'_'
    else:
        Reg = True
        file = 'data_Fp/reg_matrx/forceskernels_'+str(frames)+'_'
    data = np.zeros((300, 2*len(theta)))
    for i in range(len(theta)):
            
        file_W_tr = pd.read_csv(str(file)+str(best_res_tr[0])+'_('+str(best_res_tr[1])+','+str(theta[i])+','+str(best_res_tr[2])+')resc_False_.txt', delimiter='\s+', header=None).values
        data[:,2*i] = file_W_tr[:,0]
        data[:,2*i+1] = file_W_tr[:,1]
    if plot_splines:
        Splines = pd.read_csv('data_Fp/reg_iden/forcessplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
    if plot_LJ:
        LJ = pd.read_csv('data_Fp/reg_iden/forcesLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
    colors = ['r', 'g', 'y', 'k', 'c', 'b']
    for i in range(len(theta)):
        plt.plot(data[:,2*i], data[:,2*i+1], color=colors[i],linestyle = 'dashed', label=r'$\theta$: '+str(theta[i]))
    
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel(r'F(r)(kcal/mol/$\AA$)')
    plt.title(r"Compare the Pair Forces of Best Models (Kernels(number of basis points,$\delta,\lambda$)= ("+str(best_res_tr[0])+' ,'+str(best_res_tr[1])+' ,'+str(best_res_tr[2])+'))')
    plt.legend()
    plt.savefig('plot_compare/'+str(Reg)+'compareForce_theta'+str(measure)+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    
    plt.show() 
    plt.close()  
    return 0

def make_plots_lambda(lambdaa_l,best_res_tr, reg, measure):
    if reg == 0:
        Reg = False
        file = 'data_Fp/reg_iden/forceskernels_'+str(frames)+'_'
    else:
        Reg = True
        file = 'data_Fp/reg_matrx/forceskernels_'+str(frames)+'_'
    data = np.zeros((300, 2*len(lambdaa_l)))
    for i in range(len(lambdaa_l)):
        file_W_tr = pd.read_csv(str(file)+str(best_res_tr[0])+'_('+str(best_res_tr[1])+','+str(best_res_tr[2])+','+str(lambdaa_l[i])+')resc_False_.txt', delimiter='\s+', header=None).values
        data[:,2*i] = file_W_tr[:,0]
        data[:,2*i+1] = file_W_tr[:,1]
    if plot_splines:
        Splines = pd.read_csv('data_Fp/reg_iden/forcessplines_'+str(frames)+'_48_(-1.0,-1.0,0.0)resc_False_.txt', delimiter='\s+', header=None).values
        max_f = max(Splines[:,1])
        index = np.where( Splines[:,1]== max_f)
        Spli_f = Splines[index[0][0]:,1]
        Spli_c = Splines[index[0][0]:,0]
        
    if plot_LJ:
        LJ = pd.read_csv('data_Fp/reg_iden/forcesLJ_'+str(frames)+'_2resc_False_.txt', delimiter='\s+', header=None).values
        LJ_f = LJ[:,1]
        LJ_c = LJ[:,0]
    colors = ['r', 'g', 'y', 'k', 'c', 'b']
    for i in range(len(lambdaa_l)):
        plt.plot(data[:,2*i], data[:,2*i+1], color=colors[i],linestyle = 'dashed', label=r'$\lambda$: '+str(lambdaa_l[i]))
    if plot_splines:
        plt.plot(Spli_c, Spli_f, color='c',linestyle = 'dashed', label='Splines, 48, Linear basis')
    if plot_LJ:
        plt.plot(LJ_c, LJ_f, color='k',linestyle = 'dashed', label='Lennard Jones')
    
    plt.rcParams['font.size'] = '8'
    plt.xlabel(r'r($\AA$)')
    plt.ylabel(r'F(r)(kcal/mol/$\AA$)')
    plt.title(r"Compare the Pair Forces of Best Models (Kernels(number of basis points,$\delta,\theta$)= ("+str(best_res_tr[0])+' ,'+str(best_res_tr[1])+' ,'+str(best_res_tr[2])+'))')
    plt.legend()
    plt.savefig('plot_compare/'+str(Reg)+'compareForce_lambda'+str(measure)+'_'+str(plot_splines)+'_'+str(plot_LJ)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show() 
    plt.close()
    return 0 

if __name__ == '__main__': 
    frames = 100
    list_params = ['method', 'number of samples', 'number of basis points','Regular_param', 'delta', 'theta','lambdaa', 'train','test']
    number_of_basis_l = [15,20,48] 
    delta_l = [1.0,2.0,6.0]
    theta_l = [2.5,3.5,5.0]  
    lambdaa_l = [0.0, 0.1, 0.01, 0.001]
    
    if not os.path.exists('plot_compare'):
        os.mkdir('plot_compare')
    
    plot_splines = True
    plot_LJ = True
    
    ##------best models Plots-----------------------------------
    Best_models = Find_best_chi_wass(frames)
    
    Compare_best_models_f(Best_models)
    Compare_best_models_u(Best_models)
    
    #------print plots for different values of basis point------------------
    best = Best_models[0,0,1:]
    make_plots_basis_points(number_of_basis_l, best, 0, 'was')
    best = Best_models[1,0,1:]
    make_plots_basis_points(number_of_basis_l, best, 1, 'was')
    best = Best_models[0,2,1:]
    make_plots_basis_points(number_of_basis_l, best, 0, 'res')
    best = Best_models[1,2,1:]
    make_plots_basis_points(number_of_basis_l, best, 1, 'res')
    
    
    #------print plots for different values of delta------------------
    best = [int(Best_models[0,0,0])]+(list(Best_models[0,0,2:]))
    make_plots_delta(delta_l, best, 0, 'was')
    best = [int(Best_models[1,0,0])]+(list(Best_models[1,0,2:]))
    make_plots_delta(delta_l, best, 1, 'was')
    best = [int(Best_models[0,2,0])]+(list(Best_models[0,2,2:]))
    make_plots_delta(delta_l, best, 0, 'res')
    best = [int(Best_models[1,2,0])]+(list(Best_models[1,2,2:]))
    make_plots_delta(delta_l, best, 1, 'res')
    
    #------print plots for different values of theta------------------
    best = [int(Best_models[0,0,0])]+[(Best_models[0,0,1])]+[(Best_models[0,0,3])]
    make_plots_theta(theta_l, best, 0, 'was')
    best = [int(Best_models[1,0,0])]+[(Best_models[1,0,1])]+[(Best_models[1,0,3])]
    make_plots_theta(theta_l, best, 1, 'was')
    best = [int(Best_models[0,2,0])]+[(Best_models[0,2,1])]+[(Best_models[0,2,3])]
    make_plots_theta(theta_l, best, 0, 'res')
    best = [int(Best_models[1,2,0])]+[(Best_models[1,2,1])]+[(Best_models[1,2,3])]
    make_plots_theta(theta_l, best, 1, 'res')
    
    #make_plots_lambda(lambdaa_l)
    #------print plots for different values of lambda------------------
    best = [int(Best_models[0,0,0])]+(list(Best_models[0,0,1:3]))
    make_plots_lambda(lambdaa_l, best, 0, 'was')
    best = [int(Best_models[1,0,0])]+(list(Best_models[1,0,1:3]))
    make_plots_lambda(lambdaa_l, best, 1, 'was')
    best = [int(Best_models[0,2,0])]+(list(Best_models[0,2,1:3]))
    make_plots_lambda(lambdaa_l, best, 0, 'res')
    best = [int(Best_models[1,2,0])]+(list(Best_models[1,2,1:3]))
    make_plots_lambda(lambdaa_l, best, 1, 'res')
    
    frames = 4000
    
    Compare_best_models_f(Best_models)
    Compare_best_models_u(Best_models)
    