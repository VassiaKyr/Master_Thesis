from __future__ import print_function
import pickle
from scipy.stats import wasserstein_distance
import numpy as np
import sys
import os.path
from params import *
from auxil import read_kernels
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats.kde import gaussian_kde
from sklearn.metrics import mean_squared_error
plt.rcParams['font.size'] = '13'

def Create_Pdf(train_forces,approx_forces_train,test_forces,approx_forces_test, max_frames, number_of_basis, delta, theta, lambdaa, reg, file_r):
    if not os.path.exists('pdf'):
        os.mkdir('pdf')
    if not os.path.exists('pdf/'+str(file_r)):
        os.mkdir('pdf/'+str(file_r))
    plt.rcParams['font.size'] = '10'
    plt.ylabel('Frequency')
    plt.xlabel('Forces')
    if method == 'kernels':
        plt.title('PDF of train data set ('+str(method)+','+str(max_frames)+','+str(number_of_basis)+','+str(delta)+','+str(theta)+','+str(lambdaa)+')')
    elif method == 'LJ':
        plt.title('Probability density function of train data set ('+str(method)+','+str(max_frames)+')')
    elif method == 'splines':
        plt.title('Probability density function of train data set ('+str(method)+','+str(max_frames)+','+str(number_of_basis)+')')

    
    
    x = np.linspace(-5,5,200)
    my_pdf_a = gaussian_kde(approx_forces_train)
    my_pdf_t = gaussian_kde(train_forces)
    plt.plot(x,my_pdf_a(x),'y', label='Approximate data set')
    plt.plot(x,my_pdf_t(x),'g', label='Target data set')
    
    
    plt.legend()
    plt.xlim([-6, 6])
    plt.savefig('pdf/'+str(file_r)+'/PDF_pair_dist_train'+str(method)+','+str(max_frames)+','+str(number_of_basis)+','+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show()
    plt.close()
    
    plt.rcParams['font.size'] = '10'
    plt.ylabel('Frequency')
    plt.xlabel('Forces')
    
    if method == 'kernels':
        plt.title('PDF of test data set ('+str(method)+','+str(max_frames)+','+str(number_of_basis)+','+str(delta)+','+str(theta)+','+str(lambdaa)+')')
    elif method == 'LJ':
        plt.title('Probability density function of test data set ('+str(method)+','+str(max_frames)+')')
    elif method == 'splines':
        plt.title('Probability density function of test data set ('+str(method)+','+str(max_frames)+','+str(number_of_basis)+')')

    x = np.linspace(-5,5,200)
    my_pdf_t = gaussian_kde(test_forces)
    my_pdf_a = gaussian_kde(approx_forces_test)
    plt.plot(x,my_pdf_a(x),'y', label='Approximate data set')
    plt.plot(x,my_pdf_t(x),'g', label='Target data set')
    
    plt.legend()
    plt.xlim([-6, 6])
    plt.savefig('pdf/'+str(file_r)+'/PDF_pair_dist_test'+str(method)+','+str(max_frames)+','+str(number_of_basis)+','+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show()
    plt.close()
        
        
        
def Create_scatter_plot(train_forces,approx_forces_train,test_forces,approx_forces_test, max_frames, number_of_basis, delta, theta, lambdaa, reg, file_r):
    if not os.path.exists('scatter'):
        os.mkdir('scatter')
    if not os.path.exists('scatter/'+str(file_r)):
        os.mkdir('scatter/'+str(file_r))
    plt.rcParams['font.size'] = '13'
    plt.scatter(train_forces, approx_forces_train)
    plt.title('Scatter plot of Forces - train dataset')
    plt.xlabel('F_target')
    plt.ylabel('F_approx')
    min_f, max_f = train_forces.min(), train_forces.max()
    x,y = [min_f, max_f], [min_f, max_f ]
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.plot(x,y, color='y')
    plt.savefig('scatter/'+str(file_r)+'/scatter_train_'+str(method)+','+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'_Forces.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show()
    plt.close()
    plt.rcParams['font.size'] = '13'
    plt.scatter(test_forces, approx_forces_test)
    plt.title('Scatter plot of Forces - test dataset')
    plt.xlabel('F_target')
    plt.ylabel('F_approx')
    min_f, max_f = test_forces.min(), test_forces.max()
    x,y = [min_f, max_f], [min_f, max_f ]
    plt.xlim([-10, 10])
    plt.ylim([-10, 10])
    plt.plot(x,y, color='y')
    plt.savefig('scatter/'+str(file_r)+'/scatter_test_'+str(method)+','+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'_Forces.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
    plt.show()
    plt.close()
        

#if __name__ == '__main__':
def GAP_train(max_frames, number_of_basis,delta,theta,lambdaa, reg):
    print('Parameters: frames, number_of_basis,delta,theta,lambdaa, reg =', max_frames, number_of_basis,delta,theta,lambdaa, reg)
    train_Kernels = pd.read_csv(train_Kernels_fname, delimiter='\s+', header=None).values
    test_Kernels = pd.read_csv(test_Kernels_fname, delimiter='\s+', header=None).values
    train_forces = pd.read_csv(train_forces_fname, delimiter='\s+', header=None).values
    test_forces = pd.read_csv(test_forces_fname, delimiter='\s+', header=None).values
    print(train_Kernels.shape, test_Kernels.shape)
    
    nconfs_trainK = train_Kernels.shape[0]//( 3 * nats)
    nconfs_trainf = train_forces.shape[0] // (nats)
    
    nconfs_train = min(nconfs_trainK, nconfs_trainf)
    train_Kernels = train_Kernels[0:nconfs_train*3*nats,:]
    train_forces = train_forces[0:nconfs_train*nats,:].reshape(-1)
    print("Loaded", nconfs_train, "training configurations")
    
    nconfs_testK =  test_Kernels.shape[0]//(3 * nats)
    nconfs_testf = test_forces.shape[0] // (nats)
    nconfs_test = min(nconfs_testK, nconfs_testf)
    test_Kernels = test_Kernels[0:nconfs_test*3*nats,:]
    test_forces = test_forces[0:nconfs_test*nats,:].reshape(-1)
    print("Loaded", nconfs_test, "test configurations")
    print("shape of train_Kernels",train_Kernels.shape)
    
    
    if reg:
        file_r = 'reg_matrx'
    else:
        file_r = 'reg_iden'
        
    #----------Linear Equations--------------
    #-----solve (A^T*A+ lambda*I)x=A^T*b
    KtK = np.dot(train_Kernels.T, train_Kernels)
    print(KtK.shape,"KtK")
    KtF = np.dot(train_Kernels.T, train_forces)
    print(KtF.shape,"KtF")
    if reg:
        basis_points = np.array(np.linspace(3.25,rcut,number_of_basis))
        K_mm = np.zeros((number_of_basis, number_of_basis))
        for i in range(number_of_basis):
            for j in range(number_of_basis):
                dist = (basis_points[i]-basis_points[j])**2
                K_mm[i][j] =(delta**2)*np.exp(-dist/(2*theta**2))
        KtK_l = KtK +  lambdaa * K_mm
    else:
        KtK_l = KtK + lambdaa * np.identity(len(train_Kernels[0]), dtype = float)
         
    
    #----Find parameter alpha----------------
    a = np.linalg.solve(KtK_l, KtF)
    print("a:",a, a.shape)
    
    
    ##------save parameter a into a file-------------------
    if not os.path.exists('save_a'):
        os.mkdir('save_a')
    if not os.path.exists('save_a/'+str(file_r)):
        os.mkdir('save_a/'+str(file_r))
    f = open('save_a/'+str(file_r)+'/'+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'_parameters_a.txt', 'wb')
    pickle.dump(a, f)
    f.close()
    
    ##------save kernels matrices into a file---------------------
    if not os.path.exists('save_kernels'):
        os.mkdir('save_kernels')
    if not os.path.exists('save_kernels/'+str(file_r)):
        os.mkdir('save_kernels/'+str(file_r))
    f = open('save_kernels/'+str(file_r)+'/'+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'_train.txt', 'wb')
    pickle.dump(train_Kernels, f)
    f.close()
    f = open('save_kernels/'+str(file_r)+'/'+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'_test.txt', 'wb')
    pickle.dump(test_Kernels, f)
    f.close()
    
    ##-----------save the Chi_Square------------
    approx_forces_train = np.dot(train_Kernels,a)
    approx_forces_test = np.dot(test_Kernels,a)
    Chi_Square_tr = np.subtract(train_forces,approx_forces_train)
    Chi_Square_tr = (np.dot(Chi_Square_tr.T, Chi_Square_tr)) /(nats*3*nconfs_train)
    Chi_Square_te = np.subtract(test_forces ,approx_forces_test)
    Chi_Square_te = (np.dot(Chi_Square_te.T, Chi_Square_te)) /(nats*3*nconfs_test)
    f = open('0Chi-Square.txt','a')
    f.write(str(method)+','+str(max_frames)+','+str(number_of_basis)+','+str(reg)+','+str(delta)+','+str(theta)+','+str(lambdaa)+','+str(Chi_Square_tr)+','+str(Chi_Square_te))
    f.write("\n")
    f.close()
    
    ##-----------save the wasserstein metric------------
    Wass_trai = wasserstein_distance(train_forces,approx_forces_train)
    Wass_test = wasserstein_distance(test_forces,approx_forces_test)
    f = open('0Wasserstein_distance_new.txt','a')
    f.write(str(method)+','+str(max_frames)+','+str(number_of_basis)+','+str(reg)+','+str(delta)+','+str(theta)+','+str(lambdaa)+','+str(Wass_trai)+','+str(Wass_test))
    f.write("\n")
    f.close()
    
    ###-----------create F_pair-plots-----------
    
    x_axis = np.linspace(3.25,12.0,300)
    
    if not os.path.exists('plots_Fp'):
        os.mkdir('plots_Fp')
    if not os.path.exists('data_Fp'):
        os.mkdir('data_Fp')   
    if not os.path.exists('plots_Fp/'+str(file_r)):
        os.mkdir('plots_Fp/'+str(file_r))
    if not os.path.exists('data_Fp/'+str(file_r)):
        os.mkdir('data_Fp/'+str(file_r))
    
    if method == "kernels":
        f = open("data_Fp/"+str(file_r)+"/forces"+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+")resc_"+str(rescale)+"_.txt", "w")
        basis_points = np.array(np.linspace(3.25,rcut,number_of_basis))
        F = []
        for i in x_axis:
            k = 0
            for j in range(len(basis_points)):
                dist = (i-basis_points[j])**2
                k += a[j]*(delta**2)*np.exp(-dist/(2*theta**2))
            F.append(k)
            f.write(''.join('{} {}'.format(i,k)))
            f.write('\n')
            
        f.close()
        plt.rcParams['font.size'] = '13'
        plt.xlabel('distance(Α)')
        plt.ylabel('F[amuA/fs^2]')
        plt.plot(x_axis, F, label=r'$\theta$='+str(theta)+r', $\delta$='+str(delta)+r', $\lambda$='+str(lambdaa))
        plt.legend(loc="upper right")
        plt.savefig('plots_Fp/'+str(file_r)+'/'+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+')resc_'+str(rescale)+'_Forces.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()
        plt.close()
        
    if method == "splines":
        f = open("data_Fp/"+str(file_r)+"/forces"+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+'_('+str(delta)+','+str(theta)+','+str(lambdaa)+")resc_"+str(rescale)+"_.txt", "w")
        
        basis_points=np.array(np.linspace(3.25,rcut,number_of_basis))
        dx = basis_points[1]-basis_points[0]
        F = []
        for i in x_axis:
            k = 0
            for j in range(1,number_of_basis-1):
                if i<= basis_points[j] and i>basis_points[j-1] and j>=1:
                    k += a[j]*(i-basis_points[j-1])*(1./dx)
                    
                elif i> basis_points[j] and i<=basis_points[j+1] and j<number_of_basis:
                    k += a[j]*(basis_points[j+1]-i)*(1./dx)
                            
            F.append(k)
            f.write(''.join('{} {}'.format(i,k)))
            f.write('\n')
            
        f.close()
        plt.rcParams['font.size'] = '13'
        plt.xlabel('distance(Α)')
        plt.ylabel('F[amuA/fs^2]')
        
        plt.plot(x_axis, F, color='r',linestyle = 'dashed', label='F_spline')
        plt.title("Pair Forces")
        plt.legend()
        plt.savefig('plots_Fp/'+str(file_r)+'/'+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+')resc_'+str(rescale)+'_Forces.png',  dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()
        plt.close()  
        
    
    
        
    if method == "LJ":
        f = open("data_Fp/"+str(file_r)+"/forces"+str(method)+'_'+str(max_frames)+'_'+str(number_of_basis)+"resc_"+str(rescale)+"_.txt", "w")
        
        F = []
        for i in x_axis:
            dv = +(a[0]*6.0/(i**7)-a[1]*-12.0/(i**13))
            F.append(dv)
            f.write(''.join('{} {}'.format(i,dv)))
            f.write('\n')
            
        f.close()
        plt.rcParams['font.size'] = '13'
        plt.xlabel('distance(Α)')
        plt.ylabel('F[amuA/fs^2]')
        plt.plot(x_axis, F, color='r', label='F_LJ')
        plt.title("Pair Forces")
        plt.legend()
        plt.savefig('plots_Fp/'+str(file_r)+'/'+str(method)+'_'+str(max_frames)+'resc_'+str(rescale)+'_Forces.png', dpi = 300, pad_inches = .1, bbox_inches = 'tight')
        plt.show()
        plt.close()
    
        
    if create_scatter:
        Create_scatter_plot(train_forces,approx_forces_train,test_forces,approx_forces_test, max_frames, number_of_basis, delta, theta, lambdaa, reg, file_r)      
    
    if create_pdf:
        Create_Pdf(train_forces,approx_forces_train,test_forces,approx_forces_test, max_frames, number_of_basis, delta, theta, lambdaa, reg, file_r)

        
        