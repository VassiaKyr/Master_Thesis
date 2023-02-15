from params import *
from sep_train_test import sep_train_test
from prepare import prepare
from GAP_train import GAP_train
if method == 'kernels' and not add_trimer:
    for frame in max_frames_l:
        for number in number_of_basis_l:
            for delta in delta_l:
                for theta in theta_l:
                    prepare(frame,number,delta,theta)
                    #sep_train_test(number, train_test_s)
                    #for reg in regular_matrix:
                        #for lambdaa in lambdaa_l:
                        #    GAP_train(frame,number,delta,theta,lambdaa, reg)
if method == 'kernels' and  add_trimer:
    for frame in max_frames_l:
        for number in number_of_basis_l_tr:
            for delta in delta_l:
                for theta in theta_l_tr:
                    prepare(frame,number,delta,theta)
                    sep_train_test(number, train_test_s)
                    #for reg in regular_matrix:
                        #for lambdaa in lambdaa_l:
                        #    GAP_train(frame,number,delta,theta,lambdaa, reg)

if method != 'kernels': 
    reg = False
    for frame in max_frames_l:
        for number in number_of_basis_l:
            prepare(frame,number,-1.0,-1.0)
            sep_train_test(number, train_test_s)
            for lambdaa in lambdaa_l:
                GAP_train(frame,number,-1.0,-1.0,lambdaa, reg)