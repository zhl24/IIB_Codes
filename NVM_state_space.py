import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from jump_process_generators import *
import matplotlib.pyplot as plt


# 1D state space model driven by an NVM process, 
class SDE_1D_finite(Levy_Point_Process):
    def __init__(self,A,h,T,NVM, X_0 = 0):
        self.A = A
        self.h = h
        self.T = T
        self.NVM = NVM #An NVM Process Generator Object
        self.X_0 = X_0
    def generate_samples(self,evaluation_points,plot_NVM=False,plot_jumps=False,all_data = False): #The first argument being the ts to evaluate the process, and the second argument being the number of points in the summation
        
        NVM_paths, NVM_jumps,jump_times, subordinator_jumps = self.NVM.generate_samples(evaluation_points,all_data = True)
        samples = []
        for i,evaluation_point in enumerate(evaluation_points):
            system_jumps = NVM_jumps * np.exp(self.A*(evaluation_point-jump_times))*self.h
            samples.append(self.integrate([evaluation_point],system_jumps,jump_times)[0])#Extracting the single element at evaluation
        if plot_NVM:
            plt.figure()
            plt.plot(evaluation_points,NVM_paths)
            plt.title("NVM Process")
            plt.show()
        if plot_jumps:
            plt.figure()
            plt.plot(jump_times,NVM_jumps)
            plt.title("Subordinator Jumps")
            plt.show()
        if all_data:
            return samples,system_jumps,NVM_jumps,subordinator_jumps,jump_times
        else:
            return samples

 

#The general dimension NVM SDE
class SDE(Levy_Point_Process):
    def __init__(self,A,h,T,NVM, X_0 = 0):
        self.A = A #Define the state transition matrix A
        self.h = h #Define the stochastic transition matrix h
        self.T = T #Define the simulation time
        self.NVM = NVM #An NVM Process Generator Object
        self.X_0 = X_0
    #As beforem the evaluation points have to be specified
    def generate_samples(self,evaluation_points,plot_NVM=False,plot_jumps=False,all_data = False): #The first argument being the ts to evaluate the process, and the second argument being the number of points in the summation
        
        NVM_paths, NVM_jumps,jump_times, subordinator_jumps = self.NVM.generate_samples(evaluation_points,all_data = True)
        samples = []
        for i,evaluation_point in enumerate(evaluation_points):
            #system_jumps = NVM_jumps @ expm(A * (evaluation_point-jump_times)) @ self.h
            #system_jumps = NVM_jumps * np.exp(self.A*(evaluation_point-jump_times))*self.h
            system_jumps = []
            for j,jump_time in enumerate(jump_times):
                NVM_jump = NVM_jumps[j]
                system_jump = NVM_jump * expm(self.A * (evaluation_point-jump_time)) @ self.h
                system_jumps.append(list(system_jump))
                
            samples.append(self.general_integrate([evaluation_point],system_jumps,jump_times)[0])#Extracting the single element at evaluation
        samples = np.array(samples)
        # Squeeze the array to remove the last dimension
        if samples.shape[-1] == 1:
            samples = np.squeeze(samples, axis=-1)

        if plot_NVM:
            plt.figure()
            plt.plot(evaluation_points,NVM_paths)
            plt.title("NVM Process")
            plt.show()
        if plot_jumps:
            plt.figure()
            plt.plot(jump_times,NVM_jumps)
            plt.title("Subordinator Jumps")
            plt.show()
        if all_data:
            return samples,system_jumps,NVM_jumps,subordinator_jumps,jump_times
        else:
            return samples

 