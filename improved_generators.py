import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from basic_tools import *
from numba import jit
class Levy_Point_Process:
    #This is the parent class to define a public method for the Gamma and tempered stable processes to give the output series

    def integrate(self,evaluation_points,x_series,t_series): #Scalar integrate function
        evaluation_points = np.array(evaluation_points)
        x_series = np.array(x_series)
        t_series = np.array(t_series)
        #print("x",np.shape(x_series),x_series)
        #print("t",np.shape(t_series),t_series)
        return [x_series[t_series < point].sum() for point in evaluation_points]
    def general_integrate(self,evaluation_points,x_series,t_series): #Integration for multi-dimensional time series
        evaluation_points = np.array(evaluation_points)
        x_series = np.array(x_series)
        t_series = np.array(t_series)

        # Initialize an empty list to store the results
        results = []

        # Iterate over each evaluation point
        for point in evaluation_points:
            # Create a mask for selecting data along the time axis where t_series < point
            mask = t_series < point

            # Use the mask to select data from x_series and sum along the time axis (axis=0)
            sum_over_time = np.sum(x_series[mask], axis=0)
            # Append the result to the results list
            results.append(sum_over_time)

        return np.array(results)
#The most important generator in this project. This has the built in Gamma generator different from previous one, It is correctly implemented and is the only one that passes the validity check
#Generator for Normal Gamma process. On top of Gamma process, reuqire additional muw and sigmaw parameters. Note that a built in Gamma generator is in this class. Raw data has additional return of the jump sizes and times of the process.
class normal_gamma_process(Levy_Point_Process):
    def __init__(self,beta,C,T,muw,sigmaw):
        self.beta = beta
        self.C = C
        self.T = T
        self.muw = muw
        self.sigmaw = sigmaw
    #Note that it is wrong to specify this as c, completely different from the truncation parameter. This c here is just some pre-specified number of jumps
    def generate_gamma_samples(self,evaluation_points,raw_data = False,c = 50): #This fucntion returns the gamma process samples by putting in the resolution N (number of points to be evaluated in T) c is the truncation parameter
          #The first step generates Poisson epochs in the time interval T
     #Since only the generation in unit time interval is correct, we simply repeat the generation several times to have the correct jump sizes and times
        x_list = [] #List of jump sizes
        jump_times = [] #List of jump times
        
        poisson_epochs = []
        current_time = 0 #This stores the current arrival time
        for i in range(c):
            #random_interval = np.random.exponential(1/self.T) #Drawing from a rate T  Poisson process
            random_interval = np.random.exponential(1/self.T)
            current_time += random_interval
            poisson_epochs.append(current_time)#Storing the current arrival time gives the newest epoch
        
        #Then iterate over the Poisson epochs to generate the samples
        new_x_list = []
        for i in poisson_epochs:
            x = 1/(self.beta*(np.exp(i/self.C)-1))
            p = (1+self.beta*x)*np.exp(-self.beta*x)
            if np.random.rand() <= p: #set n = 1 to have a Bernoulli generator, and p is the probability of getting a 1.
                x_list.append(x)
                new_x_list.append(x)

        #Then generate the jump times
        for x in new_x_list: #Or N_Ga?
            jump_times.append(np.random.uniform()*self.T)
        
        x_list = [jump for jump, time in zip(x_list, jump_times) if time < self.T]
        jump_times = [time for time in jump_times if time < self.T]
        if raw_data:
            return self.integrate(evaluation_points,x_list,jump_times),x_list,jump_times
        
        else:
            return self.integrate(evaluation_points,x_list,jump_times)

    def generate_samples(self,evaluation_points,raw_data = False,all_data = False):
        if raw_data:
            gamma_paths,subordinator_jumps,jump_times = self.generate_gamma_samples(evaluation_points,True)
            gamma_paths = np.array(gamma_paths) #These are the gamma paths at the evluation points
            #Only update paths after jump times
            NVM_jumps = np.ones(len(subordinator_jumps))*self.muw*subordinator_jumps+self.sigmaw*np.sqrt(subordinator_jumps)*np.random.randn(len(subordinator_jumps))
            
            return self.integrate(evaluation_points,NVM_jumps,jump_times), NVM_jumps,jump_times
        
        elif all_data:
            gamma_paths,subordinator_jumps,jump_times = self.generate_gamma_samples(evaluation_points,True)
            gamma_paths = np.array(gamma_paths) #These are the gamma paths at the evluation points
            #Only update paths after jump times
            NVM_jumps = np.ones(len(subordinator_jumps))*self.muw*subordinator_jumps+self.sigmaw*np.sqrt(subordinator_jumps)*np.random.randn(len(subordinator_jumps))
            
            return self.integrate(evaluation_points,NVM_jumps,jump_times), NVM_jumps,jump_times, subordinator_jumps
            
        else:
            gamma_paths,subordinator_jumps,jump_times = self.generate_gamma_samples(evaluation_points,True)
            gamma_paths = np.array(gamma_paths) #These are the gamma paths at the evluation points
            #Only update paths after jump times
            NVM_jumps = np.ones(len(subordinator_jumps))*self.muw*subordinator_jumps+self.sigmaw*np.sqrt(subordinator_jumps)*np.random.randn(len(subordinator_jumps))
            
            return self.integrate(evaluation_points,NVM_jumps,jump_times)

    def generate_normal_gamma_samples_from_joint(self,evaluation_points):
        N = len(evaluation_points)
        gamma_samples = np.array(self.generate_gamma_samples(evaluation_points))#These would be the Gamma samples along the process
        normal_gamma_samples = np.random.randn(N)*np.sqrt(gamma_samples) * self.sigmaw + gamma_samples * self.muw
        return normal_gamma_samples
        
    
        