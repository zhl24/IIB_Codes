import cProfile
import pstats
import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from jump_process_generators import *
import matplotlib.pyplot as plt
from NVM_state_space import *
from NVM_state_space_filters import *
from basic_tools import *
from scipy.special import logsumexp
from tqdm import tqdm  # 导入tqdm
import seaborn as sns



def main():


    #beta_pcn = 1 #The step parameter for the pre-conditioned crank nicolson algorithm
    n_iter = 10 #Number of iterations
    theta0 = -2 #Initial guess of theta
    beta0 = 2 #Initial Guess of beta
    C0 = 5  #Initial Guess of C  
    num_particles = 10

    #The 3 step sizes for the 3 parameters
    l_theta = 1 
    l_beta = 2
    l_C = 2





    true_theta = -1 #The true theta value for the Langevin system


    kw = 1 #The prior parameter for muw

    kv = 0.001 #The observation noise scaling factor. Note that this scaling factor applies to the variance
    true_sigmaw = 2
    true_muw = 0.2
    sigma_n = true_sigmaw * np.sqrt(kv)


    #Simulation Parameters

    true_beta = 5
    true_C = 10
    T = 10


    N = 10 * T  # Resolution, 10 points per unit time now

    #Define the Langevin dynamics
    A = np.zeros((2, 2))
    A[0, 1] = 1
    A[1, 1] = true_theta
    h = np.array([[0], [1]])

    #Simulation
    evaluation_points = np.linspace(0, T, N) #Note that this would be the time axis we work on.
    normal_gamma_generator = normal_gamma_process(true_beta, true_C, T, true_muw, true_sigmaw)
    langevin = SDE(A,h,T,normal_gamma_generator)
    #Noisy observation generation
    SDE_samples,system_jumps,NVM_jumps,subordinator_jumps,jump_times = langevin.generate_samples(evaluation_points,all_data=True)
    d1,d2 = np.shape(SDE_samples)
    #Noisy_samples = SDE_samples + np.random.randn(d1,d2)*sigma_n #The noisy observations simulated. Already in the column vector form


    #Create the partial observation, observing only the integral state x here.
    Noisy_samples = SDE_samples[:,0] + np.random.randn(np.shape(SDE_samples)[0])*sigma_n #The noisy observations simulated. Already in the column vector form


    #The collection of the parameter samples
    theta_samples = [theta0]
    C_samples = [C0]
    beta_samples = [beta0]
    first_time = True

    for iter in tqdm(range(n_iter), desc="Processing"):



        if first_time:
            #print("Progress:",progress/searching_resolution, "%")
            #progress +=1
            #Prior inverted gamma parameters for sigmaw
            alphaws = 2.1 * np.ones(num_particles)
            betaws = 1 * np.ones(num_particles)
            accumulated_Es = np.zeros(num_particles)
            accumulated_Fs = np.zeros(num_particles)

            trajectory = []
            A = np.zeros((2, 2))
            A[0, 1] = 1
            A[1, 1] = theta_samples[-1]

            #Kalman filter initialisation
            X0 = Noisy_samples[0]
            nx0 = 2
            X0 = np.zeros((nx0+1,1))
            nx0_new = 3

            #The margianlised Kalman covariance
            C_prior = np.zeros((nx0_new,nx0_new))
            C_prior[-1,-1] = kw

            g = np.array([[1],[0],[0]])
            g = g.T
            R = np.array([kv]) #The noise covariance matrix. Same observation noise throughout




            #Particle filter Initialisation    
            initial_particles = []
            for i in range(num_particles):
                initial_particles.append([np.zeros((nx0,1)),np.eye(nx0)])
            previous_Xs = []
            previous_X_uncertaintys = []
            for i in range(num_particles):
                previous_X_uncertaintys.append(C_prior)
                previous_Xs.append(X0)
            # Define initial weights (uniform)
            initial_weights = np.ones(num_particles) / num_particles

            

            # Time step size and sigma for the likelihood function
            dt = evaluation_points[1] - evaluation_points[0]
            matrix_exp = expm(A*dt)
            sigma = sigma_n #The observatin noise



            #Containers for the imnference results
            inferred_Xs = []
            inferred_covs = []
            first_time = True

            # Running the particle filter
            particles = initial_particles
            histories = [particles]
            weights = initial_weights




            sigmaw_values = []
            

            

            for i in range(len(evaluation_points)): #i is the time index we want for N
                #sigmaw here needs to be updated in every step
                sigmaw2,sigmaw2_uncertainty = inverted_gamma_to_mean_variance(alphaws, betaws,weights) #Note that this is sigmaw^2 but not sigmaw
                sigmaw = np.sqrt(sigmaw2)

                sigmaw_values.append(sigmaw)
                


                incremental_normal_gamma_generator = normal_gamma_process(beta_samples[-1], C_samples[-1], dt, 0, sigmaw) #We are just using the built in gamma generator inside， putting in some random muw or sigmaw has no effect
                incremental_SDE = SDE(A,h,dt,incremental_normal_gamma_generator)
                #print(i)
                t = evaluation_points[i]
                observation = Noisy_samples[i] #Note that the observation here is still a row vector, reshaping necessary
                previous_Xs, previous_X_uncertaintys,particles,weights,alphaws, betaws, accumulated_Es, accumulated_Fs,log_marginals  = ultimate_NVM_pf(observation, previous_Xs, previous_X_uncertaintys, particles, transition_function_ultimate_NVM_pf, matrix_exp, dt,incremental_SDE,g,R,alphaws,betaws,accumulated_Es,accumulated_Fs,i,return_log_marginals=True) # N is the time nidex
                inferred_cov = weighted_sum(previous_X_uncertaintys,weights) * sigmaw2 #Note that the original parameters are marginalised by sigmaw^2
                inferred_X = weighted_sum(previous_Xs,weights)
                
                histories.append(particles)
                
                inferred_Xs.append(inferred_X)
                inferred_covs.append(inferred_cov)
                log_marginals = np.array(log_marginals)

            original_state_log_probability = logsumexp(log_marginals) - np.log(num_particles)
            first_time = False





        #####################################################################################################################################################################################################################################################
        #From here, we propose a the parameter samples, indepndent parameter assumed

        theta_proposed = theta_samples[-1] + np.random.randn() * l_theta
        beta_proposed = beta_samples[-1] + np.random.randn() * l_beta
        C_proposed = C_samples[-1] + np.random.randn() * l_C
        #Note that the Gamma process parameters have to be positive
        if beta_proposed <=0:
            beta_proposed = 0.000001
        if C_proposed <= 0:
            C_proposeed = 0.000001
        #print("Progress:",progress/searching_resolution, "%")
        #progress +=1
        #Prior inverted gamma parameters for sigmaw
        alphaws = 2.1 * np.ones(num_particles)
        betaws = 1 * np.ones(num_particles)
        accumulated_Es = np.zeros(num_particles)
        accumulated_Fs = np.zeros(num_particles)

        trajectory = []
        A = np.zeros((2, 2))
        A[0, 1] = 1
        A[1, 1] = theta_proposed

        #Kalman filter initialisation
        X0 = Noisy_samples[0]
        nx0 = 2
        X0 = np.zeros((nx0+1,1))
        nx0_new = 3

        #The margianlised Kalman covariance
        C_prior = np.zeros((nx0_new,nx0_new))
        C_prior[-1,-1] = kw

        g = np.array([[1],[0],[0]])
        g = g.T
        R = np.array([kv]) #The noise covariance matrix. Same observation noise throughout




        #Particle filter Initialisation    
        initial_particles = []
        for i in range(num_particles):
            initial_particles.append([np.zeros((nx0,1)),np.eye(nx0)])
        previous_Xs = []
        previous_X_uncertaintys = []
        for i in range(num_particles):
            previous_X_uncertaintys.append(C_prior)
            previous_Xs.append(X0)
        # Define initial weights (uniform)
        initial_weights = np.ones(num_particles) / num_particles

        

        # Time step size and sigma for the likelihood function
        dt = evaluation_points[1] - evaluation_points[0]
        matrix_exp = expm(A*dt)
        sigma = sigma_n #The observatin noise



        #Containers for the imnference results
        inferred_Xs = []
        inferred_covs = []
        first_time = True

        # Running the particle filter
        particles = initial_particles
        histories = [particles]
        weights = initial_weights




        sigmaw_values = []
        

        for i in range(len(evaluation_points)): #i is the time index we want for N
            #sigmaw here needs to be updated in every step
            sigmaw2,sigmaw2_uncertainty = inverted_gamma_to_mean_variance(alphaws, betaws,weights)
            sigmaw = np.sqrt(sigmaw2)

            sigmaw_values.append(sigmaw)
        


            incremental_normal_gamma_generator = normal_gamma_process(beta_proposed, C_proposed, dt, 0, sigmaw) #We are just using the built in gamma generator inside， putting in some random muw or sigmaw has no effect
            incremental_SDE = SDE(A,h,dt,incremental_normal_gamma_generator)
            #print(i)
            t = evaluation_points[i]
            observation = Noisy_samples[i] #Note that the observation here is still a row vector, reshaping necessary
            previous_Xs, previous_X_uncertaintys,particles,weights,alphaws, betaws, accumulated_Es, accumulated_Fs,log_marginals  = ultimate_NVM_pf(observation, previous_Xs, previous_X_uncertaintys, particles, transition_function_ultimate_NVM_pf, matrix_exp, dt,incremental_SDE,g,R,alphaws,betaws,accumulated_Es,accumulated_Fs,i,return_log_marginals=True) # N is the time nidex
            inferred_cov = weighted_sum(previous_X_uncertaintys,weights) * sigmaw2 #Note that the original parameters are marginalised by sigmaw^2
            inferred_X = weighted_sum(previous_Xs,weights)
            
            histories.append(particles)
            
            inferred_Xs.append(inferred_X)
            inferred_covs.append(inferred_cov)
            log_marginals = np.array(log_marginals)

        proposed_state_log_probability = logsumexp(log_marginals) - np.log(num_particles)


        ##################################################################################################################################################################################################################################
        # Acceptance Attempt
        log_acceptance_ratio = proposed_state_log_probability - original_state_log_probability
        if np.log(np.random.rand())< log_acceptance_ratio: #Accepted case
            theta_samples.append(theta_proposed)
            beta_samples.append(beta_proposed)
            C_samples.append(C_proposed)
            original_state_log_probability = proposed_state_log_probability
        else: #Rejected case
            theta_samples.append(theta_samples[-1])
            beta_samples.append(beta_samples[-1])
            C_samples.append(C_samples[-1])


    theta_samples = np.array(theta_samples)  # 假设已经去除了燃烧期的样本
    beta_samples = np.array(beta_samples)
    C_samples = np.array(C_samples)

    # 后验分布摘要 for theta
    mean_theta = np.mean(theta_samples)
    median_theta = np.median(theta_samples)
    conf_interval = np.percentile(theta_samples, [2.5, 97.5])  # 95%置信区间




# 运行性能分析
profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()

# 输出性能报告
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats()