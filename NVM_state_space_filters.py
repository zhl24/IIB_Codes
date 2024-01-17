import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from jump_process_generators import *
import matplotlib.pyplot as plt




def Kalman_Predict(x_mean,x_variance,x_transition,x_noise_variance,x_noise_mean = 0): #Hidden state x predict step. 
#Note that the noise is generally not zero mean in our case.
#This function takes the mean and variance of the current hidden state and the transition architecture to predict the next state mean and variance
    return x_mean*x_transition+x_noise_mean,x_variance*x_transition**2 + x_noise_variance

def Kalman_Correct(x_mean,x_variance,observation,emission,emission_noise_variance):
    I = observation - emission * x_mean
    return x_mean + (emission * x_variance)/(emission**2 * x_variance + emission_noise_variance)*I, x_variance * (1-emission**2 * x_variance/(emission**2*x_variance+emission_noise_variance))
#evaluation points just form the time axis
#We then simply need the subordinator jumps and jump times to compute the x_noise mean and variances
#emission is simply 1 in our model. emission_noise_variance is just the simga_n term in the model

def Noisy_Normal_Gamma_SDE_Filter(noisy_sequence,observation_noise_var,SDE,subordinator_jumps,jump_times,evaluation_points,Kalman_Predict,Kalman_Correct,x_mean_prior=0,x_variance_prior=1):
    A = SDE.A
    h = SDE.h
    T = SDE.T
    X0 = x_mean_prior
    var0 = x_variance_prior
    NVM_generator = SDE.NVM
    muw = NVM_generator.muw
    sigmaw = NVM_generator.sigmaw 
    s = 0
    #The priors would be removed after inference
    inferred_x = []
    inferred_var = [] 
    for index,evaluation_point in enumerate(evaluation_points):
        x_transition = np.exp(A*(evaluation_point-s))
        x_noise_mean = 0
        x_noise_variance = 0
        for i in range(len(jump_times)):
             jump = subordinator_jumps[i]
             jump_time = jump_times[i]
             if s<=jump_time<=evaluation_point:
                x_noise_mean += jump*np.exp(A*(evaluation_point-jump_time))*h
                x_noise_variance += jump*np.exp(2*A*(evaluation_point-jump_time))*h**2
        x_noise_mean = x_noise_mean * muw
        x_noise_variance = x_noise_variance * sigmaw**2
        if index != 0:
             mean_predict,var_predict = Kalman_Predict(inferred_x[-1],inferred_var[-1],x_transition,x_noise_variance,x_noise_mean)
        else:
            mean_predict,var_predict = Kalman_Predict(X0,var0,x_transition,x_noise_variance,x_noise_mean)
        mean_correct,var_correct = Kalman_Correct(mean_predict,var_predict,noisy_sequence[index],1,observation_noise_var)
        inferred_x.append(mean_correct)
        inferred_var.append(var_correct)
        s = evaluation_point
    return inferred_x,inferred_var
        




#The most general multi-dimensional Kalman filtering functions for the NVM SDE model:
#Here we define a single step in Kalman filter such that it can 
# We just need to define f, g, W (mw,Q) and V (mv,R)

#The first two inputs are the current state. f is the transition matrix, mw is the state noise mean and Q is the state noise covariance

def Kalman_transit(X,P,f,Q,mw = 0,B=0,u=0):
    if B == 0 :
        return f@X+mw, f@P@f.transpose()+Q
    else:
        return f@X+mw + B@u, f@P@f.transpose()+Q
#We again need the current states as the first two inputs, but now we need an obervation. g is the emission matrix, mv and R are 
#the observation mean and noises which determine most of the Kalman filtering difficulties
def Kalman_correct(X,P,Y,g,R,mv = 0):
    Ino = Y - g @ X -mv#The innovation term. Note that for the biased noise case there would be a subtraction here. All other terms are not affected
    S = g @ P @ g.transpose() + R #The innovation covariance
    K = P @ g.transpose() @ np.linalg.inv(S)
    n = np.shape(P)[0]
    I = np.identity(n)
    return X + K@Ino,(I-K@g)@P






# Bootstrap Particle Filtering in General State Space Model in one step. Particle filteirng is itself by nature a general dimensional method. Since pdf maps vector samples into scalar probabilitiei.e. weights
#The most important point about filtering in the state spce model is the knowledge of time, t and dt.
#Just need to additionally define the transition function to simulate the particles forwards, and likelihood function to compute the particle probabilities given the observations. Here we just assume standard normal noise
def bootstrap_particle_filtering(observation, particles, weights, transition_function, likelihood_function,dt,t,sigma):
    num_particles = len(particles)
    # Transition step: move each particle according to the transition model
    particles = transition_function(particles,dt,t)

    # Compute weights based on observation likelihood
    weights = likelihood_function(particles,observation,sigma) * weights #Th previous weights are always the uniform distribution
    weights = weights/np.sum(weights) #Normalization step
    # Resampling step: resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices] #The resample particles are returned
    weights = np.ones(num_particles)/num_particles
    return particles,weights





#Functions for particle filtering in the Levy state space system.
def transition_function_exact_case(particles,dt,matrix_exp): #dt is the length of forwards simulation. t is the evaluation point
    new_particles = []
    #We assume first that we know the exact generator for the process
    theta = -2 #The main control parameter for the Lagevin system
    beta = 5
    C = 0.1
    T = dt
    muw = 0
    sigmaw = 1
    #Define the Langevin dynamics
    A = np.zeros((2, 2))
    A[0, 1] = 1
    A[1, 1] = theta
    h = np.array([[0], [1]])

    #Simulation over dt interval to obtain the jumps and jump times
    evaluation_points = [dt] #Note that this would be the time axis we work on.
    normal_gamma_generator = normal_gamma_process(beta, C, T, muw, sigmaw)
    for particle in particles:
        ng_paths,ng_jumps,jump_times = normal_gamma_generator.generate_samples(evaluation_points,raw_data = True)
        #The jumpss and time already before dt
        

        #Then we solve for the summation
        system_jumps = []
        if len(jump_times)>1:
            for ng_jump,jump_time in zip(ng_jumps,jump_times):
                system_jump = ng_jump * expm(-A * jump_time) @ h
                system_jumps.append(system_jump)
            # Use the mask to select data from x_series and sum along the time axis (axis=0)
            sum_over_time = np.sum(system_jumps, axis=0)
            sum_over_time = np.squeeze(sum_over_time)
        elif len(jump_times) == 1:
            system_jump = ng_jumps * expm(-A * jump_times) @ h
            system_jumps.append(system_jump)
            sum_over_time = np.sum(system_jumps, axis=0)
            sum_over_time = np.squeeze(sum_over_time)
        else:
    # Initialize sum_over_time as a zero array of the same shape as a particle
            sum_over_time = np.zeros(2)
            sum_over_time = np.squeeze(sum_over_time)
        #print(np.shape(sum_over_time))
        #print(np.shape( matrix_exp@particle))
        new_particles.append(sum_over_time + matrix_exp@particle)
    return np.squeeze(np.array(new_particles))


def likelihood_function(particles, observation, sigma):
    likelihoods = []
    for particle in particles:
        l2 =np.sum((particle-observation)**2)
        likelihood = np.exp(-l2/sigma**2/2)
        likelihoods.append(likelihood)
    return np.squeeze(np.array(likelihoods))

def bootstrap_particle_filtering(observation, particles, weights, transition_function, likelihood_function, matrix_exp, dt, sigma):
    num_particles = len(particles)

    # Transition step: move each particle according to the transition model
    particles = transition_function(particles, dt, matrix_exp=matrix_exp)

    # Update weights based on observation likelihood
    weights *= likelihood_function(particles, observation, sigma)
    weights /= np.sum(weights)  # Normalization

    # Resampling step: resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]  # The resampled particles

    # Reset weights to 1/N for the resampled particles
    weights = np.full(num_particles, 1.0 / num_particles)

    return particles, weights
