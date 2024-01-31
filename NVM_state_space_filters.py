import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from jump_process_generators import *
import matplotlib.pyplot as plt
from scipy.special import logsumexp
from basic_tools import *


def Kalman_Predict_1D(x_mean,x_variance,x_transition,x_noise_variance,x_noise_mean = 0): #Hidden state x predict step. 
#Note that the noise is generally not zero mean in our case.
#This function takes the mean and variance of the current hidden state and the transition architecture to predict the next state mean and variance
    return x_mean*x_transition+x_noise_mean,x_variance*x_transition**2 + x_noise_variance

def Kalman_Correct_1D(x_mean,x_variance,observation,emission,emission_noise_variance):
    I = observation - emission * x_mean
    return x_mean + (emission * x_variance)/(emission**2 * x_variance + emission_noise_variance)*I, x_variance * (1-emission**2 * x_variance/(emission**2*x_variance+emission_noise_variance))
#evaluation points just form the time axis
#We then simply need the subordinator jumps and jump times to compute the x_noise mean and variances
#emission is simply 1 in our model. emission_noise_variance is just the simga_n term in the model

def Noisy_Normal_Gamma_SDE_Filter_1D(noisy_sequence,observation_noise_var,SDE,subordinator_jumps,jump_times,evaluation_points,Kalman_Predict,Kalman_Correct,x_mean_prior=0,x_variance_prior=1):
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
#Remember that all the operations in Kalman filter are in column vector convention.
def Kalman_transit(X, P, f, Q, mw=0, B=0, u=0, return_marginal=False):
    # Perform matrix multiplication
    X_new = f @ X
    
    try:
        if mw == 0:
            pass
        else:
            X_new += mw
    except:
        X_new += mw

    # Handle control input B and u
    if B != 0:
        # Ensure B @ u is a column vector
        Bu = B @ u
        X_new += Bu

    P_new = f @ P @ f.T + Q

    return X_new, P_new



    
#We again need the current states as the first two inputs, but now we need an obervation. g is the emission matrix, mv and R are 
#the observation mean and noises which determine most of the Kalman filtering difficulties
def Kalman_correct(X, P, Y, g, R, mv=0, return_log_marginal=False): #The log_marginal returned would be used as the particle weight in marginalised particle filtering scheme.
    Ino = Y - g @ X - mv  # Innovation term
    
    S = g @ P @ g.T + R  # Innovation covariance
    K = P @ g.T @ np.linalg.inv(S)  # Kalman gain
    n = np.shape(P)[0]
    I = np.identity(n)
    
    if return_log_marginal:
        log_cov_det = np.log(np.linalg.det(S))  # Use S for log marginal likelihood
        cov_inv = np.linalg.inv(S)
        k = Y.shape[0]
        
        log_marginal_likelihood = -0.5 * (k * np.log(2 * np.pi) + log_cov_det + np.dot((Ino).T, np.dot(cov_inv, (Ino))))
        return X + K @ Ino, (I - K @ g) @ P, log_marginal_likelihood
    else:
        update = K @ Ino
        print(np.shape(update))
        print(update)
        return X + K @ Ino, (I - K @ g) @ P









# Bootstrap Particle Filtering in General State Space Model in one step. Particle filteirng is itself by nature a general dimensional method. Since pdf maps vector samples into scalar probabilitiei.e. weights
#The most important point about filtering in the state spce model is the knowledge of time, t and dt.
#Just need to additionally define the transition function to simulate the particles forwards, and likelihood function to compute the particle probabilities given the observations. Here we just assume standard normal noise

#We the try to filter from the noisy observations using particle filtering. Define the transition function and likelihood function to do that
#The generator for particle filter!
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




def compute_log_likelihoods(particles, observation, sigma):
    distances_squared = np.sum((particles - observation)**2, axis=1)
    log_likelihoods = -distances_squared / (2 * sigma**2)
    return log_likelihoods

def normalize_log_probabilities(log_probs):
    max_log_prob = np.max(log_probs)
    stabilized_log_probs = log_probs - max_log_prob
    sum_of_exps = np.sum(np.exp(stabilized_log_probs))
    log_normalizer = np.log(sum_of_exps)
    normalized_log_probs = stabilized_log_probs - log_normalizer
    return normalized_log_probs














#Below is the code for marginalized particle filter. Note that the current implementation is based on the full particle filtering algorithm, so only a few functions that return additional features are redefined here. Other functions are the same as ther particle filteirng function

def transition_function_mpf(particles,dt,matrix_exp): #dt is the length of forwards simulation. t is the evaluation point
    new_particles = []
    particles_gaussian_parameters = []
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
        ng_paths,ng_jumps,jump_times,g_jumps = normal_gamma_generator.generate_samples(evaluation_points,all_data = True)
        #The jumpss and time already before dt
        

        #Then we solve for the summation
        system_jumps = []
        mean = 0
        cov = 0
        if len(jump_times)>1:
            for ng_jump,jump_time,g_jump in zip(ng_jumps,jump_times,g_jumps):
                special_vector = expm(-A * jump_time) @ h
                mean += muw * g_jump * special_vector
                cov+= sigmaw**2 * g_jump * special_vector @ special_vector.T
                system_jump = ng_jump * special_vector
                system_jumps.append(system_jump)
            # Use the mask to select data from x_series and sum along the time axis (axis=0)
            sum_over_time = np.sum(system_jumps, axis=0)
            sum_over_time = np.squeeze(sum_over_time)
        elif len(jump_times) == 1:
            special_vector = expm(-A * jump_times) @ h
            mean = muw * g_jumps * special_vector
            cov = sigmaw**2 * g_jumps * special_vector @ special_vector.T
            system_jump = ng_jumps * special_vector
            system_jumps.append(system_jump)
            sum_over_time = np.sum(system_jumps, axis=0)
            sum_over_time = np.squeeze(sum_over_time)
        else:
    # Initialize sum_over_time as a zero array of the same shape as a particle
            sum_over_time = np.zeros(2)
            sum_over_time = np.squeeze(sum_over_time)
            mean = np.zeros(2) #No jump so not applicable
            cov = np.zeros((2,2))
        #print(np.shape(sum_over_time))
        #print(np.shape( matrix_exp@particle))
        new_particles.append(sum_over_time + matrix_exp@particle)
        particles_gaussian_parameters.append([mean,cov])
    return np.squeeze(np.array(new_particles)),particles_gaussian_parameters





def particle_filtering_mpf(observation, particles, weights, transition_function, likelihood_function, matrix_exp, dt, sigma):
    num_particles = len(particles)

    # Transition step: move each particle according to the transition model
    particles,particles_gaussian_parameters = transition_function(particles, dt, matrix_exp=matrix_exp)

    # Update weights based on observation likelihood
    weights *= likelihood_function(particles, observation, sigma)
    weights /= np.sum(weights)  # Normalization

    # Resampling step: resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]  # The resampled particles

    # Reset weights to 1/N for the resampled particles
    weights = np.full(num_particles, 1.0 / num_particles)

    return particles, weights,particles_gaussian_parameters



def compute_inferred_gaussian_parameters(particles_gaussian_parameters, weights):
    """
    Compute the inferred Gaussian parameters (mean and covariance) from the particle parameters.

    :param particles_gaussian_parameters: List of tuples [(m1, S1), (m2, S2), ...] where m is mean and S is covariance
    :param weights: Corresponding weights of the particles
    :return: Tuple (inferred_mean, inferred_covariance)
    """
    weights = weights/np.sum(weights)
    inferred_mean = 0
    inferred_covariance = 0
    for particle_gaussian_parameters,weight in zip(particles_gaussian_parameters,weights):
        m,S = particle_gaussian_parameters
        if np.sum(S) != 0 and m.size>0:
            
            inferred_mean += m * weight
            inferred_covariance += S * weight
        else:
            continue
    return inferred_mean, inferred_covariance








#only code from this line is useful.


#This is the most general transition function that propogate particle forwards after defining a SDE model
#Setting the mpf boolean to true wuld return the particles defined by the Gamma process instead of Normal Gamma process
def transition_function_general(particles,dt,matrix_exp,SDE_model,mpf = False): #dt is the length of forwards simulation. t is the evaluation point
    new_particles = []
    #We assume first that we know the exact generator for the process. Parameters extracted from the incremental model passed
    particles_gaussian_parameters = []
    A = SDE_model.A
    h = SDE_model.h
    x_dim = np.shape(h)[0]
    #Simulation over dt interval to obtain the jumps and jump times
    evaluation_points = [dt] #Note that this would be the time axis we work on.
    normal_gamma_generator = SDE_model.NVM  #Note that tyhe system has to be defined in dt time scale. i.e. we need the incremental model
    T = SDE_model.T #Note that this should be equal to dt
    C = normal_gamma_generator.C
    beta = normal_gamma_generator.beta
    muw = normal_gamma_generator.muw
    sigmaw = normal_gamma_generator.sigmaw
    if not mpf:
        for particle in particles:
            ng_paths,ng_jumps,jump_times,g_jumps = normal_gamma_generator.generate_samples(evaluation_points,all_data = True) 
            #Make sure that these are arrays before putting them in vector computation
            #The jumpss and time already before dt
            ng_paths = np.array(ng_paths)
            ng_jumps = np.array(ng_jumps)
            jump_times = np.array(jump_times)
            g_jumps = np.array(g_jumps)

            print(g_jumps)
            #Then we solve for the summation
            system_jumps = []
            mean = 0
            cov = 0
            if len(jump_times)>1:
                for ng_jump,jump_time,g_jump in zip(ng_jumps,jump_times,g_jumps):
                    special_vector = expm(-A * jump_time) @ h
                    mean += muw * g_jump * special_vector
                    cov+= sigmaw**2 * g_jump * special_vector @ special_vector.T
                    system_jump = ng_jump * special_vector
                    system_jumps.append(system_jump)
                # Use the mask to select data from x_series and sum along the time axis (axis=0)
                sum_over_time = np.sum(system_jumps, axis=0)
                sum_over_time = np.squeeze(sum_over_time)
            elif len(jump_times) == 1:
                special_vector = expm(-A * jump_times) @ h
                mean = muw * g_jumps * special_vector
                cov = sigmaw**2 * g_jumps * special_vector @ special_vector.T
                system_jump = ng_jumps * special_vector
                system_jumps.append(system_jump)
                sum_over_time = np.sum(system_jumps, axis=0)
                sum_over_time = np.squeeze(sum_over_time)
            else:
        # Initialize sum_over_time as a zero array of the same shape as a particle
                sum_over_time = np.zeros((x_dim,1))
                sum_over_time = np.squeeze(sum_over_time)
                mean = np.zeros((x_dim,1)) #No jump so not applicable
                cov = np.zeros((x_dim,x_dim))
            #print(np.shape(sum_over_time))
            #print(np.shape( matrix_exp@particle))
            new_particle = sum_over_time + matrix_exp @ particle
            new_particles.append(new_particle)
            particles_gaussian_parameters.append([mean,cov])
    
        return np.squeeze(np.array(new_particles))
    
    else: #The particles would be the Gaussian parameters now
        for particle in particles:
            ng_paths,ng_jumps,jump_times,g_jumps = normal_gamma_generator.generate_samples(evaluation_points,all_data = True)

            ng_paths = np.array(ng_paths)
            ng_jumps = np.array(ng_jumps)
            jump_times = np.array(jump_times)
            g_jumps = np.array(g_jumps)

            #The jumpss and time already before dt
            #print(g_jumps)

            #Then we solve for the summation
            
            mean = 0
            cov = 0
            if len(jump_times)>1:
                for ng_jump,jump_time,g_jump in zip(ng_jumps,jump_times,g_jumps):
                    special_vector = expm(-A * jump_time) @ h
                    mean += muw * g_jump * special_vector
                    cov+= sigmaw**2 * g_jump * special_vector @ special_vector.T
                if muw == 0: #For dimension consistency, otherwise the mean is always a scalar
                    mean = np.zeros((x_dim,1))

            elif len(jump_times) == 1:
                special_vector = expm(-A * jump_times) @ h
                mean = muw * g_jumps * special_vector
                cov = sigmaw**2 * g_jumps * special_vector @ special_vector.T
                if muw == 0: #For dimension consistency, otherwise the mean is always a scalar
                    mean = np.zeros((x_dim,1))
            else:
        # Initialize sum_over_time as a zero array of the same shape as a particle
                
                mean = np.zeros((x_dim,1)) #No jump so not applicable
                cov = np.zeros((x_dim,x_dim))
            #print(np.shape(sum_over_time))
            #print(np.shape( matrix_exp@particle))
            particles_gaussian_parameters.append([mean,cov])
        return particles_gaussian_parameters



def bootstrap_particle_filtering_general(observation, particles, weights, transition_function, likelihood_function, matrix_exp, dt, sigma,SDE_model):
    num_particles = len(particles)

    # Transition step: move each particle according to the transition model
    particles = transition_function(particles, dt, matrix_exp,SDE_model)

    # Update weights based on observation likelihood
    weights *= likelihood_function(particles, observation, sigma)
    weights /= np.sum(weights)  # Normalization

    # Resampling step: resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    particles = particles[indices]  # The resampled particles

    # Reset weights to 1/N for the resampled particles
    weights = np.full(num_particles, 1.0 / num_particles)

    return particles, weights



#A single step in marginal particle filtering
def particle_filtering_mpf_general(observation, previous_Xs, previous_X_uncertaintys, particles, transition_function, matrix_exp, dt,incremental_SDE,g,R): #g is the observation matrix, R is the noise covariance matrix.
    # Previous X would be the hidden states inferred from the last step, and particles would be the Gaussian parameters. g and R are the observation matrix and observation noise covariance matrix.
    nx0 = np.shape(previous_Xs[0])[0] #Same dimensions here
    num_particles = len(particles)
    
    Xs_inferred = []
    uncertaintys_inferred = []
    # Transition step: move each particle according to the transition model. The new particles returned are Gaussian parameters, each one containing a pair of Gaussian mean and variance
    particles_gaussian_parameterss = transition_function(particles, dt, matrix_exp, incremental_SDE, True)  # The true boolean to turn on the mpf model of the transition function. The particles are hence just the Gaussian parameters.
    
    log_marginals = []
   
    for i,particles_gaussian_parameters in enumerate(particles_gaussian_parameterss): #Iterate over each particle to run a marginal Kalman filter for each one of them
        
        previous_X = np.array(previous_Xs[i])
        #print("X",np.shape(previous_X))
        previous_X_uncertainty = previous_X_uncertaintys[i]
        #print(particles_gaussian_parameterss)
        noise_mean, noise_cov = particles_gaussian_parameters
        try:
            observation = np.array(observation).reshape(len(observation), 1) #Note that this is a must, since the observation array is by default in the row vector form.加了brackets[]方法会被视作是新的一行，直接用逗号间隔元素会被认作是row vector
        except: #Single float case, convert the float into a single element array
            observation = np.array([observation])
        #previous_X = np.array(previous_X).reshape(nx0, 1)
        # Kalman Prediction Step
        #print(np.shape(previous_X))
        #print(previous_X)
        #print(nx0)
        #print(noise_mean)
        #print(np.shape(noise_mean))
        noise_mean = np.array(noise_mean).reshape(len(noise_mean), 1) #In the muw = 0 case, this would be 0 throughout
        inferred_X, inferred_cov = Kalman_transit(previous_X, previous_X_uncertainty, matrix_exp, noise_cov, mw=noise_mean) #This could be the main position of problem, since the noise mean passed is a row vector here
        
        
        #print(inferred_X)
        #inferred_X = inferred_X.reshape(nx0, 1)
        
        # print(np.shape(inferred_X))
        # Kalman Correction Step
        
        #print(np.shape(inferred_X))
        inferred_X, inferred_cov, log_marginal = Kalman_correct(inferred_X, inferred_cov, observation, g, R,return_log_marginal = True)
        
        log_marginals.append(log_marginal) #This is the log weight for each particle, normalize them in the log domain before tranforming them in to the usual probability domian for numerical stability
        Xs_inferred.append(inferred_X)
        uncertaintys_inferred.append(inferred_cov)
    weights = log_probs_to_normalised_probs(log_marginals)
    
    
    # Resampling step: resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    

    # Ensure indices is an array of integers
    #indices = indices.astype(int)
    # Use indices to resample the propagated particles, Gaussian parameters for the conditional noise
    particles_gaussian_parameterss = [particles_gaussian_parameterss[i] for i in indices]
    #Re run the particle filter to resample the inference results! Otherwise meaningless, since the resampled Gaussain parameters would be forgotten directly in the next time step

    


# Now the indexing operation

    # Reset weights to 1/N for the resampled particles. The particles resampled should carry the same weights, and the inference result could be found directly from the mean.
    weights = np.full(num_particles, 1.0 / num_particles)
    for i,particles_gaussian_parameters in enumerate(particles_gaussian_parameterss): #Iterate over each particle to run a marginal Kalman filter for each one of them
        #print(particles_gaussian_parameterss)
        noise_mean, noise_cov = particles_gaussian_parameters
        
        noise_mean = np.array(noise_mean).reshape(len(noise_mean), 1) #In the muw = 0 case, this would be 0 throughout
        inferred_X, inferred_cov = Kalman_transit(previous_X, previous_X_uncertainty, matrix_exp, noise_cov, mw=noise_mean) #This could be the main position of problem, since the noise mean passed is a row vector here
        
        
        #print(inferred_X)
        #inferred_X = inferred_X.reshape(nx0, 1)
        
        # print(np.shape(inferred_X))
        # Kalman Correction Step
        
        #print(np.shape(inferred_X))
        inferred_X, inferred_cov, log_marginal = Kalman_correct(inferred_X, inferred_cov, observation, g, R,return_log_marginal = True)
        
        log_marginals.append(log_marginal) #This is the log weight for each particle, normalize them in the log domain before tranforming them in to the usual probability domian for numerical stability
        Xs_inferred.append(inferred_X)
        uncertaintys_inferred.append(inferred_cov)
    weights = log_probs_to_normalised_probs(log_marginals)


    return np.array(Xs_inferred),np.array(uncertaintys_inferred), particles_gaussian_parameterss, weights





