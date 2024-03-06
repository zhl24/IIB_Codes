import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from numba import jit
import matplotlib.pyplot as plt
from scipy.special import gammaln #log gamma function



def log_probs_to_normalised_probs(log_likelihoods):
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = np.squeeze(log_likelihoods)
    log_likelihoods = np.where(np.isnan(log_likelihoods), -np.inf, log_likelihoods)
    normalization_factor = logsumexp(log_likelihoods)

    if not np.isfinite(normalization_factor):  # 检查是否为 -np.inf 或 np.inf
        # 如果normalization_factor不是有限数值，则所有概率都设为均等
        probabilities = np.full(log_likelihoods.shape, 1.0 / len(log_likelihoods))
    else:
        normalized_log_likelihoods = log_likelihoods - normalization_factor
        probabilities = np.exp(normalized_log_likelihoods)
        probabilities = probabilities/np.sum(probabilities)  # 确保概率之和为1
    
    return np.squeeze(probabilities)


def weighted_sum(particles, weights):
    

    # Initialize the sum as a zero array of the same shape as the first particle
    weighted_sum = np.zeros_like(particles[0])

    # Iterate over each particle and its weight
    for particle, weight in zip(particles, weights):
        weighted_sum += np.array(particle) * weight

    return weighted_sum


def inverted_gamma_to_mean_variance(alpha, beta):
    """
    Convert Inverted Gamma distribution parameters to mean and variance.
    
    Parameters:
    - alpha: shape parameter of the Inverted Gamma distribution (> 0).
    - beta: scale parameter of the Inverted Gamma distribution (> 0).
    
    Returns:
    - A tuple containing the mean and variance of the Inverted Gamma distribution.
      Returns (None, None) if the mean or variance does not exist.
    """
    mean = max(0,beta / (alpha - 1))
    
    
    variance = max(0,beta**2 / ((alpha - 1)**2 * (alpha - 2)))
    
    return mean, variance



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
    
    def generate_gamma_samples(self,evaluation_points,raw_data = False): #This fucntion returns the gamma process samples by putting in the resolution N (number of points to be evaluated in T)
          #The first step generates Poisson epochs in the time interval T
        repeatitions = math.ceil(self.T) #Since only the generation in unit time interval is correct, we simply repeat the generation several times to have the correct jump sizes and times
        x_list = [] #List of jump sizes
        jump_times = [] #List of jump times
        for n in range(repeatitions):
            poisson_epochs = []
            current_time = 0 #This stores the current arrival time
            while True:
                #random_interval = np.random.exponential(1/self.T) #Drawing from a rate T  Poisson process
                random_interval = np.random.exponential(1)
                current_time += random_interval
                if current_time > 10: #Only in a unit interval so 10 is more than enough
                #if len(poisson_epochs) >= self.T*10:
                    break
                poisson_epochs.append(current_time)#Storing the current arrival time gives the newest epoch
            
            #Then iterate over the Poisson epochs to generate the samples
            new_x_list = []
            for i in poisson_epochs:
                x = 1/(self.beta*(np.exp(i/self.C)-1))
                p = (1+self.beta*x)*np.exp(-self.beta*x)
                if np.random.binomial(n=1, p=p, size=1): #set n = 1 to have a Bernoulli generator, and p is the probability of getting a 1.
                    x_list.append(x)
                    new_x_list.append(x)

            #Then generate the jump times
            for x in new_x_list: #Or N_Ga?
                jump_times.append(np.random.uniform()+n)
         
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

def Kalman_correct(X, P, Y, g, R, mv=0, return_log_marginal=False,sigmaw_estimation = False): #The log_marginal returned would be used as the particle weight in marginalised particle filtering scheme.
    Ino = Y - g @ X - mv  # Innovation term, just the predicton error
    
    S = g @ P @ g.T + R  # Innovation covariance
    try:
        K = P @ g.T @ np.linalg.inv(S)  # Kalman gain
    except: #1D observation case
        K = P @ g.T /S 

    n = np.shape(P)[0]
    I = np.identity(n)
    if sigmaw_estimation:
        try:
            sign, logdet = np.linalg.slogdet(S)

            # 如果你只需要 log(det(S)) 的值
            if sign != 0:  # 行列式非零
                log_cov_det = logdet*sign  # 这是行列式的对数
                cov_inv = np.linalg.inv(S)
                k = Y.shape[0]
            else:
                # 处理行列式为零的情况
                log_cov_det = np.inf  # 或者其他适当的处理方式
                cov_inv = 0
                k = Y.shape[0]
            
            return X + K @ Ino, (I - K @ g) @ P, log_cov_det, np.dot((Ino).T, np.dot(cov_inv, (Ino)))
        except: #1D case
            if S<0:
                print("Negative Covariance")
            log_cov_det = np.log(np.abs(S))  # Use S for log marginal likelihood
            cov_inv = 1/S
            k = 1
            
            return X + K @ Ino, (I - K @ g) @ P, log_cov_det, np.dot((Ino).T, np.dot(cov_inv, (Ino)))
    
    elif return_log_marginal:
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
    

def transition_function_ultimate_NVM_pf(particles,dt,matrix_exp,SDE_model): #dt is the length of forwards simulation. t is the evaluation point

    new_particles = []
    #We assume first that we know the exact generator for the process. Parameters extracted from the incremental model passed
    particles_gaussian_parameters = []
    particles_sum_and_var = []
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
    
    for particle in particles:
        
        ng_paths,ng_jumps,jump_times,g_jumps = normal_gamma_generator.generate_samples(evaluation_points,all_data = True)

        ng_paths = np.array(ng_paths)
        ng_jumps = np.array(ng_jumps)
        jump_times = np.array(jump_times)
        g_jumps = np.array(g_jumps)

        #The jumpss and time already before dt
        #print(g_jumps)

        #Then we solve for the summation
        
        sum = 0
        cov = 0
        if len(jump_times)>1:
            for ng_jump,jump_time,g_jump in zip(ng_jumps,jump_times,g_jumps):
                special_vector = expm(-A * jump_time) @ h
                sum +=  g_jump * special_vector
                cov+=  g_jump * special_vector @ special_vector.T
            if muw == 0: #For dimension consistency, otherwise the mean is always a scalar
                mean = np.zeros((x_dim,1))

        elif len(jump_times) == 1:
            special_vector = expm(-A * jump_times) @ h
            sum = g_jumps * special_vector
            cov = g_jumps * special_vector @ special_vector.T
        else:
    # Initialize sum_over_time as a zero array of the same shape as a particle
            
            sum = np.zeros((x_dim,1)) #No jump so not applicable
            cov = np.zeros((x_dim,x_dim))
        #print(np.shape(sum_over_time))
        #print(np.shape( matrix_exp@particle))
        particles_sum_and_var.append([sum,cov])
    
    return particles_sum_and_var




    #The case with all NVM parameters estimated

def ultimate_NVM_pf(observation, previous_Xs, previous_X_uncertaintys, particles, transition_function, matrix_exp, dt,incremental_SDE,g,R,alphaws,betaws,accumulated_Es,accumulated_Fs,N, return_log_marginals = False): #N is the time index

    try:
        M = len(observation)
    except:#Scalar case
        M=1
    num_particles = len(particles)
    
    Xs_inferred = []
    uncertaintys_inferred = []
    # Transition step: move each particle according to the transition model. The new particles returned are Gaussian parameters, each one containing a pair of Gaussian mean and variance
    particles_sum_and_vars = transition_function(particles, dt, matrix_exp, incremental_SDE)  # The 2 true booleans turn on mpf and exytended state together, thus the particles returned contain the summation term and the covariane matrix for the conditional Gaussian distribution.
    
    log_marginals = []
    accumulated_log_marginals = []
    for i,particles_sum_and_var in enumerate(particles_sum_and_vars): #Iterate over each particle to run a marginal Kalman filter for each one of them
        
        previous_X = np.array(previous_Xs[i])
        #print("X",np.shape(previous_X))
        previous_X_uncertainty = previous_X_uncertaintys[i]
        #print(particles_gaussian_parameterss)
        particle_sum,noise_cov = particles_sum_and_var #Note that the noise covariance matrix is for the marginalised case
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
        
###### Additional processing of the paticles to fit them in the HMM
        #产生transition matrix for the augmented state
        n = matrix_exp.shape[0]
        # 创建一个 (n+1) x (n+1) 的矩阵
        combined_matrix = np.zeros((n + 1, n + 1))
        # 将 matrix_exp 和 particle_sum 放入新矩阵
        combined_matrix[:n, :n] = matrix_exp
        combined_matrix[:n, n] = particle_sum.T
        combined_matrix[n, n] = 1
    
        # 获取 noise_cov 的维度
        n = noise_cov.shape[0]
        # 创建一个新的 (n+1) x (n+1) 矩阵
        augmented_cov_matrix = np.zeros((n + 1, n + 1))
        # 将 noise_cov 放入新矩阵的左上角,此处直接使用BSB^T matrix product，与文献中的等效
        augmented_cov_matrix[:n, :n] = noise_cov #this is the marginalised noise covariance matrix



        #The marginalise dKalman filter transition
        inferred_X, inferred_cov = Kalman_transit(previous_X, previous_X_uncertainty, combined_matrix, augmented_cov_matrix) #This could be the main position of problem, since the noise mean passed is a row vector here
        
        
        #print(inferred_X)
        #inferred_X = inferred_X.reshape(nx0, 1)
        
        #print("inferred X",np.shape(inferred_X))
        #print("inferred cov",np.shape(inferred_cov))
        #print("g",np.shape(g))
        #print("R",np.shape(R))
        inferred_X, inferred_cov, log_det_F,Ei = Kalman_correct(inferred_X, inferred_cov, observation, g, R,return_log_marginal = False, sigmaw_estimation=True)
        accumulated_Fs[i] = accumulated_Fs[i] - 0.5 * log_det_F   #Note that this term is already negative
        accumulated_Es[i] = accumulated_Es[i]  + Ei
        #QAccumulated terms for each particle
        alphaws[i] = alphaws[i] + 1/2
        betaws[i] = betaws[i] + Ei/2
        alphaw = alphaws[i]
        betaw = betaws[i]
        accumulated_log_marginal =-M*N/2*np.log(2*np.pi) + accumulated_Fs[i] + alphaw * np.log(betaw) - (alphaw+N/2)*np.log(betaw + accumulated_Es[i]/2) + gammaln(N/2 + alphaw) - gammaln(alphaw)
        accumulated_log_marginals.append(accumulated_log_marginal)
        log_marginal = -M/2*np.log(2*np.pi) - 0.5 * log_det_F - (alphaw+N/2)*np.log(betaw+accumulated_Es[i]/2)+(alphaw+(N-1)/2)*np.log(betaw+(accumulated_Es[i]-Ei)/2)+gammaln(N/2 + alphaw) - gammaln((N-1)/2+alphaw) # The single observation log marginal to compute weight
        #print(np.shape(log_marginal))
        log_marginals.append( log_marginal.item()) #This is the log weight for each particle, normalize them in the log domain before tranforming them in to the usual probability domian for numerical stability
        Xs_inferred.append(inferred_X)
        uncertaintys_inferred.append(inferred_cov)
    #print(log_marginals)
    #print(np.shape(log_marginals))
        

    #Resampling
    weights = log_probs_to_normalised_probs(log_marginals)
    #print(np.shape(weights))
    
    # Resampling step: resample particles based on their weights
    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)
    #print("weights type:", type(weights), "shape:", np.shape(weights))
    #print("uncertaintys_inferred type:", type(uncertaintys_inferred), "length:", len(uncertaintys_inferred), "individual element type:", type(uncertaintys_inferred[0]) if uncertaintys_inferred else "N/A")
    #print("Xs_inferred type:", type(Xs_inferred), "length:", len(Xs_inferred), "individual element type:", type(Xs_inferred[0]) if Xs_inferred else "N/A")
    #print("log_marginals type:", type(log_marginals), "shape:", np.shape(log_marginals))
    #print("alphaws type:", type(alphaws), "shape:", np.shape(alphaws))
    #print("betaws type:", type(betaws), "shape:", np.shape(betaws))
    #print("accumulated_log_marginals type:", type(accumulated_log_marginals), "shape:", np.shape(accumulated_log_marginals))
    #print("accumulated_Es type:", type(accumulated_Es), "shape:", np.shape(accumulated_Es))
    #print("accumulated_Fs type:", type(accumulated_Fs), "shape:", np.shape(accumulated_Fs))

    weights_resampled = 1/num_particles * np.ones(num_particles)

    # 由于 uncertaintys_inferred 和 Xs_inferred 是列表，其中包含 NumPy 数组，我们需要保持这一结构不变
    uncertaintys_inferred_resampled = [uncertaintys_inferred[i] for i in indices]
    Xs_inferred_resampled = [Xs_inferred[i] for i in indices]

    # log_marginals 和 accumulated_log_marginals 是列表，但根据您的描述它们似乎应该是一维数组。这里我们假设它们是简单的数值列表，因此也使用列表推导式进行重采样
    log_marginals_resampled = [log_marginals[i] for i in indices]
    accumulated_log_marginals_resampled = [accumulated_log_marginals[i] for i in indices]

    # alphaws, betaws, accumulated_Es 和 accumulated_Fs 都是 NumPy 数组，可以直接使用索引进行重采样
    alphaws_resampled = alphaws[indices]
    betaws_resampled = betaws[indices]
    accumulated_Es_resampled = accumulated_Es[indices]
    accumulated_Fs_resampled = accumulated_Fs[indices]

    # 用重采样后的变量替换原有变量
    weights = weights_resampled
    uncertaintys_inferred = uncertaintys_inferred_resampled
    Xs_inferred = Xs_inferred_resampled
    log_marginals = log_marginals_resampled
    alphaws = alphaws_resampled
    betaws = betaws_resampled
    accumulated_log_marginals = accumulated_log_marginals_resampled
    accumulated_Es = accumulated_Es_resampled
    accumulated_Fs = accumulated_Fs_resampled
    #Re run the particle filter to resample the inference results! Otherwise meaningless, since the resampled Gaussain parameters would be forgotten directly in the next time step
    # Reset weights to 1/N for the resampled particles. The particles resampled should carry the same weights, and the inference result could be found directly from the mean.
    #weights = np.full(num_particles, 1.0 / num_particles)
    
    if return_log_marginals:
        return np.array(Xs_inferred),np.array(uncertaintys_inferred), particles_sum_and_vars, weights, alphaws, betaws, accumulated_Es, accumulated_Fs, accumulated_log_marginals    #Update the particle states
    else:
        return np.array(Xs_inferred),np.array(uncertaintys_inferred), particles_sum_and_vars, weights, alphaws, betaws, accumulated_Es, accumulated_Fs    #Update the particle states
    


