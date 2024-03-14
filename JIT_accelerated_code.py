import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from numba import jit, njit
import matplotlib.pyplot as plt
from scipy.special import gammaln #log gamma function
from numba import jit,objmode,float64, int32






@jit(nopython=True)
def factorial(n):
    if n == 0:
        return 1
    else:
        f = 1
        for i in range(1, n + 1):
            f *= i
        return f


@jit(nopython=True)
def expm_specialized(A, t):
    theta = A[1, 1]
    exp_theta_t = np.exp(theta * t)
    
    # 创建第一个矩阵
    mat1 = np.zeros((2, 2))
    mat1[0, 1] = 1 / theta
    mat1[1, 1] = 1
    
    # 创建第二个矩阵
    mat2 = np.zeros((2, 2))
    mat2[0, 0] = 1
    mat2[0, 1] = -1 / theta
    
    # 计算结果
    result = exp_theta_t * mat1 + mat2
    return result



@jit(nopython=True)
def expm_pade(A):
    n = 6
    I = np.eye(A.shape[0], dtype=np.float64)
    A2 = np.dot(A, A)
    U = np.zeros_like(A)
    V = np.zeros_like(A)
    
    for k in range(n+1):
        C = factorial(2*n - k) / (factorial(k) * factorial(2*n - 2*k))
        Ak = np.linalg.matrix_power(A.astype(np.float64), k)
        if k % 2 == 0:
            V += C * Ak
        else:
            U += C * Ak

    expA = np.linalg.solve(I - U, I + V)
    return expA

@jit(nopython=True)
def transition_function_ultimate_NVM_pf_accelerated_code(ng_paths, ng_jumps, jump_times, g_jumps, x_dim, A, h):
    sum = np.zeros((x_dim, 1), dtype=np.float64)
    cov = np.zeros((x_dim, x_dim), dtype=np.float64)
    A = A.astype(np.float64)  # 确保A是float64类型来兼容expm_pade
    h = h.astype(np.float64)  # 确保h是float64类型

    for ng_jump, jump_time, g_jump in zip(ng_jumps, jump_times, g_jumps):
        special_vector = expm_specialized(A,-jump_time) @ h
        sum += g_jump * special_vector
        cov += g_jump * (special_vector @ special_vector.T)

    return sum, cov





def autocorrelation(samples):
    n = len(samples)
    mean = np.mean(samples)
    var = np.var(samples, ddof=0)
    acf = np.correlate(samples - mean, samples - mean, mode='full')[n-1:] / (var*n)
    return acf

def plot_autocorrelation(samples,parameter_name = "Theta", max_lag=20):
    sample_length = np.size(samples)
    if sample_length < max_lag:
        #print("Inadequant Sample Length")
        max_lag = sample_length-1
    acf = autocorrelation(samples)[:max_lag+1]
    time_lags = np.arange(max_lag+1)
    plt.figure(figsize=(10, 6))
    plt.stem(time_lags, acf, linefmt='-', markerfmt='o', basefmt=" ")
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation Function for {parameter_name}')
    plt.grid(True)
    plt.show()

    # Estimate the autocorrelation time
    act = 1 + 2 * np.sum(acf[1:])
    print(f"Estimated Autocorrelation Time: {act}")

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

@jit (nopython = True)
def inverted_gamma_to_mean_variance(alphas, betas, weights):
    # Calculate the mean of each particle distribution
    particle_mean_vector = betas / (alphas - 1)
    
    # Calculate the weighted mean
    mean = np.dot(weights, particle_mean_vector)
    
    # Calculate the squared mean and the expectation of X^2 for each particle
    particle_mean_squared = particle_mean_vector ** 2
    particle_variance = particle_mean_squared/(alphas-2)
    particle_x2_expectation = particle_variance + particle_mean_squared
    
    # Calculate the weighted sum of E[X^2] for particles
    weighted_sum_x2 = np.dot(weights, particle_x2_expectation)
    
    # Variance of the weighted sum
    variance = weighted_sum_x2 - mean**2
    
    return mean, variance

@njit
def faster_integrate(evaluation_points, x_series, t_series):
    # 初始化结果数组
    results = np.zeros(len(evaluation_points))
    for i, point in enumerate(evaluation_points):
        sum_ = 0.0
        for x, t in zip(x_series, t_series):
            if t < point:
                sum_ += x
        results[i] = sum_
    return results


class Levy_Point_Process:
    #This is the parent class to define a public method for the Gamma and tempered stable processes to give the output series
    def integrate(self,evaluation_points,x_series,t_series,integrator = faster_integrate): #Scalar integrate function
        #print("x",np.shape(x_series),x_series)
        #print("t",np.shape(t_series),t_series)
        results = integrator(evaluation_points,x_series,t_series)
        return results
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

@jit(nopython=True)
def generate_gamma_jumps(T,beta,C):
    repeatitions = math.ceil(T)
    x_list = []
    jump_times = []
    for n in range(repeatitions):
        poisson_epochs = []
        current_time = 0
        while True:
            random_interval = np.random.exponential(1)
            current_time += random_interval
            if current_time > 10:
                break
            poisson_epochs.append(current_time)
        
        new_x_list = []
        for i in poisson_epochs:
            x = 1 / (beta * (np.exp(i /C) - 1))
            p = (1 + beta * x) * np.exp(-beta * x)
            if np.random.rand() <= p:
                x_list.append(x)
                new_x_list.append(x)
        
        for x in new_x_list:
            jump_times.append(np.random.uniform(0, 1) + n)
        
    x_list = [jump for jump, time in zip(x_list, jump_times) if time < T]
    jump_times = [time for time in jump_times if time <T]
    
    return np.array(x_list), np.array(jump_times)

#The most important generator in this project. This has the built in Gamma generator different from previous one, It is correctly implemented and is the only one that passes the validity check
#Generator for Normal Gamma process. On top of Gamma process, reuqire additional muw and sigmaw parameters. Note that a built in Gamma generator is in this class. Raw data has additional return of the jump sizes and times of the process.
class normal_gamma_process(Levy_Point_Process):
    def __init__(self,beta,C,T,muw,sigmaw):
        self.beta = beta
        self.C = C
        self.T = T
        self.muw = muw
        self.sigmaw = sigmaw
    



    def generate_gamma_samples(self,evaluation_points,raw_data = False,gamma_generator = generate_gamma_jumps): #This fucntion returns the gamma process samples by putting in the resolution N (number of points to be evaluated in T)
        x_list,jump_times = gamma_generator(self.T,self.beta,self.C)
        if raw_data:
            return self.integrate(np.array(evaluation_points),np.array(x_list),np.array(jump_times)),x_list,jump_times
        
        else:
            return self.integrate(np.array(evaluation_points),np.array(x_list),np.array(jump_times))

    def generate_samples(self,evaluation_points,raw_data = False,all_data = False):
        if raw_data:
            gamma_paths,subordinator_jumps,jump_times = self.generate_gamma_samples(evaluation_points,True)
            gamma_paths = np.array(gamma_paths) #These are the gamma paths at the evluation points
            #Only update paths after jump times
            NVM_jumps = np.ones(len(subordinator_jumps))*self.muw*subordinator_jumps+self.sigmaw*np.sqrt(subordinator_jumps)*np.random.randn(len(subordinator_jumps))
            
            return self.integrate(np.array(evaluation_points),np.array(NVM_jumps,jump_times)), NVM_jumps,jump_times
        
        elif all_data:
            gamma_paths,subordinator_jumps,jump_times = self.generate_gamma_samples(evaluation_points,True)
            gamma_paths = np.array(gamma_paths) #These are the gamma paths at the evluation points
            #Only update paths after jump times
            NVM_jumps = np.ones(len(subordinator_jumps))*self.muw*subordinator_jumps+self.sigmaw*np.sqrt(subordinator_jumps)*np.random.randn(len(subordinator_jumps))
            
            return self.integrate(np.array(evaluation_points),np.array(NVM_jumps),np.array(jump_times)), NVM_jumps,jump_times, subordinator_jumps
            
        else:
            gamma_paths,subordinator_jumps,jump_times = self.generate_gamma_samples(evaluation_points,True)
            gamma_paths = np.array(gamma_paths) #These are the gamma paths at the evluation points
            #Only update paths after jump times
            NVM_jumps = np.ones(len(subordinator_jumps))*self.muw*subordinator_jumps+self.sigmaw*np.sqrt(subordinator_jumps)*np.random.randn(len(subordinator_jumps))
            
            return self.integrate(np.array(evaluation_points),np.array(NVM_jumps),np.array(jump_times))

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
                system_jump = NVM_jump * expm_specialized(self.A , evaluation_point-jump_time) @ self.h
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
@jit(nopython=True)
def Kalman_correct(X, P, Y, g, R): #The log_marginal returned would be used as the particle weight in marginalised particle filtering scheme.
    g = g.astype(np.float64)  # 假设我们统一使用 float64 类型
    X = X.astype(np.float64)
    P = P.astype(np.float64)
    R = R.astype(np.float64)

    Ino = Y - g @X  # Innovation term, just the predicton error


    S = g @ P @ g.T + R  # Innovation covariance

    K = P @ g.T /S 

    n = np.shape(P)[0]
    I = np.identity(n)
    log_cov_det = np.log(S)  # Use S for log marginal likelihood
    cov_inv = 1/S
    
    return X + K @ Ino, (I - K @ g) @ P, log_cov_det, np.dot((Ino).T, np.dot(cov_inv, (Ino)))
    


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

        sum,cov = transition_function_ultimate_NVM_pf_accelerated_code(ng_paths,ng_jumps,jump_times,g_jumps,x_dim,A,h)
        particles_sum_and_var.append([sum,cov])
    
    return particles_sum_and_var




    #The case with all NVM parameters estimated
@jit(nopython=True)
def compute_augmented_matrices(matrix_exp, particle_sum, noise_cov):
    n = matrix_exp.shape[0]
    combined_matrix = np.zeros((n + 1, n + 1))
    combined_matrix[:n, :n] = matrix_exp
    combined_matrix[:n, n] = particle_sum.T
    combined_matrix[n, n] = 1
    
    augmented_cov_matrix = np.zeros((n + 1, n + 1))
    augmented_cov_matrix[:n, :n] = noise_cov
    return combined_matrix, augmented_cov_matrix


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

        combined_matrix,augmented_cov_matrix = compute_augmented_matrices(matrix_exp, particle_sum, noise_cov)



        #The marginalise dKalman filter transition
        inferred_X, inferred_cov = Kalman_transit(previous_X, previous_X_uncertainty, combined_matrix, augmented_cov_matrix) #This could be the main position of problem, since the noise mean passed is a row vector here

        inferred_X, inferred_cov, log_det_F,Ei = Kalman_correct(inferred_X, inferred_cov, observation, g, R)

        
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
        

    #Resampling
    weights = log_probs_to_normalised_probs(log_marginals)

    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)


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
    


