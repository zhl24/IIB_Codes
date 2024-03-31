import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
import math
from scipy.special import logsumexp
from numba import jit, njit
import matplotlib.pyplot as plt
from scipy.special import gammaln #log gamma function
from numba import jit,objmode,float64, int32




def parallel_particle_transition_function(beta,C,T,resolution,num_particles,A,h,c=10):
    h = h.astype(np.float64)  # 将 h 转换为 float64 类型

    gamma_jump_matrix,jump_time_matrix = parallel_particle_Gamma_generator(beta,C,T,resolution,num_particles,c = c)

    expA_tau = expm_specialized_parallel(A,-jump_time_matrix)
    expA_tau_h = np.matmul(expA_tau,h)
    gamma_jump_matrix = gamma_jump_matrix[...,np.newaxis,np.newaxis]
    raw_mean_matrix = gamma_jump_matrix * expA_tau_h
    pre_cov_matrix = np.sqrt(gamma_jump_matrix) * expA_tau_h
    raw_cov_matrix = pre_cov_matrix * pre_cov_matrix.swapaxes(-1, -2)
    mean_matrix = raw_mean_matrix.reshape(num_particles,resolution,c,2,1).sum(axis=2) #Grouped into groups of c and sum each group
    cov_matrix = raw_cov_matrix.reshape(num_particles,resolution,c,2,2).sum(axis=2)
    return mean_matrix,cov_matrix
    #system_jump_matrix = samples_matrix * expA_tau_h



@jit(nopython = True)
def parallel_particle_Gamma_generator(beta,C,T,resolution,num_particles,c=10): #First 2 are tyhe Gamma process parameters. T and resolution are the time axis parameters, and c is the number of samples in the Gamma process simulation
    dt = T/resolution
    samples_matrix = np.random.exponential(1/dt, (num_particles,c*resolution))  #The jump size matrix
    for n in range(num_particles):  # 对于每一组实验
        for t in range(resolution):  # 对于每一组中的时间段
            start_index = t*c
            end_index = start_index + c
            # 对当前组内的 C 个样本进行累积求和
            samples_matrix[n,start_index:end_index] = np.cumsum(samples_matrix[n,start_index:end_index])
    for i in range(num_particles):
        for j in range(c*resolution):
            poisson_epoch = samples_matrix[i,j] 
            x = 1 / (beta * (np.exp(poisson_epoch /C) - 1))
            p = (1 + beta * x) * np.exp(-beta * x)
            if np.random.rand() <= p: #Accepted case
                samples_matrix[i,j] = x
            else: #Rejection case
                samples_matrix[i,j] = 0 
    jump_time_matrix = np.random.uniform(0,dt,(num_particles,c*resolution))

    
    return samples_matrix,jump_time_matrix



def expm_specialized_parallel(A, t_matrix):
    theta = A[1, 1]
    exp_theta_t = np.exp(theta * t_matrix)
    
    # 创建第一个矩阵并复制到适当的形状
    mat1 = np.zeros((2, 2))
    mat1[0, 1] = 1 / theta
    mat1[1, 1] = 1
    mat1 = np.broadcast_to(mat1, (t_matrix.shape[0], t_matrix.shape[1], 2, 2))

    # 创建第二个矩阵并复制到适当的形状
    mat2 = np.zeros((2, 2))
    mat2[0, 0] = 1
    mat2[0, 1] = -1 / theta
    mat2 = np.broadcast_to(mat2, (t_matrix.shape[0], t_matrix.shape[1], 2, 2))

    # 使用广播机制进行缩放
    exp_theta_t_reshaped = exp_theta_t[..., np.newaxis, np.newaxis]
    result = exp_theta_t_reshaped * mat1 + mat2 #Use the broad-casted matrices for element-wise operations
    return result








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
                if jump_time > evaluation_point:
                    break
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
        



    
@jit(nopython = True)
def Kalman_transit(X, P, f, Q, mw=0, return_marginal=False):
    # Perform matrix multiplication
    X_new = f @ X
    

    X_new += mw
    # Handle control input B and u

    P_new = f @ P @ f.T + Q

    return X_new, P_new



    
#We again need the current states as the first two inputs, but now we need an obervation. g is the emission matrix, mv and R are 
#the observation mean and noises which determine most of the Kalman filtering difficulties
@jit(nopython=True) #This function is accelerated by using thge knwoledge of only 1 state being observed.
def Kalman_correct(X, P, Y, g, R,I): #The log_marginal returned would be used as the particle weight in marginalised particle filtering scheme.
    #g = g.astype(np.float64)  # 假设我们统一使用 float64 类型
    #X = X.astype(np.float64)
    #P = P.astype(np.float64)
    #R = R.astype(np.float64)

    Ino = Y - g @X  # Innovation term, just the predicton error


    S = g @ P @ g.T + R  # Innovation covariance

    K = P @ g.T /S 

    n = np.shape(P)[0]
    I = np.identity(n)
    
    
    return X + K @ Ino, (I - K @ g) @ P, np.log(S), Ino/S*Ino
    


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
@jit (nopython=True)
def compute_augmented_matrices(matrix_exp, particle_sum, noise_cov):
    #print(np.shape(matrix_exp))
    n = np.shape(matrix_exp)[0]
    combined_matrix = np.zeros((n + 1, n + 1))
    combined_matrix[:n, :n] = matrix_exp
    combined_matrix[:n, n] = particle_sum.T
    combined_matrix[n, n] = 1
    
    augmented_cov_matrix = np.zeros((n + 1, n + 1))
    augmented_cov_matrix[:n, :n] = noise_cov
    return combined_matrix, augmented_cov_matrix


@jit(nopython = True)
def accelerated_alphaw_betaw_update(alphaws,betaws,accumulated_Es,accumulated_Fs,new_Es,new_Fs):
    alphaws = alphaws+0.5
    betaws = betaws + new_Es * 0.5
    accumulated_Es = accumulated_Es + new_Es
    accumulated_Fs = accumulated_Fs + new_Fs
    return alphaws, betaws,accumulated_Es,accumulated_Fs
@jit(nopython = True)
def ultimate_NVM_pf_accelerated_code(M,N,gammaln_half_N_plus_alpha,gammaln_half_N_minus1_plus_alpha,gammaln_alphaws,accumulated_Es,accumulated_Fs,new_Es,new_Fs,alphaws,betaws):# The parallelised code after resampling
    accumulated_log_marginals = -M*N/2*np.log(2*np.pi) + accumulated_Fs + alphaws * np.log(betaws) - (alphaws+N/2)*np.log(betaws + accumulated_Es/2) + gammaln_half_N_plus_alpha - gammaln_alphaws
    log_marginals = -M/2*np.log(2*np.pi) +new_Fs - (alphaws+N/2)*np.log(betaws+accumulated_Es/2)+(alphaws+(N-1)/2)*np.log(betaws+(accumulated_Es-new_Es)/2) + gammaln_half_N_plus_alpha - gammaln_half_N_minus1_plus_alpha
    return accumulated_log_marginals,log_marginals


@njit
def resample_particles(num_particles, indices, log_marginals, accumulated_log_marginals, alphaws, betaws, accumulated_Es, accumulated_Fs):
    
    
    weights_resampled = 1 / num_particles * np.ones(num_particles)
    
    
    log_marginals_resampled = log_marginals[indices]
    accumulated_log_marginals_resampled = accumulated_log_marginals[indices]
    
    alphaws_resampled = alphaws[indices]
    betaws_resampled = betaws[indices]
    accumulated_Es_resampled = accumulated_Es[indices]
    accumulated_Fs_resampled = accumulated_Fs[indices]
    
    return weights_resampled, log_marginals_resampled, accumulated_log_marginals_resampled, alphaws_resampled, betaws_resampled, accumulated_Es_resampled, accumulated_Fs_resampled


def ultimate_NVM_pf(observation, previous_Xs, previous_X_uncertaintys, mean_proposal,cov_proposal, matrix_exp,g,R,alphaws,betaws,accumulated_Es,accumulated_Fs,N, return_log_marginals = False): #N is the time index
    
    try:
        M = len(observation)
    except:#Scalar case
        M=1
    num_particles = mean_proposal.shape[0]
    
    Xs_inferred = []
    uncertaintys_inferred = []
    # Transition step: move each particle according to the transition model. The new particles returned are Gaussian parameters, each one containing a pair of Gaussian mean and variance
    log_marginals = np.zeros(num_particles)
    accumulated_log_marginals = np.zeros(num_particles)
    I = np.identity(2)
    new_Es = np.zeros(num_particles)
    new_Fs = np.zeros(num_particles)
    for i in range(num_particles): #Iterate over each particle to run a marginal Kalman filter for each one of them
        #print(np.shape(particle_sum))
        particle_sum = mean_proposal[i]
        #print(np.shape(particle_sum))
        noise_cov = cov_proposal[i]
        previous_X = np.array(previous_Xs[i])
        #print("X",np.shape(previous_X))
        previous_X_uncertainty = previous_X_uncertaintys[i]
        combined_matrix,augmented_cov_matrix = compute_augmented_matrices(matrix_exp, particle_sum, noise_cov)
        #The marginalise dKalman filter transition
        inferred_X, inferred_cov = Kalman_transit(previous_X, previous_X_uncertainty, combined_matrix, augmented_cov_matrix) #This could be the main position of problem, since the noise mean passed is a row vector here
        inferred_X, inferred_cov, log_det_F,Ei = Kalman_correct(inferred_X, inferred_cov, observation, g, R,I)
        new_Es[i] = Ei
        new_Fs[i] = -0.5 * log_det_F
        #QAccumulated terms for each particle
        Xs_inferred.append(inferred_X)
        uncertaintys_inferred.append(inferred_cov)

    alphaws, betaws,accumulated_Es,accumulated_Fs = accelerated_alphaw_betaw_update(alphaws,betaws,accumulated_Es,accumulated_Fs,new_Es,new_Fs)
    gammaln_half_N_plus_alpha = gammaln(N/2+alphaws)
    gammaln_half_N_minus1_plus_alpha = gammaln((N-1)/2+alphaws)
    gammaln_alphaws = gammaln(alphaws)

    accumulated_log_marginals,log_marginals =  ultimate_NVM_pf_accelerated_code(M,N,gammaln_half_N_plus_alpha,gammaln_half_N_minus1_plus_alpha,gammaln_alphaws,accumulated_Es,accumulated_Fs,new_Es,new_Fs,alphaws,betaws)
    #Resampling
    weights = log_probs_to_normalised_probs(log_marginals)

    indices = np.random.choice(np.arange(num_particles), size=num_particles, p=weights)



    uncertaintys_inferred_resampled = [uncertaintys_inferred[i] for i in indices]
    Xs_inferred_resampled = [Xs_inferred[i] for i in indices]

    uncertaintys_inferred = uncertaintys_inferred_resampled
    Xs_inferred = Xs_inferred_resampled

    weights, log_marginals, accumulated_log_marginals, alphaws, betaws, accumulated_Es, accumulated_Fs =  resample_particles(num_particles, indices, log_marginals, accumulated_log_marginals, alphaws, betaws, accumulated_Es, accumulated_Fs)
    if return_log_marginals:
        return np.array(Xs_inferred),np.array(uncertaintys_inferred), mean_proposal,cov_proposal, weights, alphaws, betaws, accumulated_Es, accumulated_Fs, accumulated_log_marginals    #Update the particle states
    else:
        return np.array(Xs_inferred),np.array(uncertaintys_inferred), mean_proposal,cov_proposal, weights, alphaws, betaws, accumulated_Es, accumulated_Fs    #Update the particle states
    


