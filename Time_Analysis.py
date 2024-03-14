import cProfile
import pstats
import numpy as np
from scipy.linalg import expm #This is the automatic matrix expnent solver
from scipy.special import logsumexp
from tqdm import tqdm  # 导入tqdm
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from improved_generators import *



def main():
    
    #Again, we first generate the noisy observations.
    #We again have the noisy data first:
    #Again, we first generate the noisy observations.
    #We again have the noisy data first:

    kw = 1 #The prior parameter for muw

    kv = 0.001 #The observation noise scaling factor. Note that this scaling factor applies to the variance
    true_sigmaw2 = 0.2
    true_muw = 0.2
    true_sigmaw = np.sqrt(true_sigmaw2)
    sigma_n = true_sigmaw * np.sqrt(kv)

    num_particles = 1000

    #Prior inverted gamma parameters for sigmaw
    alphaws = 0.0000000001 * np.ones(num_particles)
    betaws = 0.0000000001 * np.ones(num_particles)
    accumulated_Es = np.zeros(num_particles)
    accumulated_Fs = np.zeros(num_particles)

    #Simulation Parameters
    theta = -2 #The main control parameter for the Lagevin system
    beta = 5
    C = 10
    T = 100


    N = 1000  # Resolution

    #Define the Langevin dynamics
    A = np.zeros((2, 2))
    A[0, 1] = 1
    A[1, 1] = theta
    h = np.array([[0], [1]])

    #Simulation
    evaluation_points = np.linspace(0, T, N) #Note that this would be the time axis we work on.
    normal_gamma_generator = normal_gamma_process(beta, C, T, true_muw, true_sigmaw)
    langevin = SDE(A,h,T,normal_gamma_generator)
    #Noisy observation generation
    SDE_samples,system_jumps,NVM_jumps,subordinator_jumps,jump_times = langevin.generate_samples(evaluation_points,all_data=True)
    d1,d2 = np.shape(SDE_samples)
    #Noisy_samples = SDE_samples + np.random.randn(d1,d2)*sigma_n #The noisy observations simulated. Already in the column vector form


    #Create the partial observation, observing only the integral state x here.
    Noisy_samples = SDE_samples[:,0] + np.random.randn(np.shape(SDE_samples)[0])*sigma_n #The noisy observations simulated. Already in the column vector form






    trajectory = []


    #Kalman filter initialisation
    X0 = Noisy_samples[0]
    nx0 = 2
    X0 = np.zeros((nx0+1,1))
    nx0_new = 3

    #The margianlised Kalman covariance
    C_prior = np.zeros((nx0_new,nx0_new))
    C_prior[-1,-1] = kw

    g = np.array([[1.],[0.],[0.]])
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




    sigmaw2_values = []
    sigmaw2_uncertainties = []
    alphas = []
    betas = []

    ### From here we pregenerate the proposals for particle filter
    mean_proposals,cov_proposals =  parallel_particle_transition_function(beta,C,T,N,num_particles,A,h,c=30)  #The dimensions are (num_particles,N,2,1) and (num_particles,N,2,2)


    for i in tqdm(range(len(evaluation_points)),desc = "Processing"): #i is the time index we want for N
        #sigmaw here needs to be updated in every step
        mean_proposal = mean_proposals[:,i,:,:]
        cov_proposal = cov_proposals[:,i,:,:]
        sigmaw2,sigmaw2_uncertainty = inverted_gamma_to_mean_variance(alphaws, betaws,weights) #Note that this is sigmaw^2 but not sigmaw
        sigmaw = np.sqrt(sigmaw2)

        sigmaw2_values.append(sigmaw2)
        sigmaw2_uncertainties.append(sigmaw2_uncertainty)

        t = evaluation_points[i]
        observation = Noisy_samples[i] #Note that the observation here is still a row vector, reshaping necessary
        previous_Xs, previous_X_uncertaintys,_,_,weights,alphaws, betaws, accumulated_Es, accumulated_Fs  = ultimate_NVM_pf(observation, previous_Xs,previous_X_uncertaintys,mean_proposal,cov_proposal ,  matrix_exp,g,R,alphaws,betaws,accumulated_Es,accumulated_Fs,i) # N is the time nidex
        inferred_cov = weighted_sum(previous_X_uncertaintys,weights) * sigmaw2 #Note that the original parameters are marginalised by sigmaw^2
        inferred_X = weighted_sum(previous_Xs,weights)
        
        
        inferred_Xs.append(inferred_X)
        inferred_covs.append(inferred_cov)










    trajectory = np.array(trajectory)
        

    #These are the returns
    ##########################################################################################################################################
    inferred_Xs_array = np.array(inferred_Xs)

    # Convert inferred_covs to a NumPy array for easier manipulation
    inferred_covs_array = np.array(inferred_covs)




    # Calculate standard deviation (sqrt of variance) for each state variable at each time step
    # Ensuring the shape of std_devs matches with inferred_Xs_array
    std_devs = np.sqrt([np.diag(cov) for cov in inferred_covs_array])

    # Convert std_devs to a numpy array for consistent array operations
    std_devs_array = np.array(std_devs)

    # Plotting with uncertainty range for the first state variable
    num_states = inferred_Xs_array.shape[1]


    last_state_muw = inferred_Xs_array[:, -1, 0]
    last_state_std_dev = std_devs_array[:, -1]


    # 这部分代码用于在绘图标题中动态显示参数值
    # 注意：这段代码假设前面的所有变量和过程已经按照原始代码正确运行

    # 获取最后的muw和sigmaw估计值及其不确定性
    final_muw_estimate = last_state_muw[-1]
    final_sigmaw2_estimate = sigmaw2_values[-1]
    final_sigmaw2_uncertainty = sigmaw2_uncertainties[-1]





    skip = 50 #skip the first few variance estimates due to instability


    # 更新绘图代码以包含动态参数值
    # 以下只展示修改后的绘图部分


    # 第一个状态变量的绘图
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_points, Noisy_samples, label="Noisy Observations")
    plt.plot(evaluation_points, SDE_samples[:, 0], label="Hidden States")
    plt.plot(evaluation_points, inferred_Xs_array[:, 0, 0], label="Marginalized Particle Filter")
    # 跳过错误带的前几个点
    plt.fill_between(evaluation_points[skip:], 
                    inferred_Xs_array[skip:, 0, 0] - std_devs_array[skip:, 0], 
                    inferred_Xs_array[skip:, 0, 0] + std_devs_array[skip:, 0], 
                    color='gray', alpha=0.3)
    plt.title(f"State Variable x, muw = {true_muw}, sigmaw = {true_sigmaw}, sigma_n = {sigma_n:.2f}, Number of Particles = {num_particles}")
    plt.xlabel('Time')
    plt.ylabel('State x')
    plt.legend()
    plt.show()

    # 第二个状态变量的绘图，同样跳过错误带的前几个点
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_points, SDE_samples[:, 1], label="Hidden States")
    plt.plot(evaluation_points, inferred_Xs_array[:, 1, 0], label="Marginalized Particle Filter")
    plt.fill_between(evaluation_points[skip:], 
                    inferred_Xs_array[skip:, 1, 0] - std_devs_array[skip:, 1], 
                    inferred_Xs_array[skip:, 1, 0] + std_devs_array[skip:, 1], 
                    color='gray', alpha=0.3)
    plt.title(f"State Variable dx/dt, muw = {true_muw}, sigmaw = {true_sigmaw}, sigma_n = {sigma_n:.2f}, Number of Particles = {num_particles}")
    plt.xlabel('Time')
    plt.ylabel('State dx/dt')
    plt.legend()
    plt.show()

    # muw的绘图
    plt.figure(figsize=(10, 6))
    plt.plot(evaluation_points, last_state_muw, label="Marginalized Particle Filter (muw)")
    plt.fill_between(evaluation_points[skip:], 
                    last_state_muw[skip:] - last_state_std_dev[skip:], 
                    last_state_muw[skip:] + last_state_std_dev[skip:], 
                    color='gray', alpha=0.3)
    plt.title(f"Last State Variable (True muw={true_muw}), Final muw = {final_muw_estimate:.2f}, sigma_n = {sigma_n:.2f}, Number of Particles = {num_particles}")
    plt.xlabel('Time')
    plt.ylabel('muw')
    plt.legend()
    plt.show()

    # 绘制sigmaw值和其不确定性范围
    plt.figure(figsize=(12, 6))
    plt.plot(evaluation_points[skip:], sigmaw2_values[skip:], label='Estimated sigmaw^2', color='blue')

    # 计算sigmaw的不确定性上下界
    sigmaw2_upper_bound = [val + np.sqrt(unc) for val, unc in zip(sigmaw2_values, sigmaw2_uncertainties)]
    sigmaw2_lower_bound = [val - np.sqrt(unc) for val, unc in zip(sigmaw2_values, sigmaw2_uncertainties)]

    # 绘制sigmaw不确定性范围
    plt.fill_between(evaluation_points[skip:], sigmaw2_lower_bound[skip:], sigmaw2_upper_bound[skip:], color='blue', alpha=0.2, label='sigmaw^2 uncertainty')

    plt.xlabel('Time')
    plt.ylabel('sigmaw')
    plt.title(f'True sigmaw2={true_sigmaw2}, Final sigmaw2 = {final_sigmaw2_estimate} (±{np.sqrt(sigmaw2_uncertainties[-1])})')
    plt.legend()
    plt.show()







# 运行性能分析
profiler = cProfile.Profile()
profiler.enable()
main()
profiler.disable()

# 输出性能报告
stats = pstats.Stats(profiler).sort_stats('cumtime')
stats.print_stats(100)