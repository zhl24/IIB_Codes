module Julia_Implementation
using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsFuns: logsumexp
using .Threads
using ProgressMeter
using SpecialFunctions
using StatsBase
using Plots


export vectorized_particle_Gamma_generator
export integrate
export expm_specialized
export log_probs_to_normalised_probs
export weighted_sum
export inverted_gamma_to_mean_variance
export generate_SDE_samples
export vectorized_particle_transition_function
export compute_augmented_matrices
export NVM_mpf_1tstep
export Normal_Gamma_Langevin_MPF
export Normal_Gamma_Langevin_GRW_MCMC
export plot_samples_distribution
export thin_samples
export parallel_particle_Gamma_generator


#Plotting Functions ##############################################################################################################################################################
function plot_samples_distribution(samples, true_value, title;bin_num = 30)
    plt = histogram(samples, bins=bin_num, normalize=false, alpha=0.6, label="Samples Distribution", title=title, xlabel="Value", ylabel="Frequency")
    vline!([mean(samples)], label="Mean", color=:red)
    vline!([true_value], label="True Value", color=:green)

    # Calculate 95% confidence interval
    conf_interval = quantile(samples, [0.025, 0.975])
    vline!(conf_interval, label="95% CI", color=:blue, linestyle=:dash)
    return plt
end


#Thinning Function ######################################################################################################################################################
function thin_samples(samples::Array, thin_factor::Int)
    # 检查 thin_factor 是否合理
    if thin_factor <= 0
        error("Thinning factor must be a positive integer.")
    end
    
    # 从样本中每隔 thin_factor 取一个样本
    thinned_samples = samples[1:thin_factor:end]
    return thinned_samples
end


#Proposal Generator########################################################################################################################################################################################################
function vectorized_particle_Gamma_generator(beta::Float64, C::Float64, T::Float64, resolution::Int, num_particles::Int, c::Int=50; ) #Resolutuon is the total number of data points, and T is the total length of the time frame
    dt = T / resolution
    #rng = MersenneTwister()  # 创建一个随机数生成器实例
    rng = Xoshiro()
    # 使用显式的 RNG 实例生成随机数
    samples_matrix = rand(rng, Exponential(1/dt), num_particles, c*resolution)
    jump_time_matrix = rand(rng, Uniform(0, dt), num_particles, c*resolution)
    for n in 1:num_particles
        t_array = Int64.(1:resolution)
        start_indices = (t_array.-1).*c.+1
        end_indices = start_indices .+ c .- 1
        for t in t_array
            start_index = start_indices[t]
            end_index = end_indices[t]
            samples_matrix[n, start_index:end_index] = cumsum(samples_matrix[n, start_index:end_index])
        end
    end
    for i in 1:num_particles
        for j in 1:(c*resolution)
            poisson_epoch = samples_matrix[i, j]
            x = 1 / (beta * (exp(poisson_epoch / C) - 1))
            p = (1 + beta * x) * exp(-beta * x)
            samples_matrix[i, j] = rand() <= p ? x : 0
        end
    end
    #jump_time_matrix = rand(Uniform(0, dt), num_particles, c*resolution)
    
    return samples_matrix, jump_time_matrix
end


function parallel_particle_Gamma_generator(beta::Float64, C::Float64, T::Float64, resolution::Int, num_particles::Int, c::Int=50; ) #Resolutuon is the total number of data points, and T is the total length of the time frame
    dt = T / resolution
    #rng = MersenneTwister()  # 创建一个随机数生成器实例
    rng = Xoshiro()
    # 使用显式的 RNG 实例生成随机数
    samples_matrix = rand(rng, Exponential(1/dt), num_particles, c*resolution)
    jump_time_matrix = rand(rng, Uniform(0, dt), num_particles, c*resolution)
    for n in 1:num_particles
        t_array = Int64.(1:resolution)
        start_indices = (t_array.-1).*c.+1
        end_indices = start_indices .+ c .- 1
        for t in t_array
            start_index = start_indices[t]
            end_index = end_indices[t]
            samples_matrix[n, start_index:end_index] = cumsum(samples_matrix[n, start_index:end_index])
        end
    end
    for i in 1:num_particles
        for j in 1:(c*resolution)
            poisson_epoch = samples_matrix[i, j]
            x = 1 / (beta * (exp(poisson_epoch / C) - 1))
            p = (1 + beta * x) * exp(-beta * x)
            samples_matrix[i, j] = rand() <= p ? x : 0
        end
    end
    #jump_time_matrix = rand(Uniform(0, dt), num_particles, c*resolution)
    
    return samples_matrix, jump_time_matrix
end



#Compatible with single scalar t or a whole t matrix
#function expm_specialized(A, t_matrix)
#    exp_theta_t = exp.(A[2, 2] .* t_matrix)  # 对每个元素进行指数计算
#    
#    # 定义 mat1 和 mat2
#    mat1 = [0.0 1.0/A[2, 2]; 0.0 1.0]
#    mat2 = [1.0 -1.0/A[2, 2]; 0.0 0.0]
#
#    # 初始化结果数组，每个元素是独立的2x2矩阵
#    result = Array{Matrix{Float64}, 2}(undef, size(t_matrix, 1), size(t_matrix, 2))
#    
#    for i in 1:size(t_matrix, 1)
#        for j in 1:size(t_matrix, 2)
#            # 计算每个元素对应的结果矩阵
#            result[i, j] = exp_theta_t[i, j] * mat1 + mat2
#        end
#    end
#
#    return result
#end

function expm_specialized(A, t)
    return [0 1/A[2,2];0 1] .* exp(A[2, 2] *t) + [1  -1/ A[2,2]; 0 0]
end




function vectorized_particle_transition_function(beta, C, T, resolution, num_particles, A, h ;c=10) #Modify the basic proposals to the mean and cov forms which would be used to compute the transition matrix anmd Gaussian covariance
    gamma_jump_matrix, jump_time_matrix = vectorized_particle_Gamma_generator(beta, C, T, resolution, num_particles, c)
    
    theta = A[2, 2]
    exp_theta_t = exp.(theta .* jump_time_matrix)  # 对每个元素进行指数计算
    
    # 定义 mat1 和 mat2
    temp_mean_matrix = [0.0 1.0/theta; 0.0 1.0] * h
    temp_mat2_h = [1.0 -1.0/theta; 0.0 0.0] * h
    # 初始化结果2D数组
    mean_matrix = Array{Matrix{Float64}, 2}(undef, size(jump_time_matrix, 1), size(jump_time_matrix, 2))
    cov_matrix = Array{Matrix{Float64}, 2}(undef, size(jump_time_matrix, 1), size(jump_time_matrix, 2))
    for i in 1:num_particles
        for j in 1:resolution * c
            mean_matrix[i, j] = exp_theta_t[i, j]*temp_mean_matrix + temp_mat2_h
            cov_matrix[i,j] = mean_matrix[i, j] * mean_matrix[i, j]' * gamma_jump_matrix[i,j]
            mean_matrix[i,j] *= gamma_jump_matrix[i,j]
        end
    end
    mean_matrix = reshape(sum(reshape(mean_matrix',c , resolution,num_particles), dims=1), resolution, num_particles)'
    cov_matrix = reshape(sum(reshape(cov_matrix',c, resolution, num_particles), dims=1), resolution, num_particles)'

    return mean_matrix, cov_matrix
end





# SDE Samples Generator #########################################################################################################################################################################################


function integrate(evaluation_points, x_series, t_series)
    # 初始化结果数组
    results = zeros(length(evaluation_points))
    
    for (i, point) in enumerate(evaluation_points)
        sum_ = 0.0
        for (x, t) in zip(x_series, t_series)
            if t < point
                sum_ += x
            end
        end
        results[i] = sum_
    end
    
    return results
end

#Only iterate for once, so not optimized at all.
function generate_SDE_samples(subordinator_jumps,jump_times,muw,sigmaw,A,h,evaluation_points) #The subordinator to be used is Gamma. jump times are inherited across all jump types
    NVM_jumps = muw .* subordinator_jumps .+ sigmaw .* sqrt.(subordinator_jumps) .* randn(size(subordinator_jumps))
    samples = zeros(2,length(evaluation_points))
    #system_jumps = zeros(2,length(evaluation_points))
    for (i, evaluation_point) in enumerate(evaluation_points)
        sample = zeros(2,1)
        #print(i)
        for (j, jump_time) in enumerate(jump_times)
            NVM_jump = NVM_jumps[j]
            if jump_time < evaluation_point
                expt = expm_specialized(A, evaluation_point - jump_time)
                #println(expt)
                #expt = exp(A .*(evaluation_point - jump_time))
                system_jump =  expt * (NVM_jump .* h)
                #system_jump = expt* h .* NVM_jump
                sample = sample .+ system_jump 

            end
        end
        #Compute the sample value at each time point
        samples[:,i] = sample
    end

    return samples
end






# Filters####################################################################################################################################################################################################

function log_probs_to_normalised_probs(log_likelihoods)
    normalization_factor = logsumexp(log_likelihoods)
    if !isfinite(normalization_factor)  # 检查是否为 -Inf 或 Inf
        probabilities = fill(1.0 / length(log_likelihoods), length(log_likelihoods))
    else
        normalized_log_likelihoods = log_likelihoods .- normalization_factor
        probabilities = exp.(normalized_log_likelihoods)
        probabilities = probabilities / sum(probabilities)  # 确保概率之和为1
    end

    return probabilities
end



function weighted_sum(particles, weights)
    weighted_sum = zeros(size(particles[1])...)
    
    for (particle, weight) in zip(particles, weights)
        weighted_sum .+= particle .* weight
    end

    return weighted_sum
end


#Combine alphaws and betaws, i.e. inverted Gamma distributions, inferred by particles.
function inverted_gamma_to_mean_variance(alphas, betas, weights)
    particle_mean_vector = betas ./ (alphas .- 1)
    mean = dot(weights, particle_mean_vector)

    particle_mean_squared = particle_mean_vector .^ 2
    particle_variance = particle_mean_squared ./ (alphas .- 2)
    particle_x2_expectation = particle_variance + particle_mean_squared

    weighted_sum_x2 = dot(weights, particle_x2_expectation)
    variance = weighted_sum_x2 - mean^2

    return mean, variance
end


function Kalman_transit(X, P, f, Q)
    return f * X, f * P * f' + Q
end

function Kalman_correct(X, P, Y, g, R)
    # Innovation term, just the prediction error
    Ino = Y - dot(g,X)
    # Innovation covariance
    S = abs(dot(g ,P * g') + R)
    K = P * g' ./ S
    return X + K * Ino, (I - K * g) * P, log(S), Ino / S * Ino
end




function compute_augmented_matrices(matrix_exp, particle_sum, noise_cov)
    n = size(matrix_exp, 1)
    combined_matrix = zeros(n + 1, n + 1)
    #println(size(noise_cov))
    #println(size(matrix_exp))
    #println(size(combined_matrix))
    combined_matrix[1:n, 1:n] = matrix_exp
    combined_matrix[1:n, end] = particle_sum'
    combined_matrix[end, end] = 1.0
    
    augmented_cov_matrix = zeros(n + 1, n + 1)
    augmented_cov_matrix[1:n, 1:n] = noise_cov
    
    return combined_matrix, augmented_cov_matrix
end




function NVM_mpf_1tstep(observation, previous_Xs, previous_X_uncertaintys, mean_proposal, cov_proposal, matrix_exp, g, R, alphaws, betaws, accumulated_Es, accumulated_Fs, N)
    #N is the time index point
    M = 1.0 #single observation
    num_particles = size(mean_proposal, 1)

    #Vector and Matrix Containers Pre-allocation
    Xs_inferred = similar(previous_Xs)
    uncertaintys_inferred = similar(previous_X_uncertaintys)

    #Scalar Containers Pre-allocation
    log_marginals = zeros(num_particles)
    accumulated_log_marginals = zeros(num_particles)
    new_Es = zeros(num_particles)
    new_Fs = zeros(num_particles)
    n = size(matrix_exp, 1)

    #Intermediate Variable Initialization for the For Loop
    combined_matrix = zeros(n + 1, n + 1)
    augmented_cov_matrix = copy(combined_matrix)
    inferred_X = similar(mean_proposal)
    inferred_cov = similar(cov_proposal)


    for i in 1:num_particles
        combined_matrix, augmented_cov_matrix = compute_augmented_matrices(matrix_exp, mean_proposal[i],cov_proposal[i])
        inferred_X, inferred_cov = Kalman_transit(previous_Xs[i], previous_X_uncertaintys[i], combined_matrix, augmented_cov_matrix)
        inferred_X, inferred_cov, log_det_F, Ei = Kalman_correct(inferred_X, inferred_cov, observation, g, R)
        new_Es[i] = Ei
        new_Fs[i] = -0.5 * log_det_F
        Xs_inferred[i] = inferred_X
        uncertaintys_inferred[i] = inferred_cov
    end
    #Already in JIT, so no need to extract the code below
    alphaws .+= 0.5
    betaws .+= new_Es .* 0.5
    accumulated_Es .+= new_Es
    accumulated_Fs .+= new_Fs
    
    gammaln_half_N_plus_alpha = lgamma.(N/2.0 .+ alphaws)
    gammaln_half_N_minus1_plus_alpha = lgamma.((N-1.0)/2.0 .+ alphaws)
    gammaln_alphaws = lgamma.(alphaws)

    
    log_marginals = -M/2*log(2.0*pi) .+ new_Fs .- (alphaws.+N/2.0) .* log.(betaws .+ accumulated_Es./2.0) .+ (alphaws .+ (N-1.0)/2.0) .* log.(betaws .+ (accumulated_Es .- new_Es)./2.0) .+ gammaln_half_N_plus_alpha .- gammaln_half_N_minus1_plus_alpha
    accumulated_log_marginals = copy(log_marginals)
    #Resampling
    weights = log_probs_to_normalised_probs(log_marginals)
    indices = sample(1:num_particles, Weights(weights), num_particles; replace=true)

    uncertaintys_inferred = uncertaintys_inferred[indices]
    Xs_inferred = Xs_inferred[indices]
    weights = ones(num_particles) ./ num_particles
    log_marginals = log_marginals[indices]
    #accumulated_log_marginals = accumulated_log_marginals[indices]
    alphaws = alphaws[indices]
    betaws = betaws[indices]
    accumulated_Es = accumulated_Es[indices]
    accumulated_Fs = accumulated_Fs[indices]

    #accumulated_log_marginals = log_marginals
    
    return Xs_inferred, uncertaintys_inferred, weights, alphaws, betaws, accumulated_Es, accumulated_Fs, accumulated_log_marginals

end




#The complete marginalised particle filter function. The corresponding model is the extended state one in the Levy state space model paper. 
function Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta,beta,C,kw,alphaw_prior,betaw_prior,kv)
    #observation is the noisy observation to be inferred
    #resolution and T are the parameters used to construct the time axis
    #num_particles is just number of particles
    #theta is the Langevin parameter. beta and C are the parameters for the Gamma process.
    #kw is the prior parameter for muw inference
    #alphaw_prior and betaw_prior are scalar parameters for the prior in posterior inference of sigmaw
    # Kalman parameters are just Kalman parameters
    #kv is the guess for the relative observation noise



    #Langevin System Specifications
    A = zeros(2,2)
    A[1,2] = 1.0
    A[2,2] = theta
    h = zeros(2,1)
    h[2,1] = 1

    #Prior construction
    alphaws = alphaw_prior .* ones(num_particles)
    betaws = betaw_prior .* ones(num_particles)

    #Kalman Filter Initialization
    X0 = zeros(3,1) #Do not use list definition, would cause the compiler to simplify the dimension
    C0 = zeros(3,3)
    C0[3,3] = kw
    g = zeros(1,3)#Observation matrix for single state
    g[1,1] = 1
    R = kv #The single state observation noise covariance. Marginalised Kalman here, so this is the relative noise scale factor.

    #Build the time axis
    evaluation_points = collect(range(0,T,resolution)) #Use collect to convert the range object to array
    dt = T/resolution #The time step size
    matrix_exp = expm_specialized(A,dt) #Constant factor used to construct the transition matrix
    #println(size(A))
    #println(size(matrix_exp))
    #Particle Filter Initialization
    mean_proposals , cov_proposals = vectorized_particle_transition_function(beta, C, T, resolution, num_particles, A, h ;c=10)
    previous_Xs = [copy(X0) for _ in 1:num_particles]
    previous_X_uncertaintys = [copy(C0) for _ in 1:num_particles]
    weights = ones(num_particles)/num_particles

    #Result conatiners pre-allocation
    sigmaw2_means = zeros(resolution)
    sigmaw2_uncertaintys = zeros(resolution) #Both are scalar terms since the parameter inferred is scalar
    accumulated_Es = zeros(num_particles)
    accumulated_Fs = zeros(num_particles)
    accumulated_log_marginals = zeros(num_particles)
    inferred_Xs = [copy(X0) for _ in 1:resolution]
    inferred_covs = [copy(C0) for _ in 1:resolution]


    #Running the marginalised particle filter
    #@showprogress 
    @showprogress 18000 for i in 1:resolution
        t = evaluation_points[i]
        observation = observations[i]

        mean_proposal = mean_proposals[:,i] #The particle proposals at the ith time point
        cov_proposal = cov_proposals[:,i] #Similarly

        sigmaw2,sigmaw2_uncertainty = inverted_gamma_to_mean_variance(alphaws,betaws,weights)
        sigmaw2_means[i] = sigmaw2
        sigmaw2_uncertaintys[i] = sigmaw2_uncertainty

        previous_Xs, previous_X_uncertaintys, weights, alphaws, betaws, accumulated_Es, accumulated_Fs, accumulated_log_marginals = NVM_mpf_1tstep(observation, previous_Xs, previous_X_uncertaintys, mean_proposal, cov_proposal, matrix_exp, g, R, alphaws, betaws, accumulated_Es, accumulated_Fs, i)

        inferred_Xs[i] = weighted_sum(previous_Xs,weights) 
        inferred_covs[i] = weighted_sum(previous_X_uncertaintys,weights) * sigmaw2

    end

    return inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals

end




#The Overloaded version specifically for particle MCMC, just taking away the common initialization and passed as parameters
function Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta,beta,C, h, alphaws, betaws, X0,C0, g,R, evaluation_points)
    #observation is the noisy observation to be inferred
    #resolution and T are the parameters used to construct the time axis
    #num_particles is just number of particles
    #theta is the Langevin parameter. beta and C are the parameters for the Gamma process.
    #kw is the prior parameter for muw inference
    #alphaw_prior and betaw_prior are scalar parameters for the prior in posterior inference of sigmaw
    # X0 and C0 are the Kalman prior parameters
    #kv is the guess for the relative observation noise
    A = zeros(2,2)
    A[1,2] = 1.0
    A[2,2] = theta
    dt = T/resolution #The time step size
    matrix_exp = expm_specialized(A,dt) #Constant factor used to construct the transition matrix


    mean_proposals , cov_proposals = vectorized_particle_transition_function(beta, C, T, resolution, num_particles, A, h ;c=10)
    previous_Xs = [copy(X0) for _ in 1:num_particles]
    previous_X_uncertaintys = [copy(C0) for _ in 1:num_particles]
    weights = ones(num_particles)/num_particles

    #Result conatiners pre-allocation
    sigmaw2_means = zeros(resolution)
    sigmaw2_uncertaintys = zeros(resolution) #Both are scalar terms since the parameter inferred is scalar
    accumulated_Es = zeros(num_particles)
    accumulated_Fs = zeros(num_particles)
    accumulated_log_marginals = zeros(num_particles)
    inferred_Xs = [copy(X0) for _ in 1:resolution]
    inferred_covs = [copy(C0) for _ in 1:resolution]


    #Running the marginalised particle filter
    for i in 1:resolution
        t = evaluation_points[i]
        observation = observations[i]

        mean_proposal = mean_proposals[:,i] #The particle proposals at the ith time point
        cov_proposal = cov_proposals[:,i] #Similarly

        sigmaw2,sigmaw2_uncertainty = inverted_gamma_to_mean_variance(alphaws,betaws,weights)
        sigmaw2_means[i] = sigmaw2
        sigmaw2_uncertaintys[i] = sigmaw2_uncertainty

        previous_Xs, previous_X_uncertaintys, weights, alphaws, betaws, accumulated_Es, accumulated_Fs, accumulated_log_marginals = NVM_mpf_1tstep(observation, previous_Xs, previous_X_uncertaintys, mean_proposal, cov_proposal, matrix_exp, g, R, alphaws, betaws, accumulated_Es, accumulated_Fs, i)

        inferred_Xs[i] = weighted_sum(previous_Xs,weights)
        inferred_covs[i] = weighted_sum(previous_X_uncertaintys,weights) * sigmaw2

    end

    return inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals

end



function Normal_Gamma_Langevin_Grid_Search(observations,resolution,T,num_particles,kw ,alphaw_prior,betaw_prior,kv, theta_values,beta_values, C_values)
    

    #Langevin System Specifications
    h = zeros(2,1)
    h[2,1] = 1

    #Prior construction
    alphaws = alphaw_prior .* ones(num_particles)
    betaws = betaw_prior .* ones(num_particles)

    #Kalman Filter Initialization
    X0 = zeros(3,1) #Do not use list definition, would cause the compiler to simplify the dimension
    Cov0 = zeros(3,3)
    Cov0[3,3] = kw
    g = zeros(1,3)#Observation matrix for single state
    g[1,1] = 1
    R = kv #The single state observation noise covariance. Marginalised Kalman here, so this is the relative noise scale factor.

    #Build the time axis
    evaluation_points = collect(range(0,T,resolution)) #Use collect to convert the range object to array

    #Container pre-allocation
    # 预分配 inferred_Xs 和 inferred_covs
    inferred_Xs = [Matrix{Float64}(undef, 2, 2) for _ in 1:resolution]
    inferred_covs = [Matrix{Float64}(undef, 2, 2) for _ in 1:resolution]
    # 预分配 sigmaw2_means 和 sigmaw2_uncertaintys
    sigmaw2_means = Vector{Float64}(undef, resolution)
    sigmaw2_uncertaintys = Vector{Float64}(undef, resolution)
    # 预分配 accumulated_Es, accumulated_Fs, 和 accumulated_log_marginals
    accumulated_Es = Vector{Float64}(undef, num_particles)
    accumulated_Fs = Vector{Float64}(undef, num_particles)
    accumulated_log_marginals = Vector{Float64}(undef, num_particles)

    l_theta = length(theta_values)
    l_beta = length(beta_values)
    l_C = length(C_values)
    log_probabilities = zeros(l_theta,l_beta,l_C)
    @showprogress for i in 1:l_theta
        for j in 1:l_beta
            for k in 1:l_C
                theta = theta_values[i]
                beta = beta_values[j]
                C = C_values[k]
                #The redundant returned values are kept for possible future use. So far, only the accumulated_log_marginals is used
                inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals = Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta,beta,C, h, alphaws, betaws, X0,Cov0, g,R, evaluation_points)
                log_state_probability = logsumexp(accumulated_log_marginals) - log(num_particles)
                log_probabilities[i,j,k] = log_state_probability

            end

            
        end
    end
    return log_probabilities

end




function Normal_Gamma_Langevin_GRW_MCMC(observations,resolution,T,num_particles,num_iter,kw ,alphaw_prior,betaw_prior,kv, theta0,beta0, C0, l_theta0,l_beta0,l_C0)
    
    theta_samples = zeros(num_iter+1)
    beta_samples = zeros(num_iter+1)
    C_samples = zeros(num_iter+1)
    theta_samples[1] = theta0
    beta_samples[1] = beta0
    C_samples[1] = C0

    #Langevin System Specifications
    h = zeros(2,1)
    h[2,1] = 1

    #Prior construction
    alphaws = alphaw_prior .* ones(num_particles)
    betaws = betaw_prior .* ones(num_particles)

    #Kalman Filter Initialization
    X0 = zeros(3,1) #Do not use list definition, would cause the compiler to simplify the dimension
    Cov0 = zeros(3,3)
    Cov0[3,3] = kw
    g = zeros(1,3)#Observation matrix for single state
    g[1,1] = 1
    R = kv #The single state observation noise covariance. Marginalised Kalman here, so this is the relative noise scale factor.

    #Build the time axis
    evaluation_points = collect(range(0,T,resolution)) #Use collect to convert the range object to array

    #Container pre-allocation
    # 预分配 inferred_Xs 和 inferred_covs
    inferred_Xs = [Matrix{Float64}(undef, 2, 2) for _ in 1:resolution]
    inferred_covs = [Matrix{Float64}(undef, 2, 2) for _ in 1:resolution]
    # 预分配 sigmaw2_means 和 sigmaw2_uncertaintys
    sigmaw2_means = Vector{Float64}(undef, resolution)
    sigmaw2_uncertaintys = Vector{Float64}(undef, resolution)
    # 预分配 accumulated_Es, accumulated_Fs, 和 accumulated_log_marginals
    accumulated_Es = Vector{Float64}(undef, num_particles)
    accumulated_Fs = Vector{Float64}(undef, num_particles)
    accumulated_log_marginals = Vector{Float64}(undef, num_particles)



    #GRW MCMC
    #Initialization
    inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals = Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta0,beta0,C0, h, alphaws, betaws, X0,Cov0, g,R, evaluation_points)
    previous_log_state_probability = logsumexp(accumulated_log_marginals) - log(num_particles)
    current_log_state_probability = 0.0
    acceptance_log_probabilities = zeros(num_iter)
    #Initial Proposal
    theta = theta_samples[1]+ randn() * l_theta0
    beta = abs(beta_samples[1]+ randn() * l_beta0)
    C = abs(C_samples[1] + randn() * l_C0)


    @showprogress for i = 1:num_iter
        #The redundant returned values are kept for possible future use. So far, only the accumulated_log_marginals is used
        inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals = Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta,beta,C, h, alphaws, betaws, X0,Cov0, g,R, evaluation_points)
        current_log_state_probability = logsumexp(accumulated_log_marginals) - log(num_particles)
        acceptance_log_probabilities[i] = current_log_state_probability - previous_log_state_probability
        #Acceptance Case
        if log(rand()) < acceptance_log_probabilities[i]
            #Accept the samples
            theta_samples[i+1] = copy(theta)
            C_samples[i+1] = copy(C)
            beta_samples[i+1] = copy(beta)
            previous_log_state_probability = current_log_state_probability
        else
            theta_samples[i+1] = theta_samples[i]
            C_samples[i+1] = C_samples[i]
            beta_samples[i+1] = beta_samples[i]
        end

        #Propose the new positions via GRW
        theta = theta_samples[i+1] + randn() * l_theta0
        beta = abs(beta_samples[i+1] + randn() * l_beta0)
        C = abs(C_samples[i+1] + randn() * l_C0)
    end

    return theta_samples, beta_samples, C_samples,acceptance_log_probabilities

end


#This PMCMC algorithm basically recompute also the current state likelihood in every iteration to help movement
function Normal_Gamma_Langevin_GRW_MCMC_double_update(observations,resolution,T,num_particles,num_iter,kw ,alphaw_prior,betaw_prior,kv, theta0,beta0, C0, l_theta0,l_beta0,l_C0;rejection_limit = 3)
    
    theta_samples = zeros(num_iter+1)
    beta_samples = zeros(num_iter+1)
    C_samples = zeros(num_iter+1)
    theta_samples[1] = theta0
    beta_samples[1] = beta0
    C_samples[1] = C0

    #Langevin System Specifications
    h = zeros(2,1)
    h[2,1] = 1

    #Prior construction
    alphaws = alphaw_prior .* ones(num_particles)
    betaws = betaw_prior .* ones(num_particles)

    #Kalman Filter Initialization
    X0 = zeros(3,1) #Do not use list definition, would cause the compiler to simplify the dimension
    Cov0 = zeros(3,3)
    Cov0[3,3] = kw
    g = zeros(1,3)#Observation matrix for single state
    g[1,1] = 1
    R = kv #The single state observation noise covariance. Marginalised Kalman here, so this is the relative noise scale factor.

    #Build the time axis
    evaluation_points = collect(range(0,T,resolution)) #Use collect to convert the range object to array

    #Container pre-allocation
    # 预分配 inferred_Xs 和 inferred_covs
    inferred_Xs = [Matrix{Float64}(undef, 2, 2) for _ in 1:resolution]
    inferred_covs = [Matrix{Float64}(undef, 2, 2) for _ in 1:resolution]
    # 预分配 sigmaw2_means 和 sigmaw2_uncertaintys
    sigmaw2_means = Vector{Float64}(undef, resolution)
    sigmaw2_uncertaintys = Vector{Float64}(undef, resolution)
    # 预分配 accumulated_Es, accumulated_Fs, 和 accumulated_log_marginals
    accumulated_Es = Vector{Float64}(undef, num_particles)
    accumulated_Fs = Vector{Float64}(undef, num_particles)
    accumulated_log_marginals = Vector{Float64}(undef, num_particles)



    #GRW MCMC
    #Initialization
    inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals = Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta0,beta0,C0, h, alphaws, betaws, X0,Cov0, g,R, evaluation_points)
    previous_log_state_probability = logsumexp(accumulated_log_marginals) - log(num_particles)
    current_log_state_probability = 0.0
    acceptance_log_probabilities = zeros(num_iter)
    dist = Normal(0,1) #The standard normal distribution to be used to compute the truncated acceptance probability
    #Initial Proposal
    theta = theta_samples[1]+ randn() * l_theta0
    beta = abs(beta_samples[1]+ randn() * l_beta0)
    C = abs(C_samples[1] + randn() * l_C0)
    rejection_count = 0 #Re-compute the original state probability

    @showprogress 1800 for i = 1:num_iter
        #The redundant returned values are kept for possible future use. So far, only the accumulated_log_marginals is used
        inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals = Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta,beta,C, h, alphaws, betaws, X0,Cov0, g,R, evaluation_points)
        current_log_state_probability = logsumexp(accumulated_log_marginals) - log(num_particles)
        acceptance_log_probabilities[i] = current_log_state_probability - previous_log_state_probability + logcdf(dist,C_samples[i])- logcdf(dist,C) + logcdf(dist,beta_samples[i]) - logcdf(dist,beta)
        #Acceptance Case
        if log(rand()) < acceptance_log_probabilities[i]
            #Accept the samples
            theta_samples[i+1] = copy(theta)
            C_samples[i+1] = copy(C)
            beta_samples[i+1] = copy(beta)
            previous_log_state_probability = current_log_state_probability
            rejection_count = 0
        #Rejection Case
        else
            theta_samples[i+1] = theta_samples[i]
            C_samples[i+1] = C_samples[i]
            beta_samples[i+1] = beta_samples[i]
            rejection_count += 1
            if rejection_count > rejection_limit
                inferred_Xs, inferred_covs, sigmaw2_means, sigmaw2_uncertaintys, accumulated_Es, accumulated_Fs, accumulated_log_marginals = Normal_Gamma_Langevin_MPF(observations,resolution,T,num_particles,theta_samples[i],beta_samples[i],C_samples[i], h, alphaws, betaws, X0,Cov0, g,R, evaluation_points)
                previous_log_state_probability = logsumexp(accumulated_log_marginals) - log(num_particles)
                #previous_log_state_probability = logsumexp([previous_log_state_probability + log(rejection_count - rejection_limit), logsumexp(accumulated_log_marginals) - log(num_particles)]) - log(rejection_count-rejection_limit + 1)
                rejection_count = 0
            end
        end

        #Propose the new positions via GRW. Truncated Gaussian is used
        theta = theta_samples[i+1] + randn() * l_theta0
        beta = beta_samples[i+1] + randn() * l_beta0
        while beta <= 0
            beta = beta_samples[i+1] + randn() * l_beta0
        end
        C = C_samples[i+1] + randn() * l_C0
        while C <= 0
            C = C_samples[i+1] + randn() * l_C0
        end
    end

    return theta_samples, beta_samples, C_samples,acceptance_log_probabilities

end




end
