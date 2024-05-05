module Julia_Implementation
using Random
using Distributions
using LinearAlgebra
using Statistics
using StatsFuns: logsumexp
using Base.Threads



export vectorized_particle_Gamma_generator
export integrate
export expm_specialized
export log_probs_to_normalised_probs
export weighted_sum
export inverted_gamma_to_mean_variance
export generate_SDE_samples
export vectorized_particle_transition_function
###Chatgpt translation from here
export compute_augmented_matrices, accelerated_alphaw_betaw_update, ultimate_NVM_pf_accelerated_code, resample_particles
export ultimate_NVM_pf





function vectorized_particle_Gamma_generator(beta, C, T, resolution, num_particles, c=50) #Resolutuon is the total number of data points, and T is the total length of the time frame
    dt = T / resolution
    samples_matrix = rand(Exponential(1/T), num_particles, c*resolution)
    @threads for n in 1:num_particles
        t_array = Int64.(1:resolution)
        start_indices = (t_array.-1).*c.+1
        end_indices = start_indices .+ c .- 1
        @threads for t in t_array
            start_index = start_indices[t]
            end_index = end_indices[t]
            samples_matrix[n, start_index:end_index] = cumsum(samples_matrix[n, start_index:end_index])
        end
    end
    @threads for i in 1:num_particles
        for j in 1:(c*resolution)
            poisson_epoch = samples_matrix[i, j]
            x = 1 / (beta * (exp(poisson_epoch / C) - 1))
            p = (1 + beta * x) * exp(-beta * x)
            samples_matrix[i, j] = rand() <= p ? x : 0
        end
    end
    jump_time_matrix = rand(Uniform(0, dt), num_particles, c*resolution)
    return samples_matrix, jump_time_matrix
end


function expm_specialized(A, t_matrix)
    theta = A[2, 2]
    exp_theta_t = exp.(theta .* t_matrix)  # 对每个元素进行指数计算
    
    # 定义 mat1 和 mat2
    mat1 = [0.0 1.0/theta; 0.0 1.0]
    mat2 = [1.0 -1.0/theta; 0.0 0.0]

    # 初始化结果数组，每个元素是独立的2x2矩阵
    result = Array{Matrix{Float64}, 2}(undef, size(t_matrix, 1), size(t_matrix, 2))
    
    @threads for i in 1:size(t_matrix, 1)
        @threads for j in 1:size(t_matrix, 2)
            # 计算每个元素对应的结果矩阵
            result[i, j] = exp_theta_t[i, j] * mat1 + mat2
        end
    end

    return result
end



function vectorized_particle_transition_function(beta, C, T, resolution, num_particles, A, h ;c=10) #Modify the basic proposals to the mean and cov forms which would be used to compute the transition matrix anmd Gaussian covariance
    gamma_jump_matrix, jump_time_matrix = vectorized_particle_Gamma_generator(beta, C, T, resolution, num_particles, c)

    expA_tau = expm_specialized(A, -jump_time_matrix) #The exponential matrix for each jump time. In Julia, it would be recognized as of the same shape as jump_time_matrix. Directly ready to be broadcasting.
    expA_tau_h = expA_tau .* [h]
    raw_mean_matrix = gamma_jump_matrix .* expA_tau_h
    pre_cov_matrix = sqrt.(gamma_jump_matrix) .* expA_tau_h
    raw_cov_matrix = (x -> x * x').(pre_cov_matrix) #A broadcasting operation
    mean_matrix = reshape(sum(reshape(raw_mean_matrix, num_particles, resolution, c), dims=3), num_particles, resolution)
    cov_matrix = reshape(sum(reshape(raw_cov_matrix, num_particles, resolution, c), dims=3), num_particles, resolution)

    return mean_matrix, cov_matrix
end



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

    for (i, evaluation_point) in enumerate(evaluation_points)
        sample = 0
        #print(i)
        for (j, jump_time) in enumerate(jump_times)
            NVM_jump = NVM_jumps[j]
            if jump_time < evaluation_point
                 #An internal check here for simplicity
            expt = expm_specialized(A, evaluation_point - jump_time)
            system_jump = NVM_jump * expt[1,1] * h
            #print(expm_specialized(A, evaluation_point - jump_time)*h)
            sample = sample .+ system_jump 
            #print(sample)
            #print(size(sample))
            end
        end
        #Compute the sample value at each time point
        samples[:,i] .= sample
    end

    return samples
end



function log_probs_to_normalised_probs(log_likelihoods)
    log_likelihoods = dropdims(log_likelihoods; dims=1)  # np.squeeze 的等价操作
    log_likelihoods = replace(log_likelihoods, x->isnan(x) ? -Inf : x)  # 替换 NaN 为 -Inf
    normalization_factor = logsumexp(log_likelihoods)

    if !isfinite(normalization_factor)  # 检查是否为 -Inf 或 Inf
        probabilities = fill(1.0 / length(log_likelihoods), length(log_likelihoods))
    else
        normalized_log_likelihoods = log_likelihoods .- normalization_factor
        probabilities = exp.(normalized_log_likelihoods)
        probabilities .= probabilities / sum(probabilities)  # 确保概率之和为1
    end

    return dropdims(probabilities; dims=1)
end



function weighted_sum(particles, weights)
    weighted_sum = zeros(size(particles[1])...)
    
    for (particle, weight) in zip(particles, weights)
        weighted_sum .+= particle .* weight
    end

    return weighted_sum
end

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


function kalman_transit(X, P, f, Q; mw=0, return_marginal=false)
    # Perform matrix multiplication
    X_new = f * X

    X_new += mw
    # Handle control input B and u (如果有，这部分在提供的代码中未包括)

    P_new = f * P * f' + Q

    return X_new, P_new
end

function kalman_correct(X, P, Y, g, R)
    # Innovation term, just the prediction error
    Ino = Y - g * X

    # Innovation covariance
    S = g * P * g' + R

    K = P * g' / S

    X_updated = X + K * Ino
    P_updated = (I - K * g) * P
    log_S = log(S)
    Ino_normalized = Ino / S * Ino

    return X_updated, P_updated, log_S, Ino_normalized
end




function compute_augmented_matrices(matrix_exp, particle_sum, noise_cov)
    n = size(matrix_exp, 1)
    combined_matrix = zeros(n + 1, n + 1)
    combined_matrix[1:n, 1:n] = matrix_exp
    combined_matrix[1:n, end] = particle_sum'
    combined_matrix[end, end] = 1
    
    augmented_cov_matrix = zeros(n + 1, n + 1)
    augmented_cov_matrix[1:n, 1:n] = noise_cov
    
    return combined_matrix, augmented_cov_matrix
end

function accelerated_alphaw_betaw_update(alphaws, betaws, accumulated_Es, accumulated_Fs, new_Es, new_Fs)
    alphaws .+= 0.5
    betaws .+= new_Es .* 0.5
    accumulated_Es .+= new_Es
    accumulated_Fs .+= new_Fs
    
    return alphaws, betaws, accumulated_Es, accumulated_Fs
end

function ultimate_NVM_pf_accelerated_code(M, N, gammaln_half_N_plus_alpha, gammaln_half_N_minus1_plus_alpha, gammaln_alphaws, accumulated_Es, accumulated_Fs, new_Es, new_Fs, alphaws, betaws)
    accumulated_log_marginals = -M*N/2*log(2*pi) + accumulated_Fs + alphaws .* log.(betaws) - (alphaws+N/2) .* log.(betaws + accumulated_Es/2) + gammaln_half_N_plus_alpha - gammaln_alphaws
    log_marginals = -M/2*log(2*pi) + new_Fs - (alphaws+N/2) .* log.(betaws + accumulated_Es/2) + (alphaws + (N-1)/2) .* log.(betaws + (accumulated_Es - new_Es)/2) + gammaln_half_N_plus_alpha - gammaln_half_N_minus1_plus_alpha
    
    return accumulated_log_marginals, log_marginals
end

function resample_particles(num_particles, indices, log_marginals, accumulated_log_marginals, alphaws, betaws, accumulated_Es, accumulated_Fs)
    weights_resampled = ones(num_particles) / num_particles
    
    log_marginals_resampled = log_marginals[indices]
    accumulated_log_marginals_resampled = accumulated_log_marginals[indices]
    
    alphaws_resampled = alphaws[indices]
    betaws_resampled = betaws[indices]
    accumulated_Es_resampled = accumulated_Es[indices]
    accumulated_Fs_resampled = accumulated_Fs[indices]
    
    return weights_resampled, log_marginals_resampled, accumulated_log_marginals_resampled, alphaws_resampled, betaws_resampled, accumulated_Es_resampled, accumulated_Fs_resampled
end


function ultimate_NVM_pf(observation, previous_Xs, previous_X_uncertaintys, mean_proposal, cov_proposal, matrix_exp, g, R, alphaws, betaws, accumulated_Es, accumulated_Fs, N; return_log_marginals = false)
    M = length(observation)
    num_particles = size(mean_proposal, 1)

    Xs_inferred = []
    uncertaintys_inferred = []
    log_marginals = zeros(num_particles)
    accumulated_log_marginals = zeros(num_particles)
    I = Matrix{Float64}(I, 2, 2)
    new_Es = zeros(num_particles)
    new_Fs = zeros(num_particles)

    for i in 1:num_particles
        particle_sum = mean_proposal[i, :]
        noise_cov = cov_proposal[i, :, :]
        previous_X = previous_Xs[i]
        previous_X_uncertainty = previous_X_uncertaintys[i]
        combined_matrix, augmented_cov_matrix = compute_augmented_matrices(matrix_exp, particle_sum, noise_cov)
        inferred_X, inferred_cov = Kalman_transit(previous_X, previous_X_uncertainty, combined_matrix, augmented_cov_matrix)
        inferred_X, inferred_cov, log_det_F, Ei = Kalman_correct(inferred_X, inferred_cov, observation, g, R, I)
        new_Es[i] = Ei
        new_Fs[i] = -0.5 * log_det_F
        push!(Xs_inferred, inferred_X)
        push!(uncertaintys_inferred, inferred_cov)
    end

    alphaws, betaws, accumulated_Es, accumulated_Fs = accelerated_alphaw_betaw_update(alphaws, betaws, accumulated_Es, accumulated_Fs, new_Es, new_Fs)
    gammaln_half_N_plus_alpha = gammaln(N/2 + alphaws)
    gammaln_half_N_minus1_plus_alpha = gammaln((N-1)/2 + alphaws)
    gammaln_alphaws = gammaln(alphaws)

    accumulated_log_marginals, log_marginals = ultimate_NVM_pf_accelerated_code(M, N, gammaln_half_N_plus_alpha, gammaln_half_N_minus1_plus_alpha, gammaln_alphaws, accumulated_Es, accumulated_Fs, new_Es, new_Fs, alphaws, betaws)

    weights = log_probs_to_normalised_probs(log_marginals)
    indices = rand(1:num_particles, num_particles, Weights(weights))

    uncertaintys_inferred_resampled = [uncertaintys_inferred[i] for i in indices]
    Xs_inferred_resampled = [Xs_inferred[i] for i in indices]

    uncertaintys_inferred = uncertaintys_inferred_resampled
    Xs_inferred = Xs_inferred_resampled

    weights, log_marginals, accumulated_log_marginals, alphaws, betaws, accumulated_Es, accumulated_Fs = resample_particles(num_particles, indices, log_marginals, accumulated_log_marginals, alphaws, betaws, accumulated_Es, accumulated_Fs)

    if return_log_marginals
        return hcat(Xs_inferred...), hcat(uncertaintys_inferred...), mean_proposal, cov_proposal, weights, alphaws, betaws, accumulated_Es, accumulated_Fs, accumulated_log_marginals
    else
        return hcat(Xs_inferred...), hcat(uncertaintys_inferred...), mean_proposal, cov_proposal, weights, alphaws, betaws, accumulated_Es, accumulated_Fs
    end
end


end
