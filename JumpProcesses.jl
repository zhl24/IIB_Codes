module JumpProcesses


using Random, Distributions, LinearAlgebra


export NormalGammaProcess, SDE, generateGammaSamples, generateSamples, integrate

# 基础集成函数
function integrate(evaluation_points, x_series, t_series)
    outputs = Vector{Float64}[]
    for i in evaluation_points
        output = 0.0
        for (x, t) in zip(x_series, t_series)
            if t <= i
                output += x
            end
        end
        push!(outputs, output)
    end
    return outputs
end

# 正态伽马过程
struct NormalGammaProcess
    beta::Float64
    C::Float64
    T::Float64
    muw::Float64
    sigmaw::Float64
end

# 生成伽马样本
function generateGammaSamples(process::NormalGammaProcess, evaluation_points; raw_data=false, gamma_generator=gamma_process)
    x_list, jump_times = gamma_generator(process.beta, process.C, process.T)
    if raw_data
        return integrate(evaluation_points, x_list, jump_times), x_list, jump_times
    else
        return integrate(evaluation_points, x_list, jump_times)
    end
end

# 生成样本
function generateSamples(process::NormalGammaProcess, evaluation_points; raw_data=false, all_data=false)
    gamma_paths, subordinator_jumps, jump_times = generateGammaSamples(process, evaluation_points, raw_data=true)
    NVM_jumps = process.muw .* subordinator_jumps .+ process.sigmaw .* sqrt.(subordinator_jumps) .* randn(length(subordinator_jumps))

    if all_data
        return integrate(evaluation_points, NVM_jumps, jump_times), NVM_jumps, jump_times, subordinator_jumps
    else
        return integrate(evaluation_points, NVM_jumps, jump_times)
    end
end

# SDE模型
struct SDE
    A::Matrix{Float64}
    h::Vector{Float64}
    T::Float64
    NVM::NormalGammaProcess
    X_0::Float64
end

# 生成SDE样本
function generateSamples(sde::SDE, evaluation_points; plot_NVM=false, plot_jumps=false, all_data=false)
    paths, jumps, jump_times, subordinator_jumps = generateSamples(sde.NVM, evaluation_points, all_data=true)
    samples = []
    for (i, evaluation_point) in enumerate(evaluation_points)
        system_jumps = [j * exp(sde.A * (evaluation_point - t)) * sde.h for (j, t) in zip(jumps, jump_times)]
        push!(samples, integrate([evaluation_point], system_jumps, jump_times)[1])
    end

    if all_data
        return samples, jumps, subordinator_jumps, jump_times
    else
        return samples
    end
end

end # module
