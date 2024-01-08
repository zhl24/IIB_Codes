module Basic_Processes
include("Gamma.jl")
include("TS.jl")

export integrate
export gamma_process
export tempered_stable_process


function integrate(evaluation_points, x_series, t_series)
    outputs = []
    for i = evaluation_points
        output = 0.0
        for j = 1:length(x_series)
            if t_series[j] <= i
                output += x_series[j]
            end
        end
        append!(outputs,output)
    end
    return outputs
end

end # module Basic_Processes
