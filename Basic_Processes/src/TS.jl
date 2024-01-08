import Random
import Distributions
import Plots
function tempered_stable_process(alpha,beta,C,T)
    poisson_epochs = []
    current_time = 0.0
    while true
        random_interval = Random.randexp()
        current_time += random_interval
        if current_time > T
            break
        end
        append!(poisson_epochs,current_time)
    end
    
    x_series = []
    N_TS = []
    for i = poisson_epochs
        x = (alpha * i / C)^(-1/alpha)
        p = exp(-beta * x)
        if rand(Distributions.Bernoulli(p))
            append!(x_series,x)
            if x in N_TS
                continue
            else
                append!(N_TS,x)
            end
        end
    end
    jump_times = []
    for x = x_series #Or N_TS?
        append!(jump_times,rand()*T)
    end
    return x_series,jump_times
end