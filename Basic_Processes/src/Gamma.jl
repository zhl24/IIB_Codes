import Random
import Distributions
import Plots
function gamma_process(beta,C,T)
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
    N_Ga = []
    for i = poisson_epochs
        x = 1/(beta*(exp(i/C)-1))
        p = (1+beta*x)*exp(-beta*x)
        if rand(Distributions.Bernoulli(p))
            append!(x_series,x)
            if x in N_Ga
                continue
            else
                append!(N_Ga,x)
            end
        end
    end
    jump_times = []
    for x = x_series #Or N_Ga?
        append!(jump_times,rand()*T)
    end
    return x_series,jump_times
end