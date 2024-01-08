@turing_testset "is.jl" begin
    function reference(n)
        as = Vector{Float64}(undef, n)
        bs = Vector{Float64}(undef, n)
        logps = Vector{Float64}(undef, n)

        for i in 1:n
            as[i], bs[i], logps[i] = reference()
        end
        logevidence = logsumexp(logps) - log(n)

        return (as = as, bs = bs, logps = logps, logevidence = logevidence)
    end

    function reference()
        x = rand(Normal(4,5))
        y = rand(Normal(x,1))
        loglik = logpdf(Normal(x,2), 3) + logpdf(Normal(y,2), 1.5)
        return x, y, loglik
    end

    @model function normal()
        a ~ Normal(4,5)
        3 ~ Normal(a,2)
        b ~ Normal(a,1)
        1.5 ~ Normal(b,2)
        a, b
    end

    alg = IS()
    seed = 0
    n = 10

    model = normal()
    for i in 1:100
        Random.seed!(seed)
        ref = reference(n)

        Random.seed!(seed)
        chain = sample(model, alg, n)
        sampled = get(chain, [:a, :b, :lp])

        @test vec(sampled.a) == ref.as
        @test vec(sampled.b) == ref.bs
        @test vec(sampled.lp) == ref.logps
        @test chain.logevidence == ref.logevidence
    end

    @turing_testset "logevidence" begin
        Random.seed!(100)

        @model function test()
            a ~ Normal(0, 1)
            x ~ Bernoulli(1)
            b ~ Gamma(2, 3)
            1 ~ Bernoulli(x / 2)
            c ~ Beta()
            0 ~ Bernoulli(x / 2)
            x
        end

        chains = sample(test(), IS(), 10000)

        @test all(isone, chains[:x])
        @test chains.logevidence ≈ - 2 * log(2)
    end
end
