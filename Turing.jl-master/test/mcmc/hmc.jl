@testset "hmc.jl" begin
    # Set a seed
    rng = StableRNG(123)
    @numerical_testset "constrained bounded" begin
        obs = [0,1,0,1,1,1,1,1,1,1]

        @model function constrained_test(obs)
            p ~ Beta(2,2)
            for i = 1:length(obs)
                obs[i] ~ Bernoulli(p)
            end
            p
        end

        chain = sample(
            rng,
            constrained_test(obs),
            HMC(1.5, 3),# using a large step size (1.5)
            1000)

        check_numerical(chain, [:p], [10/14], atol=0.1)
    end
    @numerical_testset "constrained simplex" begin
        obs12 = [1,2,1,2,2,2,2,2,2,2]

        @model function constrained_simplex_test(obs12)
            ps ~ Dirichlet(2, 3)
            pd ~ Dirichlet(4, 1)
            for i = 1:length(obs12)
                obs12[i] ~ Categorical(ps)
            end
            return ps
        end

        chain = sample(
            rng,
            constrained_simplex_test(obs12),
            HMC(0.75, 2),
            1000)

        check_numerical(chain, ["ps[1]", "ps[2]"], [5/16, 11/16], atol=0.015)
    end
    @numerical_testset "hmc reverse diff" begin
        alg = HMC(0.1, 10)
        res = sample(rng, gdemo_default, alg, 4000)
        check_gdemo(res, rtol=0.1)
    end
    @turing_testset "matrix support" begin
        @model function hmcmatrixsup()
            v ~ Wishart(7, [1 0.5; 0.5 1])
        end

        model_f = hmcmatrixsup()
        n_samples = 1_000
        vs = map(1:3) do _
            chain = sample(rng, model_f, HMC(0.15, 7), n_samples)
            r = reshape(Array(group(chain, :v)), n_samples, 2, 2)
            reshape(mean(r; dims = 1), 2, 2)
        end

        @test maximum(abs, mean(vs) - (7 * [1 0.5; 0.5 1])) <= 0.5
    end
    @turing_testset "multivariate support" begin
        # Define NN flow
        function nn(x, b1, w11, w12, w13, bo, wo)
            h = tanh.([w11 w12 w13]' * x .+ b1)
            return logistic(dot(wo, h) + bo)
        end

        # Generating training data
        N = 20
        M = N ÷ 4
        x1s = rand(M) * 5
        x2s = rand(M) * 5
        xt1s = Array([[x1s[i]; x2s[i]] for i = 1:M])
        append!(xt1s, Array([[x1s[i] - 6; x2s[i] - 6] for i = 1:M]))
        xt0s = Array([[x1s[i]; x2s[i] - 6] for i = 1:M])
        append!(xt0s, Array([[x1s[i] - 6; x2s[i]] for i = 1:M]))

        xs = [xt1s; xt0s]
        ts = [ones(M); ones(M); zeros(M); zeros(M)]

        # Define model

        alpha = 0.16                  # regularizatin term
        var_prior = sqrt(1.0 / alpha) # variance of the Gaussian prior

        @model function bnn(ts)
            b1 ~ MvNormal([0. ;0.; 0.],
                [var_prior 0. 0.; 0. var_prior 0.; 0. 0. var_prior])
            w11 ~ MvNormal([0.; 0.], [var_prior 0.; 0. var_prior])
            w12 ~ MvNormal([0.; 0.], [var_prior 0.; 0. var_prior])
            w13 ~ MvNormal([0.; 0.], [var_prior 0.; 0. var_prior])
            bo ~ Normal(0, var_prior)

            wo ~ MvNormal([0.; 0; 0],
                [var_prior 0. 0.; 0. var_prior 0.; 0. 0. var_prior])
            for i = rand(1:N, 10)
                y = nn(xs[i], b1, w11, w12, w13, bo, wo)
                ts[i] ~ Bernoulli(y)
            end
            b1, w11, w12, w13, bo, wo
        end

        # Sampling
        chain = sample(rng, bnn(ts), HMC(0.1, 5), 10)
    end

    @numerical_testset "hmcda inference" begin 
        alg1 = HMCDA(500, 0.8, 0.015)
        # alg2 = Gibbs(HMCDA(200, 0.8, 0.35, :m), HMC(0.25, 3, :s))
        
        # alg3 = Gibbs(HMC(0.25, 3, :m), PG(30, 3, :s))
        # alg3 = PG(50, 2000)

        res1 = sample(rng, gdemo_default, alg1, 3000)
        check_gdemo(res1)

        # res2 = sample(gdemo([1.5, 2.0]), alg2)
        #
        # @test mean(res2[:s]) ≈ 49/24 atol=0.2
        # @test mean(res2[:m]) ≈ 7/6 atol=0.2
    end

    @numerical_testset "hmcda+gibbs inference" begin 
        rng = StableRNG(123)
        Random.seed!(12345) # particle samplers do not support user-provided `rng` yet
        alg3 = Gibbs(PG(20, :s), HMCDA(500, 0.8, 0.25, init_ϵ = 0.05, :m))

        res3 = sample(rng, gdemo_default, alg3, 3000, discard_initial=1000)
        check_gdemo(res3)
    end

    @turing_testset "hmcda constructor" begin
        alg = HMCDA(0.8, 0.75)
        println(alg)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        alg = HMCDA(200, 0.8, 0.75)
        println(alg)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        alg = HMCDA(200, 0.8, 0.75, :s)
        println(alg)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "HMCDA"

        @test isa(alg, HMCDA)
        @test isa(sampler, Sampler{<:Turing.Hamiltonian})
    end
    @numerical_testset "nuts inference" begin
        alg = NUTS(1000, 0.8)
        res = sample(rng, gdemo_default, alg, 6000)
        check_gdemo(res)
    end
    @turing_testset "nuts constructor" begin
        alg = NUTS(200, 0.65)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"

        alg = NUTS(0.65)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"

        alg = NUTS(200, 0.65, :m)
        sampler = Sampler(alg, gdemo_default)
        @test DynamicPPL.alg_str(sampler) == "NUTS"
    end
    @turing_testset "check discard" begin
        alg = NUTS(100, 0.8)

        c1 = sample(rng, gdemo_default, alg, 500, discard_adapt = true)
        c2 = sample(rng, gdemo_default, alg, 500, discard_adapt = false)

        @test size(c1, 1) == 500
        @test size(c2, 1) == 500
    end
    @turing_testset "AHMC resize" begin
        alg1 = Gibbs(PG(10, :m), NUTS(100, 0.65, :s))
        alg2 = Gibbs(PG(10, :m), HMC(0.1, 3, :s))
        alg3 = Gibbs(PG(10, :m), HMCDA(100, 0.65, 0.3, :s))
        @test sample(rng, gdemo_default, alg1, 300) isa Chains
        @test sample(rng, gdemo_default, alg2, 300) isa Chains
        @test sample(rng, gdemo_default, alg3, 300) isa Chains
    end

    @turing_testset "Regression tests" begin
        # https://github.com/TuringLang/DynamicPPL.jl/issues/27
        @model function mwe1(::Type{T}=Float64) where {T<:Real}
            m = Matrix{T}(undef, 2, 3)
            m .~ MvNormal(zeros(2), I)
        end
        @test sample(rng, mwe1(), HMC(0.2, 4), 1_000) isa Chains

        @model function mwe2(::Type{T} = Matrix{Float64}) where T
            m = T(undef, 2, 3)
            m .~ MvNormal(zeros(2), I)
        end
        @test sample(rng, mwe2(), HMC(0.2, 4), 1_000) isa Chains

        # https://github.com/TuringLang/Turing.jl/issues/1308
        @model function mwe3(::Type{T} = Array{Float64}) where T
            m = T(undef, 2, 3)
            m .~ MvNormal(zeros(2), I)
        end
        @test sample(rng, mwe3(), HMC(0.2, 4), 1_000) isa Chains
    end

    # issue #1923
    @turing_testset "reproducibility" begin
        alg = NUTS(1000, 0.8)
        res1 = sample(StableRNG(123), gdemo_default, alg, 1000)
        res2 = sample(StableRNG(123), gdemo_default, alg, 1000)
        res3 = sample(StableRNG(123), gdemo_default, alg, 1000)
        @test Array(res1) == Array(res2) == Array(res3)
    end

    @turing_testset "prior" begin
        @model function demo_hmc_prior()
            # NOTE: Used to use `InverseGamma(2, 3)` but this has infinite variance
            # which means that it's _very_ difficult to find a good tolerance in the test below:)
            s ~ truncated(Normal(3, 1), lower=0)
            m ~ Normal(0, sqrt(s))
        end
        alg = NUTS(1000, 0.8)
        gdemo_default_prior = DynamicPPL.contextualize(demo_hmc_prior(), DynamicPPL.PriorContext())
        chain = sample(gdemo_default_prior, alg, 10_000)
        check_numerical(chain, [:s, :m], [mean(truncated(Normal(3, 1); lower=0)), 0], atol=0.1)
    end

    @turing_testset "warning for difficult init params" begin
        attempt = 0
        @model function demo_warn_init_params()
            x ~ Normal()
            if (attempt += 1) < 30
                Turing.@addlogprob! -Inf
            end
        end

        @test_logs (
            :warn,
            "failed to find valid initial parameters in 10 tries; consider providing explicit initial parameters using the `init_params` keyword",
        ) (:info,) match_mode=:any begin
            sample(demo_warn_init_params(), NUTS(), 5)
        end
    end
end
