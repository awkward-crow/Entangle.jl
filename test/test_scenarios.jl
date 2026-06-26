# ---- Helpers -----------------------------------------------------------------

# Two-node model: one scenario-tagged node (:aws) and one idiosyncratic node.
#
# aws node:  E[rate] = 2.0 × 0.2 = 0.4,  E[loss] = 1_000 + 200 = 1_200
#            E[annual] = 0.4 × 1_200 = 480
# reg node:  E[rate] = 1.0 × 0.05 = 0.05, E[loss] = 500 + 100 = 600
#            E[annual] = 0.05 × 600 = 30
#
# E[total | p_hit] = 30 + p_hit × 480
function _scenario_model(name::Symbol)
    freq_aws = FrequencyModel(Gamma(4.0, 0.5), Beta(1.0, 4.0))
    mag_aws  = MagnitudeModel(Exponential(1_000.0), Exponential(200.0))
    freq_reg = FrequencyModel(Gamma(1.0, 1.0), Beta(1.0, 19.0))
    mag_reg  = MagnitudeModel(Exponential(500.0), Exponential(100.0))
    aws_node = FAIRNode(freq_aws, mag_aws; name = :aws_outage, factor = :aws)
    reg_node = FAIRNode(freq_reg, mag_reg; name = :regulatory, factor = nothing)
    FAIRModel(name, [aws_node, reg_node])
end

# ---- FAIRNode name and factor ------------------------------------------------

@testset "FAIRNode name and factor" begin

    freq = FrequencyModel(Gamma(1.0, 1.0), Beta(1.0, 1.0))
    mag  = MagnitudeModel(Exponential(1.0), Exponential(1.0))

    @testset "defaults" begin
        node = FAIRNode(freq, mag)
        @test node.name   == :unnamed
        @test node.factor === nothing
    end

    @testset "explicit name and factor" begin
        node = FAIRNode(freq, mag; name = :aws_outage, factor = :aws)
        @test node.name   == :aws_outage
        @test node.factor == :aws
    end

    @testset "factor = nothing explicit" begin
        node = FAIRNode(freq, mag; name = :regulatory, factor = nothing)
        @test node.factor === nothing
    end

    @testset "existing fields unaffected" begin
        node = FAIRNode(freq, mag; name = :x, factor = :y)
        @test node.frequency === freq
        @test node.magnitude === mag
    end

end

# ---- Exposures ---------------------------------------------------------------

@testset "Exposures" begin

    @testset "construction and getindex" begin
        e = Exposures(aws = 0.7, ransomware = 0.4)
        @test e[:aws]       ≈ 0.7
        @test e[:ransomware] ≈ 0.4
        @test e[:crowdstrike] == 0.0
    end

    @testset "isempty" begin
        @test  isempty(Exposures())
        @test !isempty(Exposures(aws = 0.5))
    end

    @testset "names" begin
        e = Exposures(aws = 0.7, ransomware = 0.4)
        @test Set(names(e)) == Set([:aws, :ransomware])
    end

    @testset "validates range" begin
        @test_throws ArgumentError Exposures(aws = -0.1)
        @test_throws ArgumentError Exposures(aws =  1.1)
    end

    @testset "accepts boundary values 0 and 1" begin
        e = Exposures(aws = 0.0, ransomware = 1.0)
        @test e[:aws]       == 0.0
        @test e[:ransomware] == 1.0
    end

end

# ---- PortfolioExposures -------------------------------------------------------

@testset "PortfolioExposures" begin

    @testset "getindex returns 0 for absent org" begin
        se = PortfolioExposures()
        @test se[:acme, :aws] == 0.0
    end

    @testset "getindex returns 0 for absent scenario" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7)
        @test se[:acme, :ransomware] == 0.0
    end

    @testset "insert! adds entries" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7, ransomware = 0.4)
        @test se[:acme, :aws]       ≈ 0.7
        @test se[:acme, :ransomware] ≈ 0.4
    end

    @testset "insert! returns exposures" begin
        se = PortfolioExposures()
        @test insert!(se, :acme; aws = 0.5) === se
    end

    @testset "insert! throws on duplicate org/scenario" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7)
        @test_throws ArgumentError insert!(se, :acme; aws = 0.9)
    end

    @testset "insert! allows new scenario for existing org" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7)
        insert!(se, :acme; ransomware = 0.4)
        @test se[:acme, :aws]       ≈ 0.7
        @test se[:acme, :ransomware] ≈ 0.4
    end

    @testset "insert! is atomic: no partial write on duplicate" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7)
        @test_throws ArgumentError insert!(se, :acme; ransomware = 0.4, aws = 0.9)
        @test se[:acme, :ransomware] == 0.0
    end

    @testset "insert! validates range" begin
        se = PortfolioExposures()
        @test_throws ArgumentError insert!(se, :acme; aws = -0.1)
        @test_throws ArgumentError insert!(se, :acme; aws =  1.1)
    end

    @testset "insert! accepts boundary values 0 and 1" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.0, ransomware = 1.0)
        @test se[:acme, :aws]       == 0.0
        @test se[:acme, :ransomware] == 1.0
    end

    @testset "update! overwrites and returns old values" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7, ransomware = 0.4)
        (org, old) = update!(se, :acme; aws = 0.9, ransomware = 0.2)
        @test se[:acme, :aws]        ≈ 0.9
        @test se[:acme, :ransomware] ≈ 0.2
        @test org                    == :acme
        @test old[:aws]              ≈ 0.7
        @test old[:ransomware]       ≈ 0.4
    end

    @testset "update! throws if scenario not set" begin
        se = PortfolioExposures()
        @test_throws ArgumentError update!(se, :acme; aws = 0.5)
    end

    @testset "update! throws if org absent" begin
        se = PortfolioExposures()
        @test_throws ArgumentError update!(se, :acme; aws = 0.5)
    end

    @testset "update! is atomic: no partial write on missing scenario" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7)
        @test_throws ArgumentError update!(se, :acme; aws = 0.9, ransomware = 0.5)
        @test se[:acme, :aws] ≈ 0.7
    end

    @testset "update! validates range" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7)
        @test_throws ArgumentError update!(se, :acme; aws = -0.1)
        @test_throws ArgumentError update!(se, :acme; aws =  1.1)
    end

    @testset "insert! accepts Exposures struct" begin
        se = PortfolioExposures()
        insert!(se, :acme, Exposures(aws = 0.7, ransomware = 0.4))
        @test se[:acme, :aws]        ≈ 0.7
        @test se[:acme, :ransomware] ≈ 0.4
    end

    @testset "update! accepts Exposures struct and returns Exposures" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7, ransomware = 0.4)
        (org, old) = update!(se, :acme, Exposures(aws = 0.9, ransomware = 0.2))
        @test org                    == :acme
        @test se[:acme, :aws]        ≈ 0.9
        @test se[:acme, :ransomware] ≈ 0.2
        @test old isa Exposures
        @test old[:aws]              ≈ 0.7
        @test old[:ransomware]       ≈ 0.4
    end

    @testset "update! round-trip leaves se unchanged" begin
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.7, ransomware = 0.4)
        update!(se, update!(se, :acme, Exposures(aws = 0.9, ransomware = 0.2))...)
        @test se[:acme, :aws]        ≈ 0.7
        @test se[:acme, :ransomware] ≈ 0.4
    end

end

# ---- rand_scenario_losses ----------------------------------------------------

@testset "rand_scenario_losses" begin

    function _portfolio_and_exposures(; p_hit)
        p  = Portfolio()
        insert!(p, _scenario_model(:acme))
        se = PortfolioExposures()
        insert!(se, :acme; aws = p_hit)
        return p, se
    end

    @testset "returns correct length" begin
        p, se = _portfolio_and_exposures(p_hit = 0.5)
        losses = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 200)
        @test length(losses) == 200
    end

    @testset "non-negative" begin
        p, se = _portfolio_and_exposures(p_hit = 0.5)
        losses = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 500)
        @test all(>=(0.0), losses)
    end

    @testset "reproducible" begin
        p, se = _portfolio_and_exposures(p_hit = 0.5)
        l1 = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 200)
        l2 = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 200)
        @test l1 == l2
    end

    @testset "insertion order independence" begin
        p1 = Portfolio()
        insert!(p1, _scenario_model(:acme))
        insert!(p1, _scenario_model(:globex))

        p2 = Portfolio()
        insert!(p2, _scenario_model(:globex))
        insert!(p2, _scenario_model(:acme))

        se = PortfolioExposures()
        insert!(se, :acme;   aws = 0.6)
        insert!(se, :globex; aws = 0.4)

        l1 = rand_scenario_losses(TEST_SEED, :aws, se, p1; n_samples = 300)
        l2 = rand_scenario_losses(TEST_SEED, :aws, se, p2; n_samples = 300)
        @test l1 == l2
    end

    # E[total | p_hit = 0]   = 30   (only :regulatory fires)
    # E[total | p_hit = 1]   = 510  (both nodes always fire)
    # E[total | p_hit = 0.5] = 270

    @testset "zero exposure: only baseline fires" begin
        p, se = _portfolio_and_exposures(p_hit = 0.0)
        losses = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 100_000)
        @test mean(losses) ≈ 30.0  atol=8.0
    end

    @testset "full exposure: scenario node always fires" begin
        p, se = _portfolio_and_exposures(p_hit = 1.0)
        losses = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 100_000)
        @test mean(losses) ≈ 510.0  atol=50.0
    end

    @testset "partial exposure: mean scales linearly with p_hit" begin
        p, se = _portfolio_and_exposures(p_hit = 0.5)
        losses = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 100_000)
        @test mean(losses) ≈ 270.0  atol=30.0
    end

    @testset "absent org treated as unexposed" begin
        p  = Portfolio()
        insert!(p, _scenario_model(:acme))
        se = PortfolioExposures()
        # :acme not in exposures → p_hit = 0 → only baseline
        losses = rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 100_000)
        @test mean(losses) ≈ 30.0  atol=8.0
    end

    @testset "unmatched scenario leaves all nodes at baseline" begin
        p, se = _portfolio_and_exposures(p_hit = 1.0)
        # :ransomware scenario — no nodes tagged :ransomware, so all nodes fire at baseline
        losses = rand_scenario_losses(TEST_SEED, :ransomware, se, p; n_samples = 100_000)
        @test mean(losses) ≈ 510.0  atol=50.0
    end

    @testset "does not require pre-computed marginal samples" begin
        p  = Portfolio()
        insert!(p, _scenario_model(:acme))
        se = PortfolioExposures()
        insert!(se, :acme; aws = 0.5)
        @test !has_loss_samples(p, :acme)
        @test_nowarn rand_scenario_losses(TEST_SEED, :aws, se, p; n_samples = 100)
    end

end
