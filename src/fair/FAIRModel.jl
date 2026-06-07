"""
    FAIRModel(name, nodes)
    FAIRModel(name, node)

Single-organisation FAIR model: a named collection of `FAIRNode`s whose annual
losses are summed to give total annual loss.

    S_org = ∑_k S_k,   S_k ~ compound Poisson (node k)
"""
struct FAIRModel
    name::String
    nodes::Vector{FAIRNode}
end

FAIRModel(name::String, node::FAIRNode) = FAIRModel(name, [node])

# ---- Sampling ----------------------------------------------------------------

function rand_annual_loss(rng::AbstractRNG, model::FAIRModel)
    return sum(node -> rand_annual_loss(rng, node), model.nodes; init=0.0)
end

rand_annual_loss(model::FAIRModel) = rand_annual_loss(Random.default_rng(), model)

# ---- Moments -----------------------------------------------------------------

mean_annual_loss(model::FAIRModel) =
    sum(mean_annual_loss, model.nodes; init=0.0)
