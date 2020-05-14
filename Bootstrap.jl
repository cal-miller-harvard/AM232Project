using Distributions
using LinearAlgebra
using Statistics
using JuMP
using MosekTools

function estimateAB(xt, ut)
    model = Model(Mosek.Optimizer)
    @variable(model, Atilde[1:size(xt[1])[1], 1:size(xt[1])[1]])
    @variable(model, Btilde[1:size(xt[1])[1], 1:size(ut[1])[1]])
    obj = 0
    N = size(xt)[1]
    T = size(xt[1])[2]
    for l in 1:N
        for t in 1:T-1
            obj += sum((Atilde*xt[l][:, t] + Btilde*ut[l][:, t] - xt[l][:, t+1]).^2)
        end
    end
    @objective(model, Min, obj)
    optimize!(model)
    return (value.(Atilde), value.(Btilde))
end

function bootstrap(δ, M, Ahat, Bhat, σw, σu, xt, ut)
    ϵAs = zeros(M)
    ϵBs = zeros(M)
    N = length(xt)
    T = size(xt[1])[2]
    xhat = deepcopy(xt)

    u = Distributions.MvNormal(size(ut[1])[1], σu)
    w = Distributions.MvNormal(size(xt[1])[1], σw)

    for i in 1:M
        us = [rand(u, T) for i in 1:N]
        for l in 1:N
            xhat[l][:, 1] .= xt[l][:,1]
            for t in 1:T-1
                xhat[l][:, t+1] .= Ahat * xhat[l][:, t] + Bhat * us[l][:, t] + rand(w)
            end
        end
        model = Model(Mosek.Optimizer)
        @variable(model, Atilde[1:size(Ahat)[1], 1:size(Ahat)[2]])
        @variable(model, Btilde[1:size(Bhat)[1], 1:size(Bhat)[2]])
        @expression(model, obj, 0)
        for l in 1:N
            for t in 1:T
                obj += sum((Atilde*xhat[l][:, t] + Btilde*us[l][:, t] - xhat[l][:, t+1]).^2)
            end
        end
        @objective(model, Min, obj)
        optimize!(model)

        ϵAs[i] = norm(Ahat - value.(Atilde))
        ϵBs[i] = norm(Bhat - value.(Btilde))
    end
    return (quantile(ϵAs,[δ,1-δ]), quantile(ϵBs,[δ,1-δ]))
end
