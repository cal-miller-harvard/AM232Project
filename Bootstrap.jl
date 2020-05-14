using Distributions
using LinearAlgebra
using Statistics

function estimateAB(xt, ut, Z)
    d = size(xt[1])[1]
    dZ = length(Z(xt[1][:,1]))
    du = length(u(xt[1][:,1],0))

    Xt = zeros(M*(T-1), d)
    Zt = zeros(M*(T-1), dZ+du)
    for i in 1:M
        Xt[(T-1)*(i-1)+1:(T-1)*i, :] .= transpose(xt[i][:,2:end])
        for j in 1:T-1
            Zt[(T-1)*(i-1)+j, 1:d] .= Z(xt[i][:,j])
            Zt[(T-1)*(i-1)+j, d+1:end] .= ut[i][:,j]
        end
    end
    Θ = pinv(Zt, rtol=sqrt(eps(Float64)))*Xt
    Ahat = transpose(Θ[1:d,:])
    Bhat = transpose(Θ[d+1:end,:])
    return (Ahat, Bhat)
end

function bootstrap(δ, M, Ahat, Bhat, σw, σu, xt, ut, Z)
    ϵAs = zeros(M)
    ϵBs = zeros(M)
    N = size(xt[1])[1]
    Nz = size(Z(xt[1][:,1]))[1]
    T = size(xt[1])[2]
    xhat = deepcopy(xt)

    u = Distributions.MvNormal(size(ut[1])[1], σu)
    w = Distributions.MvNormal(size(xt[1])[1], σw)

    for i in 1:M
        us = [rand(u, T) for i in 1:M]
        for l in 1:N
            xhat[l][:, 1] .= xt[l][:,1]
            for t in 1:T-1
                xhat[l][:, t+1] .= Ahat * Z(xhat[l][:, t]) + Bhat * us[l][:, t] + rand(w)
            end
        end
        Atilde, Btilde = estimateAB(xhat, us, Z)
        ϵAs[i] = norm(Ahat - Atilde)
        ϵBs[i] = norm(Bhat - Btilde)
    end
    return (quantile(ϵAs,[δ,1-δ]), quantile(ϵBs,[δ,1-δ]))
end
