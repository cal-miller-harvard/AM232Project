using SumOfSquares
using DynamicPolynomials
using MosekTools
using DifferentialEquations
using LinearAlgebra
using Plots

include("Bootstrap.jl")

# System Dynamics
A(x) = Matrix{Float64}(I(2))
B(x) = [0.; 1.]
Z(x) = [x[2]; x[1]^2]
dZdX(x) = [0. 1.; 2x[1] 0.]
C1(x) = [x[2] -x[1]; 0 -x[2]]
C2(x) = [-x[1]^2 2.]
C1_dim = 2
C2_dim = 1
N = 2
n_vars = 2
O = 3
ϵ = 1E-4
w(x,t) = zero(x)

@inline function xdot(x, u, t, w)
    return A(x)*Z(x) + B(x)*u(x, t) + w(x,t)
end

# Solve ODE
Tmax = 10
τ = 1
x0 = [0.5, 0.5]
u(x,t) = -sin(t)
times = [i for i in 0:τ:Tmax]


function solve_discrete(x0, times, u, w)
    states = zeros((length(x0), length(times)))
    for (i, t) in enumerate(times)
        if i == 1
            states[:,i] .= x0
        else
            states[:,i] .= xdot(states[:,i-1], u, t, w)
        end
    end
    return states
end

states = solve_discrete(x0, times, u, w)

# Generate data matrices from ODE solution
T = length(times)
X0T = states
U01T = hcat([u(states[:, i], times[i]) for i=1:size(states, 2)]...)
X1T = hcat([xdot(states[:, i], u, times[i], w) for i=1:size(states, 2)]...)
Z0T = hcat([Z(states[:, i]) for i=1:size(states, 2)]...)
UZ = vcat(U01T, Z0T)


# Create a Sum of Squares JuMP model with the Mosek solver
model = SOSModel(Mosek.Optimizer)
@variable(model, Y0[1:T,1:N])

@polyvar x[1:n_vars]
X = monomials(x, 0:O)
@variable(model, Y1[1:T, 1:N], Poly(X))


@constraint(model, Z0T*Y1 .== 0.0)
# <<<<<<< HEAD

# # if size(Z0T*Y0, 1) > 1
# #     @SDconstraint(model, Z0T*Y0 >= ϵ*I)
# # else
# @constraint(model, Z0T*Y0 ∈ PSDCone())
# # end
# =======
# @SDconstraint(model, Z0T*Y0 >= ϵ*I)
# >>>>>>> 2edf902927a0f72f984dae3ffb115f32eb92660d


I_tot = Matrix{Float64}(I(C1_dim + C2_dim))
I1 = I_tot[:, 1:C1_dim]
I2 = I_tot[:, C1_dim+1:C1_dim+C2_dim]

# <<<<<<< HEAD
# ϵ2(y) = ϵ*y[1]^4 + ϵ*y[2]^4 # 0.001
# # Q = dZdX(x)*X1T*(Y0+Y1)+transpose(dZdX(x)*X1T*(Y0+Y1))
# =======
# ϵ2(y) = ϵ*y[1]^4 + ϵ*y[2]^4
# >>>>>>> 2edf902927a0f72f984dae3ffb115f32eb92660d
Q_11 = dZdX(x)*X1T*(Y0+Y1)+transpose(dZdX(x)*X1T*(Y0+Y1)) + ϵ2(x)*I(N)
Q_21 = ((I1*C1(x) + I2*C2(x))*Z0T + I2*U01T)*(Y0+Y1)
Q_12 = transpose(Q_21)
Q_22 = -I(size(Q_21, 1))#*(1-ϵ2(x))
Q_aug = Matrix([Q_11 Q_12; Q_21 Q_22])
#=
@polyvar v[1:size(Q_aug, 1)]
# <<<<<<< HEAD

# @constraint(model, -transpose(v)*Q_aug*v ∈ SOSCone())

# @variable(model, W[1:size(Z0T*Y0, 1),1:size(Z0T*Y0, 1)])
# trace_mat = Matrix([W I(size(W, 1)); I(size(W, 1)) Z0T*Y0])
# @polyvar v2[1:size(trace_mat, 1)]
# @constraint(model, transpose(v2)*trace_mat*v2 ∈ SOSCone())

# @objective(model, Min, W[1, 1]+W[2, 2])


# =======
# @constraint(model, -transpose(v)*Q_aug*v >= 0)
# >>>>>>> 2edf902927a0f72f984dae3ffb115f32eb92660d
optimize!(model)
display(termination_status(model))
# display(objective_value(model))

F = U01T*value.(Y0+Y1)*inv(Z0T*value.(Y0))

# solve ODE with new controller
x0 = [0.5, 0.5]
new_u(x, t) = [F[i](x...) for i=1:N]'*Z(x)
states = solve_discrete(x0, times, new_u)
plot(times, states')
=#

# Bootstrap

δ = 0.1
M = 20
Mbootstrap = 500
σw = 1E-1
σu = 2E-1
dw = Distributions.MvNormal(length(x0), σw)
w(x,t) = rand(dw)
times = [i for i in 0:10]

T = length(times)
xt = [Matrix{Float64}(undef, length(x0), T) for i in 1:M]
ut = [Matrix{Float64}(undef, length(u(x0, 0)), T) for i in 1:M]

for i in 1:M
    ut[i] .= σu*randn(size(ut[i]))
    itr = Iterators.Stateful(ut[i])
    xt[i] .= solve_discrete(zero(x0), times, (x,t) -> popfirst!(itr), w)
end

Ahat, Bhat = estimateAB(xt, ut, Z)
println("Ahat:")
display(Ahat)
println("Bhat:")
display(Bhat)

println("A error:")
display(norm(Ahat - A(x0)))
println("B error:")
display(norm(Bhat - B(x0)))

ϵA, ϵB = bootstrap(δ, Mbootstrap, Ahat, Bhat, σw, σu, xt, ut, Z)

println("ϵA:")
display(ϵA)
println("ϵB:")
display(ϵB)
