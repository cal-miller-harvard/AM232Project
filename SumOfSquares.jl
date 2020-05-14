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
C2(x) = [-1. x[1]]
C1_dim = 2
C2_dim = 1
N = 2
n_vars = 2
O = 3
ϵ = 1E-4
w(x,t) = zero(x)

@inline function xdot(x, u, t)
    return A(x)*Z(x) + B(x)*u(x, t) + w(x,t)
end

# Solve ODE
Tmax = 10
τ = 1
x0 = [0.5, 0.5]
u(x,t) = -sin(t)
times = [i for i in 0:τ:Tmax]


function solve_discrete(x0, times, u)
    states = zeros((length(x0), length(times)))
    for (i, t) in enumerate(times)
        if i == 1
            states[:,i] .= x0
        else
            states[:,i] .= xdot(states[:,i-1], u, t)
        end
    end
    return states
end

states = solve_discrete(x0, times, u)

# Generate data matrices from ODE solution
T = length(times)
X0T = states
U01T = hcat([u(states[:, i], times[i]) for i=1:size(states, 2)]...)
X1T = hcat([xdot(states[:, i], u, times[i]) for i=1:size(states, 2)]...)
Z0T = hcat([Z(states[:, i]) for i=1:size(states, 2)]...)

# Create a Sum of Squares JuMP model with the Mosek solver
model = SOSModel(Mosek.Optimizer)
@variable(model, Y0[1:T,1:N])

@polyvar x[1:n_vars]
X = monomials(x, 0:O)
@variable(model, Y1[1:T, 1:N], Poly(X))


@constraint(model, Z0T*Y1 .== 0.0)
@SDconstraint(model, Z0T*Y0 >= ϵ*I)


I_tot = Matrix{Float64}(I(C1_dim + C2_dim))
I1 = I_tot[:, 1:C1_dim]
I2 = I_tot[:, C1_dim+1:C1_dim+C2_dim]

ϵ2(y) = ϵ*y[1]^4 + ϵ*y[2]^4
Q_11 = dZdX(x)*X1T*(Y0+Y1)+transpose(dZdX(x)*X1T*(Y0+Y1)) + ϵ2(x)*I(N)
Q_21 = ((I1*C1(x) + I2*C2(x))*Z0T + I2*U01T)*(Y0+Y1)
Q_12 = transpose(Q_21)
Q_22 = -I(size(Q_21, 1))*(1-ϵ2(x))
Q_aug = Matrix([Q_11 Q_12; Q_21 Q_22])

@polyvar v[1:size(Q_aug, 1)]
@constraint(model, -transpose(v)*Q_aug*v >= 0)
optimize!(model)
display(termination_status(model))
# display(objective_value(model))

F = U01T*value.(Y0+Y1)*inv(Z0T*value.(Y0))

# solve ODE with new controller
x0 = [0.5, 0.5]
new_u(x, t) = [F[i](x...) for i=1:N]'*Z(x)
states = solve_discrete(x0, times, new_u)
# plot(times, states')

# Bootstrap

δ = 0.1
M = 50
σw = 0.1
σu = 0.1
times = [i for i in 0:10]

T = length(times)
xt = [Matrix{Float64}(undef, length(x0), T) for i in 1:M]
ut = [Matrix{Float64}(undef, length(u(x0, 0)), T) for i in 1:M]

for i in 1:M
    ut[i] .= randn(size(ut[i]))
    itr = Iterators.Stateful(ut[i])
    xt[i] .= solve_discrete(randn(size(x0)), times, (x,t) -> popfirst!(itr))
end

Ahat, Bhat = estimateAB(xt, ut)

println(bootstrap(δ, M, Ahat, Bhat, σw, σu, xt, ut))