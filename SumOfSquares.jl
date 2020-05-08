using SumOfSquares
using DynamicPolynomials
using MosekTools
using DifferentialEquations
using LinearAlgebra

# System Dynamics
A = 1.0
B = 1.0
Z(x) = x[1]^2
dZdX(x) = 2*x[1]
N = 1
O = 1
ϵ = 1E-6

@inline function f(x, u, t)
    return A*Z(x) + B*u(x,t)
end

@inline function f(dx, x, u, t)
    dx[1] = f(x, u, t)
end

# Solve ODE
Tmax = 1.0
τ = 0.1
u(x,t) = -sin(t)
prob = ODEProblem{true}(f, [0.5-rand()], [0.0, Tmax], u)
sol = DifferentialEquations.solve(prob, Tsit5(), saveat=τ, dense=false, save_end=false, dtmin=1E-6)

# Generate data from ODE solution
T = length(sol.t)
X0T = reshape(sol.u, (N, T))
U01T = reshape(u.(sol.u, sol.t), (N, T))
X1T = reshape(f.(sol.u, u, sol.t), (N, T))
Z0T = reshape(Z.(sol.u), (N, T))

# Create a Sum of Squares JuMP model with the Mosek solver
model = SOSModel(Mosek.Optimizer)
@variable(model, P[1:N,1:N], Symmetric)
@variable(model, Y0[1:T,1:N])

@polyvar x[1:N]
X = monomials(x, 0:O)
@variable(model, Y1[1:T, 1:N], Poly(X))

Y = Y0+Y1

@constraint(model, Z0T*Y0 .== P)
@constraint(model, Z0T*Y1 .== 0.0)
if size(P)[1] > 1
    @SDconstraint(model, P >= ϵ*I)
else
    @constraint(model, P[1,1] >= ϵ*1.0)
end

@variable(model, γ)
@objective(model, Max, γ)

Q = -(dZdX(x)*X1T*Y+transpose(dZdX(x)*X1T*Y))
if size(Q)[1] > 1
    @SDconstraint(model, Q >= γ)
else
    @constraint(model, Q[1,1] >= γ)
end
@constraint(model, γ >= 0)
optimize!(model)
display(termination_status(model))
display(value.(P))
display(value.(Y))
# display(objective_value(model))

F = U01T*value.(Y)*inv(Z0T*value.(Y))
