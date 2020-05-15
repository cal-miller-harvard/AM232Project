using SumOfSquares
using DynamicPolynomials
using MosekTools
using DifferentialEquations
using LinearAlgebra
using Plots

pyplot()

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

@inline function xdot(x, u, t)
    return A(x)*Z(x) + B(x)*u(x, t)
end

@inline function f!(dx, x, u, t)
    dx .= A(x)*Z(x) + B(x)*u(x, t)
end

# Solve ODE
Tmax = 1.0
τ = 0.25
x0 = [0.5, 0.5]
u(x,t) = -sin(t)
prob = ODEProblem(f!, x0, (0.0, Tmax), u)
sol = DifferentialEquations.solve(prob, AutoTsit5(Rosenbrock23()), saveat=τ, dense=false, save_end=false, dtmin=1E-6)

times = sol.t
states = hcat(sol.u...)
# Generate data matrices from ODE solution
T = length(times)
X0T = states
U01T = hcat([u(states[:, i], times[i]) for i=1:size(states, 2)]...)
X1T = hcat([xdot(states[:, i], u, times[i]) for i=1:size(states, 2)]...)
Z0T = hcat([Z(states[:, i]) for i=1:size(states, 2)]...)
function make_controller(opt)
    # Create a Sum of Squares JuMP model with the Mosek solver
    model = SOSModel(Mosek.Optimizer)
    @variable(model, Y0[1:T,1:N])

    @polyvar x[1:n_vars]
    X = monomials(x, 0:O)
    @variable(model, Y1[1:T, 1:N], Poly(X))


    @constraint(model, Z0T*Y1 .== 0.0)

    # if size(Z0T*Y0, 1) > 1
    #     @SDconstraint(model, Z0T*Y0 >= ϵ*I)
    # else
    @SDconstraint(model, Z0T*Y0 >= ϵ*I)
    # end


    I_tot = Matrix{Float64}(I(C1_dim + C2_dim))
    I1 = I_tot[:, 1:C1_dim]
    I2 = I_tot[:, C1_dim+1:C1_dim+C2_dim]

    ϵ2(y) = ϵ*y[1]^4 + ϵ*y[2]^4
    # Q = dZdX(x)*X1T*(Y0+Y1)+transpose(dZdX(x)*X1T*(Y0+Y1))
    Q_11 = dZdX(x)*X1T*(Y0+Y1)+transpose(dZdX(x)*X1T*(Y0+Y1)) + ϵ2(x)*I(N)
    Q_21 = ((I1*C1(x) + I2*C2(x))*Z0T + I2*U01T)*(Y0+Y1)
    Q_12 = transpose(Q_21)
    Q_22 = -I(size(Q_21, 1))*(1-ϵ2(x))
    Q_aug = Matrix([Q_11 Q_12; Q_21 Q_22])

    @polyvar v[1:size(Q_aug, 1)]

    @constraint(model, -transpose(v)*Q_aug*v >= 0)

    ### COMMENT OUT THESE FIVE LINES TO REMOVE OPTIMAL CONTROL OBJECTIVE
    # @variable(model, W[1:size(Z0T*Y0, 1),1:size(Z0T*Y0, 1)])
    # trace_mat = Matrix([W I(size(W, 1)); I(size(W, 1)) Z0T*Y0])
    # @polyvar v2[1:size(trace_mat, 1)]
    # @constraint(model, transpose(v2)*trace_mat*v2 >= 0)
    # @objective(model, Min, W[1, 1]+W[2, 2])
    #####


    optimize!(model)
    display(termination_status(model))
    # display(objective_value(model))

    return U01T*value.(Y0+Y1)*inv(Z0T*value.(Y0))
end

Fstab = make_controller(false)
Fopt = make_controller(false)

# solve ODE with new stabilizing controller
Tmax = 100.0
τ = 0.1
x0 = [0.5, 0.5]
new_u(x, t) = [Fstab[i](x...) for i=1:N]'*Z(x)
prob2 = ODEProblem(f!, x0, (0.0, Tmax), new_u)
solstab = DifferentialEquations.solve(prob2, Tsit5(), saveat=τ, dense=false, save_end=false)

# solve ODE with new optimal controller
new_u(x, t) = [Fopt[i](x...) for i=1:N]'*Z(x)
prob2 = ODEProblem(f!, x0, (0.0, Tmax), new_u)
solopt = DifferentialEquations.solve(prob2, Tsit5(), saveat=τ, dense=false, save_end=false)

times = solstab.t
statesstab = hcat(solstab.u...)
statesopt = hcat(solstab.u...)
plt = plot(times, [norm(statesstab[:,i]) for i in 1:length(times)], xlabel="t", ylabel="||x(t)||_2", label="stabilizing")
plot!(plot, times, [norm(statesopt[:,i]) for i in 1:length(times)], label="optimal")
savefig("data_driven.pdf")
