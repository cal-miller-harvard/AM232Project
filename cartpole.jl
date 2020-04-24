using DifferentialEquations, Flux, DiffEqSensitivity
using Suppressor, Plots, Luxor, Printf, Distributed, Statistics

# Produces animation of cartpole
function animate_sol(sol, p, K, fname)
    framerate = 30
    size = (400, 200)
    cart = (50, 20)
    scale = 50
    fscale = 50
    time = last(sol.t) - sol.t[1]
    l = p[4]

    path = mktempdir()
    is = 0:round(Int, framerate * time)
    for i in is
        t = sol.t[1] + i/framerate
        solt = sol(t)
        x, θ, v, ω, Lval = solt
        kt = K(solt, t)
        cartpos = Point(scale*x, 0.0)
        polepos = Point(scale*(x+l*sin(θ)), scale*l*cos(θ))
        forcepos = Point(scale*x+fscale*kt, 0.0)
        textpos = Point(-size[1]/2 + 25, -size[2]/2 + 25)
        textpos2 = Point(-size[1]/2 + 25, -size[2]/2 + 50)

        Drawing(size[1], size[2], path*"/$i.png")
        origin()
        background("white")
        sethue("black")
        box(cartpos, cart[1], cart[2], :stroke)
        line(cartpos, polepos, :stroke)
        fontsize(20)
        Luxor.text(@sprintf("Time: %.3f",t), textpos, halign=:left, valign=:top)
        Luxor.text(@sprintf("Loss: %.3f",Lval), textpos2, halign=:left, valign=:top)
        sethue("red")
        if kt != 0
            Luxor.arrow(cartpos, forcepos)
        end
        finish()
    end
    @suppress run(`ffmpeg -f image2 -i $(path)/%d.png -vf palettegen -y $(path)/palette.png`)
    @suppress run(`ffmpeg -framerate $framerate -f image2 -i $(path)/%d.png -i $(path)/palette.png -lavfi paletteuse -y $fname`)
end

function run_cartpole(K::Function, L::Function, animate=false, plotsol=false, concrete=true, randomize=true)
    # Calculates the derivative of state per http://underactuated.mit.edu/acrobot.html#cart_pole
    function f!(du, u, p, t)
        x, θ, v, ω, Lval = u
        g, mc, mp, l = p
        f = K(u, t)
        du[1] = v
        du[2] = ω
        du[3] = (f+mp*sin(θ)*(l*ω^2+g*cos(θ)))/(mc+mp*sin(θ)^2)
        du[4] = (-f*cos(θ)-mp*l*ω^2*cos(θ)*sin(θ)-(mc+mp)*g*sin(θ))/(l*(mc+mp*sin(θ)^2))
        du[5] = L(u, f)
    end
    
    # Set up ODE problem
    if randomize
        u0 = randn(5)
    else
        u0 = zeros(5)
    end
    tspan = (0.0, 10.0)
    p = [9.81, 1.0, 0.1, 1.0]
    # p = CartPole(K, L)
    prob = ODEProblem(f!, u0, tspan, p)

    # Solve ODE problem
    if concrete
        sol = concrete_solve(prob, Tsit5(), calck=false, alias_u0=true, sensealg=TrackerAdjoint())
    else
        sol = solve(prob, Tsit5(), calck=false, alias_u0=true)
    end

    # Animate solution
    if animate
        animate_sol(sol, p, K, "cartpole.gif")
    end

    # Plot solution
    if plotsol
        plot(sol)
        savefig("cartpole.pdf")
    end
    return sol
end

function train_cartpole()
    # Define controller and loss function
    nn = Chain(Dense(4,4,relu,initb=randn), Dense(4,4,relu), Dense(4,1))
    ps = params(nn)
    K(u, t) = nn(u[1:4])[1]

    function L(u, f)
        x, θ, v, ω, Lval = u
        return x^2 + (θ - pi)^2 + v^2 + ω^2 + f^2
    end
    loss() = last(run_cartpole(K, L, false, false, true))[5]

    # Train model
    η = 0.001 # Learning rate
    opt = ADAM(η)

    for i in 1:50
        training_loss = 0.0
        gs = gradient(ps) do
            batchsize = 50
            losses = [loss() for i in 1:batchsize]
            training_loss = mean(losses)
            return training_loss
        end
        println("Epoch: $i  Loss: $training_loss")
        Flux.update!(opt, ps, gs)
    end
    
    return K, L
end

k, l = train_cartpole()
# Plot and animate results
sol = run_cartpole(k, l, true, true, false, false)
println("Done!")