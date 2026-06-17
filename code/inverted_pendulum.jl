using Flux
using Zygote
using Plots
using LinearAlgebra
using BSON
using JLD2
using ProgressMeter  # Added for real-time progress logging
using Statistics

# Load the Generic Problem Interface and Custom Method Solvers
include("src/ProblemInterface.jl")
include("src/GeneralPINN.jl")
include("src/GeneralBSDE.jl")
include("src/GeneralADP.jl")
include("src/GeneralNVI.jl")

using .ProblemInterface
using .GeneralPINN
using .GeneralADP
using .GeneralBSDE
using .GeneralNVI

# ==============================================================================
# 2. Problem Interface Configuration (Float32 Type Stability)
#    State: x = [θ, ω]  (angle, angular velocity)
#    Control: u  (torque)
#    Dynamics:
#      dθ = ω dt
#      dω = (g/l · sin(θ) + 1/(m·l²) · u) dt + σ dW
#    Cost: c(x, u) = (1 - exp(-β·θ²)) + γ·u²
# ==============================================================================
const g     = 9.81f0    # gravity
const l     = 1.0f0     # pendulum length
const m     = 1.0f0     # mass
const b     = 0.1f0     # damping coefficient (available for extended dynamics)
const sigma = 0.5f0     # noise intensity
const beta  = 1.0f0     # angle penalty
const gamma = 0.01f0    # control effort penalty

# State bounds for sampling: θ ∈ [-π, π], ω ∈ [-5, 5]
const THETA_MAX = Float32(π)
const OMEGA_MAX = 5.0f0

prob = StochasticProblem(
    0.05f0, # rho (discount rate)
    2,      # x_dim: state is [θ, ω]
    1,      # u_dim: scalar torque

    # Cost function c(x, u): angle deviation + control effort
    (x, u) -> (1.0f0 - exp(-beta * x[1]^2)) + gamma * u[1]^2,

    # Drift function f(x, u): pendulum SDE drift
    # dx/dt = [ω,  g/l·sin(θ) + u/(m·l²)]
    (x, u) -> [x[2],
               (g / l) * sin(x[1]) + (1.0f0 / (m * l^2)) * u[1]],

    # Diffusion matrix σ(x): noise enters only the ω equation
    # Returns a 2×1 matrix: [0; σ]
    (x) -> reshape([0.0f0, sigma], 2, 1),

    # Discrete step dynamics (Euler-Maruyama) with state clamping.
    # Clamping prevents BSDE/NVI rollouts from diverging to ±Inf during
    # early training when the policy is still random.
    (x, u, dt) -> begin
        dW    = Float32(sqrt(dt)) * randn(Float32)
        θ_new = x[1] + x[2] * dt
        ω_new = x[2] + ((g / l) * sin(x[1]) + (1.0f0 / (m * l^2)) * u[1]) * dt +
                sigma * dW
        # Clamp to valid state space so sin() never receives ±Inf
        θ_new = clamp(θ_new, -THETA_MAX, THETA_MAX)
        ω_new = clamp(ω_new, -OMEGA_MAX, OMEGA_MAX)
        return [θ_new, ω_new]
    end,

    # Analytical optimal control law u*(∇V, x)
    # From HJB first-order condition:  2γu + (1/(m·l²)) · ∂V/∂ω = 0
    # => u* = -1/(2γ·m·l²) · ∂V/∂ω
    # Clamped to [-U_MAX, U_MAX] so untrained gradients don't blow up rollouts.
    (grad_V, x) -> begin
        dV_dω  = grad_V[2]
        u_star = -dV_dω / (2.0f0 * gamma * m * l^2)
        return [clamp(u_star, -10.0f0, 10.0f0)]
    end,

    # State space uniform sampler: [θ, ω] ∈ [-π, π] × [-Ω_max, Ω_max]
    (num_samples) -> [
        [
            (2.0f0 * rand(Float32) - 1.0f0) * THETA_MAX,
            (2.0f0 * rand(Float32) - 1.0f0) * OMEGA_MAX
        ]
        for _ in 1:num_samples
    ],

    # PINN boundary condition: V(0, 0) = 0 (upright equilibrium has zero cost)
    (model) -> model([0.0f0, 0.0f0])[1]^2
)

# ==============================================================================
# 3. Initialize Models and Optimizers for All 4 Methods
# ==============================================================================
epochs     = 2000
batch_size = 64
dt         = 0.05f0   # Euler-Maruyama step size

println("Initializing Network Models...")

# --- Method 1: PINN ---
pinn_model = Chain(Dense(prob.x_dim, 32, sin), Dense(32, 32, tanh), Dense(32, 1))
opt_pinn   = Flux.setup(Flux.Adam(0.002), pinn_model)

# --- Method 2: Deep BSDE ---
V0_net    = Chain(Dense(prob.x_dim, 32, relu), Dense(32, 1))
Z_net     = Chain(Dense(prob.x_dim + 1, 32, relu), Dense(32, 1))   # +1 for time
opt_bsde  = Flux.setup(Flux.Adam(0.002), (V0_net, Z_net))
T_horizon = 1.0f0   # short horizon: pendulum dynamics diverge quickly without a good policy

# --- Method 3: ADP (Actor-Critic) ---
adp_critic = Chain(Dense(prob.x_dim, 32, relu), Dense(32, 1))
const U_MAX = 10.0f0
adp_actor  = Chain(Dense(prob.x_dim, 32, relu), Dense(32, prob.u_dim), x -> U_MAX .* tanh.(x))
opt_critic = Flux.setup(Flux.Adam(0.002), adp_critic)
opt_actor  = Flux.setup(Flux.Adam(0.0005), adp_actor)

# --- Method 4: Neural Value Iteration (NVI) ---
nvi_active = Chain(Dense(prob.x_dim, 32, relu), Dense(32, 1))
nvi_target = Chain(Dense(prob.x_dim, 32, relu), Dense(32, 1))
opt_nvi    = Flux.setup(Flux.Adam(0.002), nvi_active)
Flux.loadmodel!(nvi_target, nvi_active)

# ==============================================================================
# 4. Multi-Method Orchestrated Training Loop
# ==============================================================================
println("\nBeginning Coordinated Benchmarking Training Pipeline...")
println("(Note: Epoch 1 may take 10-20 seconds to pre-compile the Zygote gradient graphs...)")

prog = Progress(epochs, 1, "Training Models: ")

for epoch in 1:epochs
    x_batch = prob.sample_states(batch_size)

    # --- 4.1 Train PINN ---
    loss_pinn, grads_pinn = Zygote.withgradient(pinn_model) do m
        pinn_hjb_loss(m, x_batch, prob)
    end
    Flux.update!(opt_pinn, pinn_model, grads_pinn[1])

    # --- 4.2 Train Deep BSDE ---
    loss_bsde, grads_bsde = Zygote.withgradient(V0_net, Z_net) do v_net, z_net
        bsde_loss(v_net, z_net, x_batch, T_horizon, dt, prob)
    end
    Flux.update!(opt_bsde, (V0_net, Z_net), grads_bsde)

    # --- 4.3 Train ADP (Actor-Critic) ---
    loss_critic, grads_critic = Zygote.withgradient(adp_critic) do c
        adp_critic_loss(c, adp_actor, x_batch, dt, prob)
    end
    Flux.update!(opt_critic, adp_critic, grads_critic[1])

    loss_actor, grads_actor = Zygote.withgradient(adp_actor) do a
        adp_actor_loss(adp_critic, a, x_batch, dt, prob)
    end
    Flux.update!(opt_actor, adp_actor, grads_actor[1])

    # --- 4.4 Train Neural Value Iteration ---
    loss_nvi, grads_nvi = Zygote.withgradient(nvi_active) do m
        nvi_fitted_loss(m, nvi_target, x_batch, dt, prob, num_samples=3)
    end
    Flux.update!(opt_nvi, nvi_active, grads_nvi[1])

    if epoch % 20 == 0
        Flux.loadmodel!(nvi_target, nvi_active)
    end

    ProgressMeter.next!(prog; showvalues = [
        (:Epoch,       epoch),
        (:PINN_Loss,   round(loss_pinn,   digits=4)),
        (:BSDE_Loss,   round(loss_bsde,   digits=4)),
        (:Critic_Loss, round(loss_critic, digits=4)),
        (:NVI_Loss,    round(loss_nvi,    digits=4))
    ])
end

println("\nTraining complete! Generating comparison metrics...")

# ==============================================================================
# 5. Evaluation and Comparison Plots
#    We evaluate on a 2-D slice: vary θ ∈ [-π, π] with ω = 0 (static snapshots)
#    and separately vary ω ∈ [-Ω_max, Ω_max] with θ = 0.
# ==============================================================================
println("\nGenerating validation metrics...")

mkpath("figures")

grid_points  = 200

# --- Slice 1: θ sweep at ω = 0 ---
theta_range  = collect(range(-THETA_MAX, THETA_MAX, length=grid_points))
x_test_theta = [[th, 0.0f0] for th in theta_range]

# --- Slice 2: ω sweep at θ = 0 ---
omega_range  = collect(range(-OMEGA_MAX, OMEGA_MAX, length=grid_points))
x_test_omega = [[0.0f0, om] for om in omega_range]

# --- 5.1 Extract Value Function Predictions ---
v_pinn_theta = [pinn_model(x)[1]  for x in x_test_theta]
v_bsde_theta = [V0_net(x)[1]      for x in x_test_theta]
v_adp_theta  = [adp_critic(x)[1]  for x in x_test_theta]
v_nvi_theta  = [nvi_active(x)[1]  for x in x_test_theta]

v_pinn_omega = [pinn_model(x)[1]  for x in x_test_omega]
v_bsde_omega = [V0_net(x)[1]      for x in x_test_omega]
v_adp_omega  = [adp_critic(x)[1]  for x in x_test_omega]
v_nvi_omega  = [nvi_active(x)[1]  for x in x_test_omega]

# --- 5.2 Extract Implied Policies (via finite-difference gradient of V) ---
function extract_policy(model, x_vec::Vector{Float32})
    h = 1f-4
    # ∂V/∂ω: perturb second component only
    x_p = [x_vec[1], x_vec[2] + h]
    x_m = [x_vec[1], x_vec[2] - h]
    dV_dω = (model(x_p)[1] - model(x_m)[1]) / (2f0 * h)
    grad_V = [0.0f0, dV_dω]
    return prob.optimal_control_law(grad_V, x_vec)[1]
end

u_pinn_theta = [extract_policy(pinn_model, x)          for x in x_test_theta]
u_adp_theta  = [adp_actor(x)[1]                        for x in x_test_theta]
u_nvi_theta  = [extract_policy(nvi_active, x)          for x in x_test_theta]

# --- 5.3 Performance Reporting (relative std as a spread proxy; no closed form) ---
function coeff_variation(vals)
    μ = mean(vals)
    μ == 0 && return NaN
    return std(vals) / abs(μ)
end

println("\n=================================================")
println("   INVERTED PENDULUM: INTER-METHOD COMPARISON   ")
println("=================================================")
for (name, vals) in [
    ("PINN",    v_pinn_theta),
    ("BSDE",    v_bsde_theta),
    ("ADP",     v_adp_theta),
    ("NVI",     v_nvi_theta)
]
    println("$(lpad(name, 4)) V(θ,0) range: [$(round(minimum(vals), digits=3)), $(round(maximum(vals), digits=3))]  CV=$(round(coeff_variation(vals), digits=3))")
end
println("=================================================")

# --- 5.4 Plot 1: Value Function — θ sweep (ω = 0) ---
plt_v_theta = plot(theta_range, v_pinn_theta, label="PINN",       lw=2, color=:blue)
plot!(plt_v_theta, theta_range, v_bsde_theta, label="Deep BSDE",  lw=2, color=:green, ls=:dash)
plot!(plt_v_theta, theta_range, v_adp_theta,  label="ADP",        lw=2, color=:red,   ls=:dot)
plot!(plt_v_theta, theta_range, v_nvi_theta,  label="NVI",        lw=2, color=:purple, ls=:dashdot)
title!(plt_v_theta, "Value Function V(θ, ω=0)")
xlabel!(plt_v_theta, "Angle θ (rad)")
ylabel!(plt_v_theta, "V(θ, 0)")
plot!(plt_v_theta, legend=:top, grid=true)

# --- 5.5 Plot 2: Value Function — ω sweep (θ = 0) ---
plt_v_omega = plot(omega_range, v_pinn_omega, label="PINN",       lw=2, color=:blue)
plot!(plt_v_omega, omega_range, v_bsde_omega, label="Deep BSDE",  lw=2, color=:green, ls=:dash)
plot!(plt_v_omega, omega_range, v_adp_omega,  label="ADP",        lw=2, color=:red,   ls=:dot)
plot!(plt_v_omega, omega_range, v_nvi_omega,  label="NVI",        lw=2, color=:purple, ls=:dashdot)
title!(plt_v_omega, "Value Function V(θ=0, ω)")
xlabel!(plt_v_omega, "Angular Velocity ω (rad/s)")
ylabel!(plt_v_omega, "V(0, ω)")
plot!(plt_v_omega, legend=:top, grid=true)

# --- 5.6 Plot 3: Policy u*(θ) at ω = 0 ---
plt_policy = plot(theta_range, u_pinn_theta, label="PINN Policy",     lw=2, color=:blue)
plot!(plt_policy, theta_range, u_adp_theta,  label="ADP Actor",       lw=2, color=:red,   ls=:dot)
plot!(plt_policy, theta_range, u_nvi_theta,  label="NVI Policy",      lw=2, color=:purple, ls=:dashdot)
hline!(plt_policy, [0.0], label="", color=:black, ls=:dot, lw=1)
title!(plt_policy, "Implied Optimal Policy u*(θ, ω=0)")
xlabel!(plt_policy, "Angle θ (rad)")
ylabel!(plt_policy, "Torque u (N·m)")
plot!(plt_policy, legend=:topleft, grid=true)

master_plot = plot(plt_v_theta, plt_v_omega, plt_policy,
                   layout=(3, 1), size=(800, 900))
savefig(master_plot, "figures/inverted_pendulum_comparison.png")
println("Saved figure to figures/inverted_pendulum_comparison.png")

# ==============================================================================
# 6. Save Results
# ==============================================================================
mkpath("results")

models_file = "results/pendulum_models.bson"
data_file   = "results/pendulum_data.jld2"

try
    BSON.@save models_file pinn_model V0_net Z_net adp_critic adp_actor nvi_active nvi_target
    println("Saved models to $(models_file)")
catch e
    @error "Failed to save models to BSON" exception=(e,)
end

try
    JLD2.@save data_file theta_range omega_range \
        v_pinn_theta v_bsde_theta v_adp_theta v_nvi_theta \
        v_pinn_omega v_bsde_omega v_adp_omega v_nvi_omega \
        u_pinn_theta u_adp_theta u_nvi_theta
    println("Saved numeric data to $(data_file)")
catch e
    @error "Failed to save data to JLD2" exception=(e,)
end