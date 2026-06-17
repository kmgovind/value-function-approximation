using Flux
using Zygote
using Plots
using LinearAlgebra
using BSON
using JLD2
using ProgressMeter  # Added for real-time progress logging

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
# ==============================================================================
const α_param = 0.5f0
const r_param = 3.0f0
const σ_param = 0.1f0
const X_MAX   = 15.0f0

prob = StochasticProblem(
    0.05f0, # rho (Discount rate)
    1,    # x_dim
    1,    # u_dim
    
    # Reward function c(x, u)
    (x, u) -> 1.0f0 - exp(-α_param * u[1]),
    
    # Drift function f(x, u)
    (x, u) -> [r_param * x[1] - u[1]],
    
    # Diffusion matrix σ(x)
    (x) -> reshape([σ_param * x[1]], 1, 1),
    
    # Discrete step dynamics (Euler-Maruyama with absorbing boundary at 0)
    (x, u, dt) -> begin
        if x[1] <= 0.0f0 return [0.0f0] end
        dx = (r_param * x[1] - u[1]) * dt + (σ_param * x[1]) * Float32(sqrt(dt)) * randn(Float32)
        return [max(0.0f0, x[1] + dx)]
    end,
    
    # Analytical optimal policy u*(∇V, x)
    (grad_V, x) -> begin
        dV_dx = grad_V[1]
        if dV_dx > 0.0f0 && (dV_dx / α_param) < 1.0f0
            return [- (1.0f0 / α_param) * log(dV_dx / α_param)]
        else
            return [0.0f0]
        end
    end,
    
    # State space uniform sampler (Float32 native)
    (num_samples) -> [rand(Float32, 1) .* X_MAX for _ in 1:num_samples],
    
    # PINN boundary condition penalty: V(0) = 0
    (model) -> model([0.0f0])[1]^2
)

# ==============================================================================
# 3. Initialize Models and Optimizers for All 4 Methods
# ==============================================================================
epochs = 2000      
batch_size = 64
dt = 0.05f0        # INCREASED dt: Prevents Zygote from choking on massive AD graphs

println("Initializing Network Models...")

# --- Method 1: PINN ---
pinn_model = Chain(Dense(prob.x_dim, 32, sin), Dense(32, 32, tanh), Dense(32, 1))
opt_pinn   = Flux.setup(Flux.Adam(0.002), pinn_model)

# --- Method 2: Deep BSDE ---
V0_net   = Chain(Dense(prob.x_dim, 32, relu), Dense(32, 1))
Z_net    = Chain(Dense(prob.x_dim + 1, 32, relu), Dense(32, 1)) 
opt_bsde = Flux.setup(Flux.Adam(0.002), (V0_net, Z_net))
T_horizon = 6.0f0 # REDUCED Horizon: Keeps the backward trajectory graph manageable

# --- Method 3: ADP (Actor-Critic) ---
adp_critic = Chain(Dense(prob.x_dim, 32, relu), Dense(32, 1))
const U_MAX = 10.0f0
adp_actor  = Chain(Dense(prob.x_dim, 32, relu), Dense(32, prob.u_dim), Dense(prob.u_dim, prob.u_dim, σ))
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

# Initialize the Progress Bar
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
        adp_actor_loss(adp_critic, a, x_batch, prob)
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
    
    # Update the Progress Bar with the current losses
    ProgressMeter.next!(prog; showvalues = [
        (:Epoch, epoch),
        (:PINN_Loss, round(loss_pinn, digits=4)),
        (:BSDE_Loss, round(loss_bsde, digits=4)),
        (:Critic_Loss, round(loss_critic, digits=4)),
        (:NVI_Loss, round(loss_nvi, digits=4))
    ])
end

println("\nTraining complete! Generating comparison metrics...")

# ==============================================================================
# 5. Evaluation and Comparison Plots for Asset Harvesting
# ==============================================================================
println("\nGenerating validation metrics against TRUE Harvesting Baseline...")

grid_points = 200
x_test_raw  = collect(range(0.01f0, X_MAX, length=grid_points))
x_test      = [[x] for x in x_test_raw]

# --- 5.1 Compute Mathematically Exact Harvesting Baselines ---
const slope_true = α_param / (α_param * r_param - prob.rho)
v_true = [slope_true * x for x in x_test_raw]

const constant_u = -(1.0f0 / α_param) * log(1.0f0 / (α_param * r_param - prob.rho))
u_true = [constant_u for _ in x_test_raw]

# --- 5.2 Extract Predictions ---
v_pinn   = [pinn_model(x)[1] for x in x_test]
v_bsde   = [V0_net(x)[1] for x in x_test]
v_adp    = [adp_critic(x)[1] for x in x_test]
v_nvi    = [nvi_active(x)[1] for x in x_test]

# --- 5.3 Performance Reporting (RMSE) ---
rmse(pred, true_val) = sqrt(sum((pred .- true_val).^2) / length(true_val))

println("\n==================================================")
println("   ASSET HARVESTING QUANTITATIVE METRICS (RMSE)   ")
println("==================================================")
println("PINN / DGM RMSE:            ", rmse(v_pinn, v_true))
println("Deep BSDE RMSE:             ", rmse(v_bsde, v_true))
println("ADP (Actor-Critic) RMSE:    ", rmse(v_adp, v_true))
println("Neural Value Iteration RMSE:", rmse(v_nvi, v_true))
println("==================================================")

# --- 5.4 Plot 1: Value Function Comparison ---
plt_val = plot(x_test_raw, v_true, label="Ground Truth (Analytical)", linewidth=3, color=:black)
plot!(plt_val, x_test_raw, v_pinn, label="PINN / DGM", linewidth=2.0, color=:blue)
plot!(plt_val, x_test_raw, v_bsde, label="Deep BSDE (Truncated)", linewidth=2.0, color=:green, linestyle=:dash)
plot!(plt_val, x_test_raw, v_adp, label="ADP (Actor-Critic)", linewidth=2.0, color=:red, linestyle=:dot)
plot!(plt_val, x_test_raw, v_nvi, label="Neural Value Iteration", linewidth=2.0, color=:purple, linestyle=:dashdot)

title!(plt_val, "Value Function V(x) - Asset Harvesting")
xlabel!(plt_val, "Resource Population Size (x)")
ylabel!(plt_val, "Expected Discounted Harvest V(x)")
plot!(plt_val, legend=:bottomright, grid=true)

# --- 5.5 Plot 2: Implied Policy Optimization Comparison ---
function extract_policy(model, x_vec)
    et = eltype(x_vec)
    h = convert(et, 1e-4)
    v_up = model(x_vec .+ h)[1]
    x_minus = x_vec .- h
    x_minus_clamped = [ max(xx, zero(et)) for xx in x_minus ]
    v_dn = model(x_minus_clamped)[1]
    grad = [(v_up - v_dn) / (2 * h)]
    return prob.optimal_control_law(grad, x_vec)[1]
end

u_pinn = [extract_policy(pinn_model, x) for x in x_test]
u_adp  = [max(0.0f0, adp_actor(x)[1]) for x in x_test] 
u_nvi  = [extract_policy(nvi_active, x) for x in x_test]

plt_pol = plot(x_test_raw, u_true, label="Ground Truth Policy", linewidth=3, color=:black)
plot!(plt_pol, x_test_raw, u_pinn, label="PINN Induced Policy", linewidth=2.0, color=:blue)
plot!(plt_pol, x_test_raw, u_adp, label="ADP Explicit Actor", linewidth=2.0, color=:red, linestyle=:dot)
plot!(plt_pol, x_test_raw, u_nvi, label="NVI Induced Policy", linewidth=2.0, color=:purple, linestyle=:dashdot)

title!(plt_pol, "Optimal Harvesting Policy u*(x)")
xlabel!(plt_pol, "Resource Population Size (x)")
ylabel!(plt_pol, "Harvesting Extraction Rate (u)")
plot!(plt_pol, legend=:topleft, grid=true)


master_plot = plot(plt_val, plt_pol, layout=(2,1), size=(800, 700))
savefig(master_plot, "figures/asset_harvesting_comparison.png")

# Ensure results directory exists
mkpath("results")

# Save models (BSON) and numeric arrays/metrics (JLD2)
models_file = "results/asset_harvest_models.bson"
data_file = "results/asset_harvest_data.jld2"

try
    BSON.@save models_file pinn_model V0_net Z_net adp_critic adp_actor nvi_active nvi_target
    println("Saved models to $(models_file)")
catch e
    @error "Failed to save models to BSON" exception=(e,)
end

try
    JLD2.@save data_file v_true u_true v_pinn v_bsde v_adp v_nvi u_pinn u_adp u_nvi
    println("Saved numeric data/metrics to $(data_file)")
catch e
    @error "Failed to save data to JLD2" exception=(e,)
end