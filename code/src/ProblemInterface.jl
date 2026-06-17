# ProblemInterface.jl
module ProblemInterface
export StochasticProblem

struct StochasticProblem
    rho::Float32                  # Infinite horizon discount rate (Float32 for type stability)
    x_dim::Int                    # State space dimensionality
    u_dim::Int                    # Control space dimensionality

    # Core mathematical operators (Functions)
    reward::Function              # (x, u) -> Scalar
    drift::Function               # (x, u) -> Vector of length x_dim
    diffusion::Function           # (x)    -> Matrix of size (x_dim, noise_dim)
    step_dynamics::Function       # (x, u, dt) -> Vector of length x_dim
    optimal_control_law::Function # (grad_V, x) -> Vector of length u_dim

    # Bounding/Sampling helpers
    sample_states::Function       # (num_samples) -> Vector of Vectors
    boundary_penalty::Function    # (model) -> Scalar (for PINN boundary constraints)
end

end # module