# GeneralNVI.jl
module GeneralNVI

using Flux

export nvi_fitted_loss

function nvi_fitted_loss(active_model, target_model, x_batch, dt, prob; num_samples=5)
    N = length(x_batch)
    loss = 0.0

    for x in x_batch
        et = eltype(x)
        h  = convert(et, 1e-3)

        # Finite-difference gradient of the target value network.
        # No boundary clamping — valid for any state space (including
        # pendulum where θ ∈ [-π,π] and ω ∈ [-Ω,Ω], both can be negative).
        grad_V = [begin
            e_d  = [i == d ? h : zero(et) for i in 1:prob.x_dim]
            V_up = target_model(x .+ e_d)[1]
            V_dn = target_model(x .- e_d)[1]
            (V_up - V_dn) / (2h)
        end for d in 1:prob.x_dim]

        u_star = prob.optimal_control_law(grad_V, x)

        # Monte-Carlo rollout expectation
        future_expectation = zero(typeof(target_model(x)[1]))
        for _ in 1:num_samples
            x_next = prob.step_dynamics(x, u_star, dt)
            future_expectation += target_model(x_next)[1]
        end
        future_expectation /= num_samples

        target     = prob.reward(x, u_star) * dt + exp(-prob.rho * dt) * future_expectation
        prediction = active_model(x)[1]

        loss += (prediction - target)^2
    end

    return loss / N
end

end # module