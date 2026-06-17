# GeneralBSDE.jl
module GeneralBSDE

using Flux
using LinearAlgebra

export bsde_loss

function bsde_loss(V0_net, Z_net, x0_batch, T, dt, prob)
    batch_size = length(x0_batch)
    steps = round(Int, T / dt)

    x = x0_batch
    Y = [V0_net(x_i)[1] for x_i in x]

    for step in 0:(steps-1)
        t  = step * dt
        discount = exp(-prob.rho * t)

        # Z_net maps [x; t] -> noise_dim-vector, approximating σᵀ∇V
        Z = [Z_net(vcat(x[i], [eltype(x[i])(t)])) for i in 1:batch_size]

        x_and_Y = map(1:batch_size) do i
            g_val = prob.diffusion(x[i])   # (x_dim × noise_dim) matrix

            Tz = eltype(Z[i])
            gT = Tz.(g_val)               # (x_dim × noise_dim)
            zi = Tz.(Z[i])                # (noise_dim,)  — Z_net output

            # Recover ∇V from the relation  Z ≈ discount · σᵀ ∇V
            #   σ is (x_dim × noise_dim), so σᵀ is (noise_dim × x_dim)
            #   Z = σᵀ ∇V  =>  ∇V = σ (σᵀσ)⁻¹ Z  (pseudo-inverse left-multiply)
            # For a single noise channel (noise_dim=1) this is just:
            #   ∇V = gT * zi  (matrix-vector: (x_dim×1) * scalar = x_dim-vector)
            # We use the Moore-Penrose pseudo-inverse for the general case.
            grad_V = if discount > 1f-5
                (gT * gT' + Tz(1e-6) * I) \ (gT * zi) ./ convert(Tz, discount)
            else
                zeros(Tz, prob.x_dim)
            end

            u_star = prob.optimal_control_law(grad_V, x[i])
            r_val  = prob.reward(x[i], u_star)

            # Brownian increment has noise_dim components
            noise_dim = size(gT, 2)
            dW = sqrt(dt) .* randn(Tz, noise_dim)

            dtT        = convert(Tz, dt)
            discountT  = convert(Tz, discount)
            rT         = convert(Tz, r_val)
            y_i_next   = convert(Tz, Y[i]) - discountT * rT * dtT + dot(zi, dW)

            Tx = eltype(x[i])
            x_i_next = Tx.(prob.step_dynamics(x[i], u_star, dt))

            return (x_i_next, y_i_next)
        end

        x = [pair[1] for pair in x_and_Y]
        Y = [pair[2] for pair in x_and_Y]
    end

    return sum(Y .^ 2) / batch_size
end

end # module