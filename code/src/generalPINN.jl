# GeneralPINN.jl
module GeneralPINN

using Flux
using Zygote
using LinearAlgebra

export pinn_hjb_loss

function pinn_hjb_loss(model, x_batch, prob)
    loss = 0.0f0
    N = length(x_batch)

    for x in x_batch
        V(vec) = model(vec)[1]

        # ∇V — Zygote reverse-mode; differentiable w.r.t. model params
        grad_V, = Zygote.gradient(V, x)

        # Hessian via column-wise finite differences of ∇V.
        # Built functionally (no mutation) so Zygote can backprop through it.
        # Each column: (∇V(x+eᵢ) - ∇V(x-eᵢ)) / 2h
        et = eltype(x)
        h  = convert(et, 1f-3)
        hess_V = let
            cols = map(1:prob.x_dim) do i
                ei   = [j == i ? h : zero(et) for j in 1:prob.x_dim]
                gp,  = Zygote.gradient(V, x .+ ei)
                gm,  = Zygote.gradient(V, x .- ei)
                (gp .- gm) ./ (2 .* h)
            end
            # hcat the column vectors into an (x_dim × x_dim) matrix — no mutation
            hcat(cols...)
        end
        hess_V = (hess_V .+ hess_V') ./ 2  # symmetrize

        # Optimal control and HJB components
        u_star = prob.optimal_control_law(grad_V, x)

        v_val  = V(x)
        r_val  = prob.reward(x, u_star)
        f_val  = prob.drift(x, u_star)
        g_val  = prob.diffusion(x)   # (x_dim × noise_dim)

        # 0.5 · Tr(σσᵀ ∇²V)
        σσᵀ            = g_val * g_val'
        diffusion_term = 0.5f0 * sum(σσᵀ .* hess_V)

        # Stationary HJB residual: ρV - [c + f·∇V + 0.5·Tr(σσᵀ∇²V)] = 0
        residual = prob.rho * v_val - (r_val + dot(f_val, grad_V) + diffusion_term)

        loss += residual^2 + 10.0f0 * prob.boundary_penalty(model)
    end

    return loss / N
end

end # module