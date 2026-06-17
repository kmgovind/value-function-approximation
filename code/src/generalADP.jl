# GeneralADP.jl
module GeneralADP

using Flux

export adp_critic_loss, adp_actor_loss

function adp_critic_loss(critic, actor, x_batch, dt, prob)
    loss = 0.0
    N = length(x_batch)

    for x in x_batch
        u      = actor(x)
        x_next = prob.step_dynamics(x, u, dt)

        target     = prob.reward(x, u) * dt + exp(-prob.rho * dt) * critic(x_next)[1]
        prediction = critic(x)[1]

        loss += (prediction - target)^2
    end
    return loss / N
end

function adp_actor_loss(critic, actor, x_batch, dt, prob)
    # Actor loss: minimize one-step cost + discounted value of next state.
    # Gradients flow: actor(x) -> step_dynamics -> critic(x_next)
    # This is the correct signature: actor takes state x (x_dim),
    # critic takes next state x_next (x_dim). actor(x) is never fed to critic.
    N = length(x_batch)
    loss = 0.0
    for x in x_batch
        u      = actor(x)
        x_next = prob.step_dynamics(x, u, dt)
        loss  += prob.reward(x, u) * dt + exp(-prob.rho * dt) * critic(x_next)[1]
    end
    return loss / N
end

end # module