include("cartpole_env.jl")
using .CartPoleEnv
using Flux
using Statistics: mean
using Dates: now
using Distributions


function loss(model, γ, states_t, states_t1, actions_t, rewards_t, unfinished_t1, sz)
    
    # calculate y
    q_max_t1 = view(findmax(model(states_t1), dims=1)[1], 1, :)
    y = rewards_t .+ γ .* unfinished_t1 .* q_max_t1

    # calculate predicted y
    q_vals = model(states_t)
    s = 0:n_actions:sz*n_actions-n_actions
    idxs = actions_t .+ 1 .+ s
    ŷ = q_vals[idxs]

    # calculate the loss
    q_loss = (y .- ŷ) .^ 2
    l = mean(q_loss)

    return l
end

function train()
    
    # create the Q network which estimates the Q values for all posible actions (0, 1)
    model = Chain(
        Dense(state_size, 64, tanh),
        Dense(64, 64, tanh),
        Dense(64, n_actions)
    )
    opt = ADAM(3e-4)

    # set hyper-parameters
    γ = 0.99
    ϵ = 0.2
    max_steps = 200
    batch_size = 256

    # initialise buffers
    N = 1000000
    states_t = Array{Float32}(undef, state_size, N)
    states_t1 = Array{Float32}(undef, state_size, N)
    actions_t = Array{Int32}(undef, N)
    rewards_t = Array{Float32}(undef, N)
    unfinished_t1 = Array{Int32}(undef, N)
    n = 1

    # train
    for i = 1:1000

        # run an episode
        state = env_reset()
        r = 0
        for j = 1:max_steps
            # add the initial state to the buffer
            states_t[:, n] = state
            
            # use epsilon-greedy method of action selection
            action = 0
            if rand() > ϵ
                # select the action that has the highest value
                v = model(state)
                action = findmax(v)[2] - 1
            else
                if rand() > 0.5
                    action = 1
                end
            end

            # update the state from this action
            reward, done = env_step(state, action)
            r += reward

            # add the new state, action, reward and done to the buffers
            states_t1[:, n] = state
            actions_t[n] = action
            rewards_t[n] = reward
            if done
                unfinished_t1[n] = 0
            else
                unfinished_t1[n] = 1
            end
            n += 1

            # update Q function
            if n > 1000
                # choose a random batch from the buffer
                idxs = rand(1:n-1, batch_size)
                batch_states_t = states_t[:, idxs]
                batch_states_t1 = states_t1[:, idxs]
                batch_actions_t = actions_t[idxs]
                batch_rewards_t = rewards_t[idxs]
                batch_unfinished_t1 = unfinished_t1[idxs]

                # update
                ps = Flux.params(model)
                grad = gradient(() -> loss(model, γ, batch_states_t, batch_states_t1, batch_actions_t, batch_rewards_t, batch_unfinished_t1, batch_size), ps)
                Flux.update!(opt, ps, grad)

            end

            # shift buffer if full
            if n > N
                states_t[:, 1:900000] = view(states_t, :, 100001:N)
                states_t1[:, 1:900000] = view(states_t1, :, 100001:N)
                actions_t[:, 1:900000] = view(actions_t, 100001:N)
                rewards_t[:, 1:900000] = view(rewards_t, 100001:N)
                unfinished_t1[:, 1:900000] = view(unfinished_t1, 100001:N)
                n = 900001
            end

            if done
                break
            end

        end
        println(r)

    end
end

train()