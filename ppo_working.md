### Proximal Policy Optimization (PPO) Algorithm

PPO is an actor-critic method, meaning it uses two neural networks: an **actor** (policy) to decide actions and a **critic** (value function) to estimate how good those actions are. The "proximal" part refers to keeping new policies close to old ones to avoid drastic changes, which is achieved through clipping.

---

#### Step-by-Step Explanation of PPO

PPO operates in iterations, where it collects data from the environment, computes advantages, and updates the policy. 

---

1. **Initialize the Policy and Value Function**:
   - Start with a policy $\pi$ (actor) that maps observations to actions and a value function $V$ (critic) that estimates the expected return from a state.
   - The policy is a neural network outputting action probabilities (for discrete actions like phase selection).

---

2. **Collect Rollouts**:
   - Use the current policy to interact with the environment for a batch of timesteps (e.g., `train_batch_size_per_learner=256`).
   - For each timestep $t$:
     - Observe state $s_t$ (e.g., traffic state).
     - Sample action $a_t$ from $\pi(s_t)$.
     - Execute $a_t$, get reward $r_t$ and next state $s_{t+1}$.
   - This generates trajectories of $(s_t, a_t, r_t, s_{t+1})$.
   - In multi-agent setups, rollouts are collected across all agents (traffic lights) simultaneously in the shared environment.

---

3. **Compute Advantages and Returns**:
   - Estimate the advantage $A_t$ for each timestep, which measures how much better an action is than expected.
   - Use Generalized Advantage Estimation (GAE) if enabled (`use_gae=True`):

     $$
     A_t = \delta_t + (\gamma \lambda) \delta_{t+1} + (\gamma \lambda)^2 \delta_{t+2} + \dots
     $$

     where:

     $$
     \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
     $$

     - $\gamma$ is the discount factor (your `gamma=0.99`).
     - $\lambda$ is the GAE parameter (your `lambda_=0.95`).

   - Advantages help the policy learn which actions lead to better outcomes.

---

4. **Update the Policy with Clipped Objective**:
   - PPO optimizes a surrogate loss function:

     $$
     L(\theta) = \min \Bigg( 
     \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)} A_t , \;
     \text{clip}\Big( 
     \frac{\pi_\theta(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}, 
     1 - \epsilon, 1 + \epsilon 
     \Big) A_t 
     \Bigg)
     $$

     - $\pi_\theta$ is the new policy, $\pi_{\text{old}}$ is the old policy, $A_t$ is the advantage, and $\epsilon$ is the clip parameter (`clip_param=0.2`).
     - Clipping prevents large policy shifts, ensuring stability.

   - Add entropy to the loss for exploration (`entropy_coeff=0.1`): encourages random actions to avoid premature convergence.
   - Use KL divergence regularization (`kl_coeff=0.2`) to penalize large deviations from the old policy.

---

5. **Update the Value Function**:
   - The critic is updated to minimize the squared error between predicted and actual returns:

     $$
     L_V = \frac{1}{2} \Big( V(s_t) - R_t \Big)^2
     $$

     with clipping (`vf_clip_param=10.0`) to avoid large updates.
   - The value loss is weighted by `vf_loss_coeff=0.5`.

---

6. **Gradient Clipping and Optimization**:
   - Apply gradient clipping (`grad_clip=0.5`) to prevent exploding gradients.
   - Use an optimizer (e.g., Adam) with learning rate (`lr=2e-5`) to update weights over multiple epochs on the batch.

---

7. **Repeat for Iterations**:
   - Repeat for `n_iterations`, checkpointing every `checkpoint_freq`.
   - The batch size (`train_batch_size_per_learner=256`) determines how much data is used per update.

---

#### What Each Hyperparameter Does and Means

- **`lr` (Learning Rate)**: Controls the step size in gradient descent. 

- **`train_batch_size_per_learner` (Batch Size)**: Number of samples per update. `256` means smaller, frequent updates. Larger batches give smoother gradients but need more memory.

- **`entropy_coeff` (Entropy Coefficient)**: Encourages exploration. `0.1` prevents the agent from always choosing the same action.

- **`kl_coeff` (KL Divergence Coefficient)**: Penalizes large policy shifts. `0.2` balances stability with adaptation.

- **`clip_param` (Clip Parameter)**: $\epsilon$ in PPO objective. `0.2` means ratios stay in $[0.8, 1.2]$, ensuring safe updates.

- **`vf_clip_param` (Value Function Clip Parameter)**: Clips large critic updates. `10.0` prevents instability in value learning.

- **`gamma` (Discount Factor)**: Future reward weighting. `0.99` emphasizes long-term optimization (important in traffic control).

- **`lambda_` (GAE Lambda)**: Bias-variance tradeoff in GAE. `0.95` smooths advantage estimates for stability.

- **`use_gae` (Generalized Advantage Estimation)**: If True, reduces variance in advantage calculation. Your setup: `True`.

- **`vf_loss_coeff` (Value Loss Coefficient)**: Weight for critic loss. `0.5` balances policy and critic learning.

- **`grad_clip` (Gradient Clipping)**: Caps gradient norm. `0.5` stabilizes training.

---

PPO’s clipping and entropy make it more stable than vanilla policy gradients, which is why it's effective for traffic signal control (TSC). Small, clipped updates let the shared policy learn coordinated timings without risking gridlock.
