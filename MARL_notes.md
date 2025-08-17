# Basics of Multi-Agent Architectures Related to Traffic Signal Control (TSC) Use Case

## 1. Independent Learners (IL)

- **What It Is**: Each agent (traffic light) learns its own policy independently, ignoring other agents' actions. They treat the environment (including other agents) as part of the dynamics to adapt to.
- **How It Works**: Each light observes its local traffic (e.g., lane queues) and chooses phases (e.g., green for north-south) based on its own reward (e.g., local waiting time reduction). Other agents' actions are seen as noise or part of the environment.
- **Pros**:
  - Simple to implement; no need for communication or shared models.
  - Works if intersections are largely independent (e.g., far apart with little interaction).
- **Cons**:
  - Poor coordination; one light’s green might block another’s flow, causing gridlock.
  - Struggles with non-stationarity (other lights’ policies change, making the environment unpredictable).
- **Relevance to TSC**: Not ideal for 4x4 grid, where lights are interconnected. For example, a green wave (coordinated greens) across intersections won’t emerge naturally.
- **In RLlib**: Use separate policies for each agent (e.g., `policies={'agent_1': ..., 'agent_2': ...}`) with no `policy_mapping_fn` sharing.

## 2. Centralized Training with Decentralized Execution (CTDE) - Current Approach

- **What It Is**: Agents train together using a centralized model or critic that sees all agents’ states, but execute decisions independently based on local observations.
- **How It Works**: During training, a shared policy or value function uses global info (e.g., all lights’ states) to learn. At runtime, each light uses only its local observation to act (e.g., switch phases based on its lanes’ traffic).
- **Pros**:
  - Balances coordination and scalability; the centralized critic handles credit assignment (who caused a delay?).
  - Stable for cooperative tasks like TSC, where all lights aim to reduce network delay.
- **Cons**:
  - Requires more computation during training due to global state processing.
  - Assumes symmetry (all lights are similar), which might not hold for complex networks.
- **Relevance to TSC**: This is the current setup with a shared policy in RLlib (e.g., `policies={'shared_policy': ...}` and `policy_mapping_fn=lambda agent_id: 'shared_policy'`). It’s effective for 4x4 grid, where lights should coordinate (e.g., avoiding simultaneous conflicting greens). The `diff-waiting-time` reward in sumo-rl supports this by reflecting network-wide effects.
- **In RLlib**: Use a single policy with a centralized critic (default in PPO) or customize with `multi_agent.policies_to_train` for advanced cases.

## 3. Fully Centralized (Centralized Everything)

- **What It Is**: One central agent controls all decisions, using a single policy that processes the full state of all agents.
- **How It Works**: The central model observes all lights’ traffic data and decides all phase switches simultaneously. It’s like one brain for the entire grid.
- **Pros**:
  - Perfect coordination; optimizes the whole network as a single problem.
  - Simplifies credit assignment since one entity decides everything.
- **Cons**:
  - Impractical for large or real-time systems; state space grows exponentially with agents (e.g., 16 lights in a 4x4 grid = huge complexity).
  - Not scalable; fails if network expands or communication lags.
- **Relevance to TSC**: Unsuitable for the project. A 4x4 grid with 16 lights would overwhelm a single policy, and real-world deployment would need local decisions.
- **In RLlib**: Define one policy with all agents’ observations concatenated, but this isn’t recommended.

## 4. Communication-Based Architectures

- **What It Is**: Agents can share info (e.g., their intended actions or observations) to coordinate better.
- **How It Works**: Lights might send signals (e.g., “I’m turning green in 5s”) to neighbors. This info is added to observations, and the policy learns to use it.
- **Pros**:
  - Improves coordination in dense networks (e.g., urban centers).
  - Can mimic traffic engineer strategies (e.g., green waves).
- **Cons**:
  - Adds complexity; requires defining communication channels (not natively in sumo-rl).
  - Increases observation space, slowing training.
- **Relevance to TSC**: Useful if 4x4 grid has heavy congestion where local decisions fail. We could extend sumo-rl observations (e.g., add neighbor states), but this is advanced and optional.
- **In RLlib**: Customize observation space in the policy (e.g., via a custom model) to include communication data.

## 5. Parameter Sharing (Current Approach)

- **What It Is**: All agents share the same policy network (weights), trained on their individual experiences. This is a technique where all agents (traffic lights) share the same neural network policy (i.e., the same weights and architecture). Each agent inputs its local observation (e.g., lane densities at its intersection) into this shared policy, which outputs an action (e.g., switch to the next phase).
- **How It Works**: Each light feeds its local observation to the shared policy, which outputs an action. The policy updates based on all agents’ rewards, averaged or combined.
- **Pros**:
  - Efficient; one model serves all agents, reducing memory/compute needs.
  - Encourages symmetry, fitting your grid where lights have similar roles.
  - Handles non-stationarity well with PPO’s stability.
- **Cons**:
  - Assumes agents are similar; fails if intersections differ greatly (e.g., major vs. minor roads).
  - Limited coordination if rewards are too local (e.g., only own waiting time).
- **Relevance to TSC**: This is the current setup with RLlib’s `shared_policy`. It’s ideal for 4x4 grid with uniform intersections and `diff-waiting-time` reward, which captures network effects. It’s the simplest cooperative architecture for your timeline.
- **In RLlib**: Defined as `policies={'shared_policy': (None, None, None, {})}` with `policy_mapping_fn=lambda agent_id: 'shared_policy'`.

## Key Takeaways for the Project

- **Best Fit**: Use **Parameter Sharing (CTDE with shared policy)**, as implemented in current RLlib code. It’s efficient, matches sumo-rl’s design. The shared policy learns coordination implicitly via the reward.
- **When to Adjust**:
  - If coordination fails (e.g., gridlock), consider **Communication-Based** by adding neighbor data to observations.
  - For tiny tests, try **Fully Centralized**, but switch back for scale.
  - Avoid **Independent Learners**.
- **Focus**: Tune the shared policy (e.g., adjust `entropy_coeff` for exploration) and reward function to enhance cooperation. Monitor `episode_reward_mean` to ensure network-wide improvement.

## Current Setup
In the current RLlib configuration (in Config.multi_agent())

```py
policies={'shared_policy': (None, None, None, {})}, policy_mapping_fn=lambda agent_id: 'shared_policy'
```

both parameter sharing and CTDE are employed because they complement each other in a cooperative MARL setting like TSC. 

Here’s how they integrate:

> Parameter Sharing as the Core Mechanism:

The setup uses a single shared_policy for all traffic lights. This means the same neural network weights are applied to each agent’s local observation to produce actions. This is the "parameter sharing" part.
Example: Light 1 and Light 2 both use the same policy to decide phases based on their respective lane data, trained on all agents’ experiences.


> CTDE as the Training Strategy:

RLlib’s PPO implementation, by default, supports CTDE when using a shared policy. During training, the algorithm collects data from all agents (e.g., their observations, actions, and rewards) and uses a centralized value function to compute advantages. This value function can consider the global state (implicitly through the shared policy’s updates across agents) to stabilize learning.
At execution (e.g., during algo.evaluate()), each agent uses only its local observation to act, making it decentralized.
Example: The policy learns that a green phase at one light might reduce waiting time network-wide, guided by the centralized training data, but each light decides based on its own traffic.


How They Overlap:

Parameter sharing is a specific implementation choice within the CTDE framework. The shared policy acts as the decentralized execution component, while the training process leverages centralized data aggregation (all agents’ rollouts) to update that single policy.
In RLlib, the default PPO setup with a shared policy automatically incorporates CTDE because it pools experiences across agents to train the shared model, aligning with the cooperative goal of minimizing total delay (e.g., via diff-waiting-time reward).

