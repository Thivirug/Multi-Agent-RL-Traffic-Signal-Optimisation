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
- **Relevance to TSC**: Not ideal for grids, where lights are interconnected. For example, a green wave (coordinated greens) across intersections won’t emerge naturally.
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
- **Relevance to TSC**: Useful if 2x2 grid has heavy congestion where local decisions fail. We could extend sumo-rl observations (e.g., add neighbor states), but this is advanced and optional.
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
- **Relevance to TSC**: This is the current setup with RLlib’s `shared_policy`. It’s ideal for 2x2 grid with uniform intersections and `diff-waiting-time` reward, which captures network effects. It’s the simplest cooperative architecture for your timeline.



