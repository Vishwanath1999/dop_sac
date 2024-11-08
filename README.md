# Decomposed Off-Policy Gradients based on SAC for Multi-Agent RL

This code implements a multi-agent Reinforcement Learning (RL) algorithm using Decomposed Off-Policy Gradients based on Soft Actor-Critic (SAC). The algorithm involves training multiple agents simultaneously in an environment. The agents interact with the environment, collect experiences, and use those experiences to update their policies. The SAC algorithm is used for both the actors and the mixers to optimize the agents' policies.

## Code Structure

The code is divided into several classes:

### 1. `ReplayBuffer`:

A replay buffer is used to store experiences (state, action, reward, next state, done flag, and noise) for each agent. The replay buffer is used to sample batches of experiences for training the agents.

### 2. `ActorNetwork`:

This class defines the actor network for each agent. The actor takes the state as input and outputs the action, following a normal distribution. The actor is updated using the SAC algorithm, and the policy is reparameterized to allow for efficient gradient computation.

### 3. `MixerNetwork`:

This class defines the mixer network for each agent. The mixer takes the state, action, and noise as input and outputs the Q-value for each agent. There are two mixers, and the minimum Q-value among them is used for policy improvement.

---

The `MixerNetwork` is a critical component of the multi-agent system, responsible for aggregating the Q-values of all agents to produce a global Q-value. This allows for coordination and cooperation among the agents to optimize their collective behavior.

#### Mixer Network Architecture:

The Mixer Network consists of multiple components:

1. **GRUs (Gated Recurrent Units) for Each Agent**:
   - The Mixer Network employs a separate GRU for each agent. The GRU processes the agent's individual state information and captures temporal dependencies in the sequence of states experienced by the agent.

2. **Fully Connected Layers (`fc1s`) for Each Agent**:
   - After processing the agent's state sequence through the corresponding GRU, the network concatenates the last hidden state of the GRU with the agent's action and noise vector.
   - The concatenated vector is then passed through a fully connected layer (`fc1`) to capture the interactions between the agent's current state, action, and noise.

3. **Q-value Estimation (`qs`) for Each Agent**:
   - Following the `fc1` layer, another fully connected layer (`qs`) is used to compute the Q-value for each agent. This Q-value reflects the expected future reward for an agent taking a specific action in its current state.

4. **Hypernetwork for Weight Generation**:
   - The Mixer Network utilizes a hypernetwork to generate weight coefficients that determine the importance of each agent's Q-value when computing the global Q-value.
   - The hypernetwork is conditioned on the global state, allowing the Mixer Network to adjust the weight coefficients based on the overall context of the multi-agent system.

#### Weighted Q-value Aggregation:

To calculate the global Q-value, the Mixer Network performs the following steps:

1. The last hidden state of the GRU for each agent, along with the agent's action and noise vector, is passed through the corresponding `fc1` layer to obtain individual Q-values for all agents.

2. The hypernetwork generates weight coefficients based on the global state. These coefficients determine the importance of each agent's Q-value when computing the global Q-value.

3. The individual Q-values of all agents are linearly combined using the weight coefficients obtained from the hypernetwork. This aggregation process yields a global Q-value, reflecting the overall expected future reward for the entire multi-agent system.

4. The global Q-value can then be used for policy improvement and decision-making, enabling coordinated actions among the agents in the environment.

By employing the Mixer Network to calculate a global Q-value, the multi-agent system benefits from enhanced coordination and improved performance in complex environments where agents' actions are interdependent.

### 4. `dop_agent`:

The `dop_agent` class combines the actors and mixers for all the agents. It handles the agent's interaction with the environment, storing experiences in the replay buffer, and performing updates using the decomposed off-policy gradients.

## Usage

To use this code for training agents in a multi-agent environment, follow these steps:

1. Import the required libraries.
2. Define the environment and create an instance of the `dop_agent` class.
3. Interact with the environment using the agent's `choose_action` method to get actions, and use the `remember` method to store experiences in the replay buffer.
4. Periodically call the agent's `learn` method to update the actor and mixer networks using the decomposed off-policy gradients based on SAC.
5. Optionally, save and load models using the `save_models` and `load_models` methods.

Example usage:

```python
# 1. Import libraries
import numpy as np
import torch

# 2. Define the environment and create the dop_agent instance
env = YourEnvironment()  # Replace 'YourEnvironment' with the actual environment class
input_dims = env.observation_space.shape
n_agents = 2  # Number of agents in the environment
n_actions = env.action_space.shape[0]
agent = dop_agent(input_dims=input_dims, n_agents=n_agents, n_actions=n_actions)

# 3. Interact with the environment and store experiences in the replay buffer
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        actions = agent.choose_action(state)  # Get actions from the agent
        next_state, reward, done, _ = env.step(actions)
        noise = np.random.normal(0, 0.1, size=(n_agents, n_actions))
        agent.remember(state, actions, reward, next_state, done, noise)  # Store experiences in the replay buffer
        state = next_state

# 4. Periodically update the agents' policies
for _ in range(num_updates):
    agent.learn()

# 5. Optionally, save and load models
agent.save_models()  # Save the current models' parameters
agent.load_models()  # Load the previously saved models' parameters
```

Please note that you need to replace `YourEnvironment` with the actual environment class you are using.

This covers the basic usage of the code for training multi-agent RL agents using Decomposed Off-Policy Gradients based on SAC. You may need to modify the code to suit your specific environment and requirements. Also, it's essential to tune hyperparameters and experiment with different configurations for optimal performance.

Sure! Here's the References section with the citation for the paper "Dop: Off-policy multi-agent decomposed policy gradients":

---

## References

Wang, Y., Han, B., Wang, T., Dong, H., & Zhang, C. (2020). Dop: Off-policy multi-agent decomposed policy gradients. In *International Conference on Learning Representations*.

---
