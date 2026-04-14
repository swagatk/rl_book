import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import wandb

class DynaQAgent:
    """
    An implementation of the Dyna-Q algorithm for discrete environments.
    Matches the logic described in Chapter 9 of the textbook.
    """
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.95, epsilon=0.1, n_planning_steps=10):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n_planning_steps = n_planning_steps
        
        # Initialize Q-table with zeros
        self.q_table = np.zeros((n_states, n_actions))
        
        # Initialize Model: deterministic dictionary (state, action) -> (next_state, reward)
        # We also keep track of visited state-action pairs for the planning phase
        self.model = {}
        self.observed_sa = []

    def select_action(self, state):
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        else:
            # Handle ties by picking randomly among the maximums
            max_q = np.max(self.q_table[state])
            actions_with_max_q = np.where(self.q_table[state] == max_q)[0]
            return np.random.choice(actions_with_max_q)

    def update_q(self, s, a, r, s_next):
        """Standard Q-Learning update rule."""
        target = r + self.gamma * np.max(self.q_table[s_next])
        self.q_table[s, a] += self.alpha * (target - self.q_table[s, a])

    def train_step(self, s, a, r, s_next):
        # 1. Direct RL Update
        self.update_q(s, a, r, s_next)
        
        # 2. Model Learning
        # Update model and track visited state-action pairs
        if (s, a) not in self.model:
            self.observed_sa.append((s, a))
        self.model[(s, a)] = (s_next, r)
        
        # 3. Planning Phase ("Dreaming")
        for _ in range(self.n_planning_steps):
            # Randomly select a previously observed state-action pair
            s_sim, a_sim = random.choice(self.observed_sa)
            
            # Query the model for the simulated outcome
            s_next_sim, r_sim = self.model[(s_sim, a_sim)]
            
            # Update Q-table using simulated experience
            self.update_q(s_sim, a_sim, r_sim, s_next_sim)

def run_experiment(env_name, n_episodes=200, planning_steps=[0, 5, 50]):
    """Runs Dyna-Q with different planning horizons and plots comparison."""
    results = {}
    
    for n in planning_steps:
        print(f"Training with n_planning_steps = {n}...")
        wandb.init(
            project="dyna-q-experiments",
            name=f"{env_name}_steps_{n}",
            config={"env_name": env_name, "n_planning_steps": n, "n_episodes": n_episodes}
        )
        
        env = gym.make(env_name)
        n_states = env.observation_space.n
        n_actions = env.action_space.n
        
        agent = DynaQAgent(n_states, n_actions, n_planning_steps=n)
        episode_rewards = []
        
        for ep in range(n_episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.train_step(state, action, reward, next_state)
                
                state = next_state
                total_reward += reward
            
            episode_rewards.append(total_reward)
            
            avg_reward = np.mean(episode_rewards)
            avg_reward_100 = np.mean(episode_rewards[-100:])
            
            wandb.log({
                "episode": ep,
                "total_reward": total_reward,
                "average_reward": avg_reward,
                "average_reward_100": avg_reward_100
            })
        
        results[n] = episode_rewards
        env.close()
        wandb.finish()
    
    # Visualization
    plt.figure(figsize=(10, 6))
    for n, rewards in results.items():
        # Smoothing for better visualization
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed, label=f'n = {n} steps')
        
    plt.title(f'Dyna-Q Performance on {env_name}')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward (Smoothed)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    # We use CliffWalking-v0 as it highlights the benefits of planning well
    # Note: Taxi-v3 is another great discrete environment for this.
    run_experiment('CliffWalking-v1', n_episodes=200, planning_steps=[0, 5, 50])
    #run_experiment('Taxi-v3', n_episodes=200, planning_steps=[0, 5, 50])