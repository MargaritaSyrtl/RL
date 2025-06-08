import numpy as np

rewards = np.loadtxt("results/test_results-100-len-11.txt")
best_idx = np.argmin(rewards)
best_reward = rewards[best_idx]

print(f"Best route: task №{best_idx}, time: {best_reward:.2f}")
