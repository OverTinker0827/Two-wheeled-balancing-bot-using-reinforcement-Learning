import time
import numpy as np
from env import World  # make sure the class is saved in world_env.py

def test_env():
    # Instantiate environment
    env = World(show=True)

    num_episodes = 3
    steps_per_episode = 50

    for episode in range(num_episodes):
        obs, _ = env.reset()
        print(f"\n--- Episode {episode + 1} ---")
        
        for step in range(steps_per_episode):
            action = env.action_space.sample()  # random action
            obs, reward, terminated, truncated, info = env.step(action)

            print(f"Step {step + 1} | Action: {action} | Reward: {reward:.3f} | Terminated: {terminated}")

            # Optional: slow down simulation for observation
            time.sleep(0.05)

            if terminated or truncated:
                print("Episode ended early.")
                break

    env.close()
    print("\nTest complete.")

if __name__ == "__main__":
    test_env()
