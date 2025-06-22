import numpy as np
import gymnasium as gym
from gymnasium import spaces
import genesis as gs

class World(gym.Env):
    def __init__(self, show=False):
        super(World, self).__init__()

        # Initialize Genesis simulation
        gs.init(backend=gs.cpu)
        self.scene = gs.Scene(show_viewer=show)

        # Add ground plane
        self.scene.add_entity(gs.morphs.Plane())

        # Load robot URDF
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(file="/home/windowsuser/rl_bb/bot.urdf")
        )

        # Assuming 2 continuous wheel actions: [left_wheel_vel, right_wheel_vel]
        max_vel = 10.0
        self.action_space = spaces.Box(
            low=-max_vel, high=max_vel, shape=(2,), dtype=np.float32
        )

        # Observation includes position, velocity, orientation, angular vel, wheel vels, targets
        obs_dim = 16  # adjust if needed
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def get_obs(self):
        # Get position (x, y, z)
        pos = self.robot.get_position()

        # Get linear velocity (vx, vy, vz)
        lin_vel = self.robot.get_linear_velocity()

        # Get orientation (yaw, pitch, roll)
        ori = self.robot.get_orientation_euler()

        # Get angular velocity (roll_rate, pitch_rate, yaw_rate)
        self.ang_vel = self.robot.get_angular_velocity()

        # Get wheel velocities (assuming 2 joints)
        wheel_vels = self.robot.get_joint_velocities()

      

        # Combine everythingnp.array([0.0])
        obs = np.concatenate([
            pos, lin_vel, ori, self.ang_vel, wheel_vels, self.target_lin_vel, self.target_ang_vel
        ])

        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Reset the scene and robot
        self.scene.reset()
        self.robot.set_joint_positions([0.0, 0.0])      # Left and right wheel
        self.robot.set_joint_velocities([0.0, 0.0])
          # Example: target linear and angular velocity (placeholder values)
        self.target_lin_vel = 1
        self.target_ang_vel = 1
        # Step once to apply reset
        self.scene.step()

        obs = self.get_obs()
        return obs, {}

    def step(self, action):
        # Apply action: set joint targets (velocities)
        self.robot.set_joint_targets(action.tolist())

        # Step the simulation
        self.scene.step()

        # Get next observation
        obs = self.get_obs()

        # Example reward: negative L2 distance from upright
        pitch = obs[5]  # pitch angle from orientation
        reward = -abs(pitch)-abs(self.target_ang_vel-self.ang_vel)  # penalize falling over

        # Termination condition: fall threshold
        terminated = abs(pitch) > 1.0  # e.g., fallen over
        truncated = False

        return obs, reward, terminated, truncated, {}

    def close(self):
        self.scene.close()
