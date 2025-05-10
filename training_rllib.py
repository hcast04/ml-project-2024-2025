import numpy as np
import torch
from pathlib import Path
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env import ParallelPettingZooEnv
from ray.tune.registry import register_env
from ray.rllib.core.rl_module import MultiRLModuleSpec, RLModuleSpec
#from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import DefaultPPOTorchRLModule
import pettingzoo.utils.conversions
from pettingzoo.utils.env import AECEnv
from gymnasium import spaces
from pettingzoo.utils import BaseWrapper
from typing import List
from pettingzoo.butterfly import knights_archers_zombies_v10 as kaz_v10

class CustomWrapper(BaseWrapper):
    """
    Custom wrapper to preprocess observations and normalize them.
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_history = []
        self.max_history = 10000  # Limit history size
        self.obs_mean = None
        self.obs_std = None

    def observation_space(self, agent):
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent):
        obs = super().observe(agent)
        flat_obs = obs.flatten()

        # Store observations for stats calculation
        if len(self.observation_history) < self.max_history:
            self.observation_history.append(flat_obs)

        # Apply normalization if stats are available
        if self.obs_mean is not None and self.obs_std is not None:
            flat_obs = self.normalize_observation(flat_obs)
        return flat_obs

    def normalize_observation(self, obs):
        """Normalize observation using robust techniques"""
        # Ensure std doesn't have zeros
        safe_std = np.maximum(self.obs_std, 1e-6)

        # Normalize
        normalized_obs = (obs - self.obs_mean) / safe_std

        # Handle extreme values
        normalized_obs = np.nan_to_num(normalized_obs, nan=0.0, posinf=5.0, neginf=-5.0)
        normalized_obs = np.clip(normalized_obs, -10.0, 10.0)  # More permissive clipping

        return normalized_obs

    def update_obs_stats(self, force_update=False):
        """
        Update observation statistics for normalization from collected history.
        """
        if len(self.observation_history) > 100 or force_update:  # Wait for enough samples
            observations = np.array(self.observation_history)

            # Use robust statistics - percentiles instead of mean/std
            self.obs_mean = np.percentile(observations, 50, axis=0)  # median
            # Use percentile differences for scale instead of std
            p75 = np.percentile(observations, 75, axis=0)
            p25 = np.percentile(observations, 25, axis=0)
            iqr = p75 - p25
            # Fall back to std where IQR is too small
            self.obs_std = np.maximum(iqr / 1.349, np.std(observations, axis=0))

            print(f"Updated observation stats from {len(observations)} samples")
            if force_update:
                self.observation_history = []  # Clear history after forced update

    def reset(self, seed=None, options=None):
        """Add minor random variations on reset for better generalization"""
        result = super().reset(seed, options)

        # Apply small random environment variations if possible
        if hasattr(self.env, "spawn_rate"):
            # Vary spawn rate slightly each episode
            self.env.spawn_rate = np.random.randint(15, 25)

        return result

def algo_config(env_name: str, policies: List[str], policies_to_train: List[str]):
    """
    Configure the PPO algorithm with custom hyperparameters.
    """
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=True,
            enable_env_runner_and_connector_v2=True,
        )
        .environment(env=env_name, disable_env_checking=True)
        .env_runners(num_env_runners=8)
        .multi_agent(
            policies={x for x in policies},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .rl_module(
            rl_module_spec=MultiRLModuleSpec(
                rl_module_specs={
                    x: RLModuleSpec(
                        module_class=DefaultPPOTorchRLModule,
                        model_config={
                            "fcnet_hiddens": [128, 256, 128],  # Changed architecture
                            "fcnet_activation": "relu",
                            # Add dropout for regularization
                            "use_dropout": True,
                            "dropout_rate": 0.1,
                            "use_l2_regularization": True,
                            "l2_reg_weight": 1e-5
                        })
                    if x in policies_to_train
                    else RLModuleSpec(module_class=RandomRLModule)
                    for x in policies
                },
            )
        )
        .training(
            train_batch_size=16384,  # Larger batch for more stable training
            lr=2e-4,  # Slightly lower learning rate
            gamma=0.995,  # Higher discount factor for long-term rewards
            lambda_=0.95,  # GAE parameter
            clip_param=0.2,  # Standard PPO clipping
            kl_coeff=0.01,  # Small KL penalty to prevent policy collapse
            num_sgd_iter=8,  # More SGD iterations
            minibatch_size=512,  # Balanced minibatch size
            entropy_coeff=0.02,  # Slightly higher entropy for exploration
            vf_loss_coeff=0.8,  # Balance value function learning

        )
        .debugging(log_level="ERROR")
        .framework("torch")
    )
    return config


def training(env: AECEnv, checkpoint_path: str, max_iterations: int = 2000):
    """
    Train the RL agents using PPO in a multi-agent environment.
    """
    # Convert the PettingZoo environment to an RLLib-compatible environment
    rllib_env = ParallelPettingZooEnv(pettingzoo.utils.conversions.aec_to_parallel(env))
    env_name = "knights_archers_zombies_v10"
    register_env(env_name, lambda config: rllib_env)

    # Fix seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    # Define the configuration for the PPO algorithm
    policies = [x for x in env.agents]
    policies_to_train = policies
    config = algo_config(env_name, policies, policies_to_train)

    # Add better learning rate scheduling
    #config.training(lr=[[0, 2e-4], [max_iterations * 0.5, 1e-4], [max_iterations * 0.75, 5e-5]])

    best_reward = 0
    patience = 200
    patience_counter = 0
    update_frequency = 5

    # Train the model
    algo = config.build()

    env.update_obs_stats(force_update=True)  # Force update stats at the beginning

    for i in range(max_iterations):
        # Periodically update observation stats for better normalization
        if i % update_frequency == 0:
            env.update_obs_stats()

        result = algo.train()
        # result.pop("config")
        if "env_runners" in result and "agent_episode_returns_mean" in result["env_runners"]:
            #print(i, result["env_runners"]["agent_episode_returns_mean"])
            #if result["env_runners"]["agent_episode_returns_mean"]["archer_0"] > 10:  # Example stopping criterion
            #    break

            current_reward = result["env_runners"]["agent_episode_returns_mean"]["archer_0"]
            print(i, current_reward)

            if current_reward > best_reward:
                best_reward = current_reward
                patience_counter = 0
                # Save this as the best checkpoint
                save_result = algo.save(checkpoint_path)

                # Record the best checkpoint path
                best_checkpoint_file = Path(checkpoint_path) / "best_checkpoint.txt"
                with open(best_checkpoint_file, "w") as f:
                    f.write(str(save_result.checkpoint.path))


                # Also save metrics for this checkpoint
                metrics_file = Path(save_result.checkpoint.path) / "metrics.json"
                with open(metrics_file, "w") as f:
                    import json
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj: any) -> any:
                            if isinstance(obj, np.integer):
                                return int(obj)
                            elif isinstance(obj, np.floating):
                                return float(obj)
                            elif isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, Path):
                                return str(obj)
                            try:
                                return super().default(obj)
                            except TypeError:
                                return str(obj)  # Convert non-serializable objects to string

                    json.dump(result, f, cls=NumpyEncoder)
                    # Also save observation statistics immediately with the best checkpoint
                    stats_path = Path(checkpoint_path) / "obs_stats.npz"
                    np.savez(stats_path, mean=env.obs_mean, std=env.obs_std)
                    print(f"Saved observation statistics with best model (reward: {current_reward:.2f})")

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping after {patience} iterations without improvement")
                    break
        if i % 10 == 0:
            save_result = algo.save(checkpoint_path)
            path_to_checkpoint = save_result.checkpoint.path
            print(
                "An Algorithm checkpoint has been created inside directory: "
                f"'{path_to_checkpoint}'."
            )
            stats_path = Path(checkpoint_path) / "obs_stats.npz"
            np.savez(stats_path, mean=env.obs_mean, std=env.obs_std)

    # Save observation statistics for later use in submission_single.py
    if hasattr(env, "obs_mean") and hasattr(env, "obs_std") and env.obs_mean is not None and env.obs_std is not None:
        stats_path = Path(checkpoint_path) / "obs_stats.npz"
        np.savez(stats_path, mean=env.obs_mean, std=env.obs_std)
        print(f"Saved observation statistics to {stats_path}")
    else:
        print("Warning: Could not save observation statistics - env.obs_mean or env.obs_std is None")

"""
def create_environment(num_agents=1, visual_observation=False, rand=True):
    # Original environment creation logic
    env = kaz_v10.env(
        num_archers=num_agents,
        num_knights=0,
        spawn_rate=np.random.randint(15, 25) if rand else 20,
        max_cycles=2000,
        #seed=None,  # Use None instead of fixed seed for randomization
        use_typemasks=False,
        vector_state=not visual_observation
    )

    # Add randomization if enabled
    if rand:
        # Set random seed (if the env supports it) instead of in constructor
        if hasattr(env, 'seed'):
            env.seed(None)  # Use None for random seed

        # Randomize spawn rates slightly for better generalization
        if hasattr(env, 'spawn_rate'):
            variation = np.random.uniform(0.7, 1.3)
            env.spawn_rate = int(20 * variation)  # Base on the original spawn_rate of 20

        # Randomize enemy movement patterns slightly
        if hasattr(env, 'enemy_speed'):
            env.enemy_speed *= np.random.uniform(0.8, 1.2)

        # Randomize archer starting positions if applicable
        if hasattr(env, 'archer_pos'):
            for i in range(len(env.archer_pos)):
                # Position archers in more advantageous starting positions
                x = np.random.randint(env.WIDTH // 5, 4 * env.WIDTH // 5)
                y = np.random.randint(env.HEIGHT // 5, 4 * env.HEIGHT // 5)
                env.archer_pos[i] = (x, y)

    return env
"""


if __name__ == "__main__":
    from utils import create_environment  # Replace with your actual environment creation function

    num_agents = 1
    visual_observation = False
    rand = False

    # Create the PettingZoo environment for training
    env = create_environment(num_agents=num_agents, visual_observation=visual_observation)
    env = CustomWrapper(env)

    # Run the training routine
    checkpoint_path = str(Path("results").resolve())
    training(env, checkpoint_path, max_iterations=2000)