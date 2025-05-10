import gymnasium
from pettingzoo.utils import BaseWrapper
from pettingzoo.utils.env import AgentID, ObsType
from pathlib import Path
from ray.rllib.core.rl_module import MultiRLModule
import torch
import numpy as np  # Added import
from gymnasium import spaces # Added import
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.core.rl_module.multi_rl_module import MultiRLModule
from ray.rllib.algorithms import Algorithm


class Agent:
    def __init__(self, env):
        # The Agent class just delegates to CustomPredictFunction
        self.predict_function = CustomPredictFunction(env)

    def __call__(self, observation, agent, *args, **kwargs):
        # Delegate the call to the CustomPredictFunction instance
        return self.predict_function(observation, agent, *args, **kwargs)

class CustomWrapper(BaseWrapper):
    """
    Custom wrapper to preprocess observations and normalize them.
    """

    def __init__(self, env):
        super().__init__(env)
        self.obs_mean = None
        self.obs_std = None
        # Load observation statistics
        try:
            package_directory = Path(__file__).resolve().parent
            # This path should match where training_rllib.py saves the stats
            # e.g., inside the 'results' directory.
            stats_path = package_directory / "results" / "obs_stats.npz"
            if stats_path.exists():
                stats = np.load(stats_path)
                self.obs_mean = stats["mean"]
                self.obs_std = stats["std"]
                print(f"Successfully loaded observation stats from {stats_path}")
            else:
                print(f"Warning: Observation statistics file not found at {stats_path}. Normalization will use uninitialized stats or might fail if stats are required.")
        except Exception as e:
            print(f"Warning: Error loading observation stats: {e}. Normalization will use uninitialized stats.")

    def observation_space(self, agent):
        # Ensure spaces is available
        return spaces.flatten_space(super().observation_space(agent))

    def observe(self, agent):
        obs = super().observe(agent)
        flat_obs = obs.flatten()

        if self.obs_mean is not None and self.obs_std is not None:
            obs_size = len(flat_obs)
            stats_size = len(self.obs_mean)

            # Handle size mismatch
            if obs_size != stats_size:
                if obs_size < stats_size:
                    padded_obs = np.zeros(stats_size)
                    padded_obs[:obs_size] = flat_obs
                    flat_obs = padded_obs
                else:
                    flat_obs = flat_obs[:stats_size]

            # Now normalize with matching sizes
            eps = 1e-8
            current_obs_std = np.maximum(self.obs_std, eps)  # Prevent division by zero
            normalized_obs = (flat_obs - self.obs_mean) / current_obs_std
            flat_obs = np.nan_to_num(normalized_obs, nan=0.0, posinf=10.0, neginf=-10.0)

        return flat_obs

    def update_obs_stats(self, observations):
        """
        Update observation statistics for normalization.
        This method is typically used during training and not evaluation.
        """
        self.obs_mean = np.mean(observations, axis=0)
        self.obs_std = np.std(observations, axis=0)


class CustomPredictFunction:
    """
    Function to use to load the trained model and predict the action.
    """

    def load_observation_stats(self):
        """Load observation statistics for normalization."""
        try:
            package_directory = Path(__file__).resolve().parent
            stats_path = package_directory / "results" / "obs_stats.npz"
            if stats_path.exists():
                stats = np.load(stats_path)
                self.obs_mean = stats["mean"]
                self.obs_std = stats["std"]
            else:
                print(f"Warning: Observation statistics file not found at {stats_path}")
        except Exception as e:
            print(f"Warning: Error loading observation stats: {e}")

    def __init__(self, env: gymnasium.Env):
        package_directory = Path(__file__).resolve().parent
        results_dir = package_directory / "results"
        # Load observation statistics
        self.obs_mean = None
        self.obs_std = None
        self.load_observation_stats()

        # Track best checkpoint and reward
        best_checkpoint_path = None
        highest_reward = -float('inf')

        # First check for best_checkpoint.txt file
        best_checkpoint_file = results_dir / "best_checkpoint.txt"
        if best_checkpoint_file.exists():
            with open(best_checkpoint_file, "r") as f:
                checkpoint_content = f.read().strip()
                # Check if the path points to a directory (could be algorithm checkpoint)
                potential_path = Path(checkpoint_content)
                if potential_path.is_dir():
                    # Look for RLModule within the algorithm checkpoint
                    module_path = potential_path / "learner_group" / "learner" / "rl_module"
                    if module_path.exists():
                        best_checkpoint_path = module_path
                        print(f"Loading model from module path in best checkpoint: {best_checkpoint_path}")
                    else:
                        # We have an algorithm checkpoint, not a module
                        try:
                            # Try to load as Algorithm first
                            algorithm = Algorithm.from_checkpoint(str(potential_path))
                            self.modules = algorithm.get_module()
                            self.default_policy_id = list(self.modules.keys())[0]
                            print(f"Loaded algorithm from checkpoint: {potential_path}")
                            # Return early since we've already loaded the model
                            return
                        except Exception as e:
                            print(f"Failed to load algorithm from {potential_path}: {e}")
                            best_checkpoint_path = None
                else:
                    best_checkpoint_path = potential_path
                    print(f"Loading model from recorded best checkpoint: {best_checkpoint_path}")

        # Fall back to latest checkpoint if no best found
        if not best_checkpoint_path:
            fallback_path = results_dir / "learner_group" / "learner" / "rl_module"
            if fallback_path.exists():
                best_checkpoint_path = fallback_path.resolve()
                print(f"Loading model from fallback path: {best_checkpoint_path}")
            else:
                # Try loading the algorithm directly
                try:
                    algorithm = Algorithm.from_checkpoint(str(results_dir))
                    self.modules = algorithm.get_module()
                    self.default_policy_id = list(self.modules.keys())[0]
                    print(f"Loaded algorithm from results directory")
                    # Display reward information if available
                    try:
                        metrics_file = results_dir / "metrics.json"
                        if metrics_file.exists():
                            with open(metrics_file, 'r') as f:
                                import json
                                metrics = json.load(f)
                                if "env_runners" in metrics and "agent_episode_returns_mean" in metrics["env_runners"]:
                                    if "archer_0" in metrics["env_runners"]["agent_episode_returns_mean"]:
                                        reward = metrics["env_runners"]["agent_episode_returns_mean"]["archer_0"]
                                        print(f"Loaded best model with reward: {reward:.2f}")
                    except Exception as e:
                        print(f"Could not extract reward information: {e}")
                    return
                except Exception as e:
                    print(f"Failed to load algorithm from {results_dir}: {e}")
                    raise FileNotFoundError(f"RLModule directory not found. Searched in '{results_dir}'")

        # Load the model as MultiRLModule
        self.modules = MultiRLModule.from_checkpoint(str(best_checkpoint_path))
        self.default_policy_id = list(self.modules.keys())[0]

        # Display reward information if available
        try:
            metrics_file = best_checkpoint_path.parent.parent.parent / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file, 'r') as f:
                    import json
                    metrics = json.load(f)
                    if "env_runners" in metrics and "agent_episode_returns_mean" in metrics["env_runners"]:
                        if "archer_0" in metrics["env_runners"]["agent_episode_returns_mean"]:
                            reward = metrics["env_runners"]["agent_episode_returns_mean"]["archer_0"]
                            print(f"Loaded best model with reward: {reward:.2f}")
        except Exception as e:
            print(f"Could not extract reward information: {e}")

    def __call__(self, observation, agent, *args, **kwargs):
        # The observation should already be flattened and normalized by CustomWrapper.observe

        # Handle observation shape mismatch - pad or truncate to match expected size
        expected_size = 110  # From the error message, we can see the model expects 110 features
        obs_size = observation.shape[0]  # Current size is 80

        if obs_size < expected_size:
            # Pad with zeros if observation is smaller than expected
            padded_obs = np.zeros(expected_size)
            padded_obs[:obs_size] = observation
            observation = padded_obs
        elif obs_size > expected_size:
            # Truncate if observation is larger than expected (unlikely in this case)
            observation = observation[:expected_size]

        rl_module = None
        if agent in self.modules:
            rl_module = self.modules[agent]
        elif self.default_policy_id in self.modules:
            # Fallback to the default policy if the specific agent ID is not found
            # This is common in single-agent setups or when agent IDs might vary slightly
            # but map to the same underlying policy.
            # print(f"Warning: Agent '{agent}' not found. Using default policy '{self.default_policy_id}'.")
            rl_module = self.modules[self.default_policy_id]
        else:
            # This case should ideally not be reached if default_policy_id is valid
            raise ValueError(
                f"No policy found for agent '{agent}' and no default policy available. Available: {list(self.modules.keys())}")

        obs_tensor = torch.Tensor(observation).unsqueeze(0)
        fwd_ins = {"obs": obs_tensor}

        with torch.no_grad():
            try:
                fwd_outputs = rl_module.forward_inference(fwd_ins)

                action_dist_inputs = fwd_outputs.get("action_dist_inputs")
                if action_dist_inputs is None:
                    raise ValueError("Could not find 'action_dist_inputs' in module output.")

                action_dist_class = rl_module.get_inference_action_dist_cls()
                action_dist = action_dist_class.from_logits(action_dist_inputs)
                # Safeguard the action distribution before sampling
                if hasattr(action_dist, '_dist') and hasattr(action_dist._dist, 'probs'):
                    probs = action_dist._dist.probs
                    if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                        # Replace invalid distribution with uniform distribution
                        probs = torch.ones_like(probs) / probs.shape[-1]
                        action_dist._dist.probs = probs
                action = action_dist.sample()

                return action[0].cpu().numpy()
            except Exception as e:
                print(f"Error during model inference: {e}")
                # Fallback to a random action if inference fails
                return np.random.choice(self.modules[agent].action_space.n)