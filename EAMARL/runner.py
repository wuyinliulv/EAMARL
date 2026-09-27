import csv
import json
from datetime import datetime
import copy
from tqdm import tqdm
from agent import Agent
from common.replay_buffer import Buffer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import random
import time
from ea.PSO.PSO_manager import PSOManager  # 修改为PSO管理器
from contextlib import contextmanager

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# print("Using device:", device)
class Runner:
    def __init__(self, args, env):
        self.args = args
        self.noise = args.noise_rate
        self.epsilon = args.epsilon
        self.episode_limit = args.max_episode_len
        self.env = env
        self.eval_env = copy.deepcopy(env)
        # self.device = device
        self._set_seeds(args.seed)
        self.evolve_interval = args.evolve_interval
        if self.evolve_interval < 1:
            raise ValueError("evolve_interval must be at least 1")
        if args.pso_eval_episodes < 1 or args.pso_gate_episodes < 1:
            raise ValueError("PSO evaluation episode counts must be at least 1")
        if args.pso_inject_threshold < 0:
            raise ValueError("pso_inject_threshold must be non-negative")
        if args.evaluate_rate < 1:
            raise ValueError("evaluate_rate must be at least 1")
        # Selection, injection gating, and routine evaluation must not consume
        # one another's random streams.  This keeps routine test scenarios
        # paired when population size or evolution frequency changes.
        self.pso_seed_rng = np.random.RandomState(args.seed + 10000)
        self.gate_seed_rng = np.random.RandomState(args.seed + 20000)
        self.eval_seed_rng = np.random.RandomState(args.seed + 30000)
        self.base_dir = os.path.join(
            self.args.save_dir,
            self.args.scenario_name,
            self.args.run_subdir
        )
        os.makedirs(self.base_dir, exist_ok=True)

        next_index = self._get_next_run_index()

        if self.args.resume or self.args.evaluate:
            self.args.model_dir = self._resolve_load_model_dir()

        self.run_index = next_index
        self.save_path = os.path.join(self.base_dir, f'result{next_index}')
        os.makedirs(self.save_path, exist_ok=True)
        self.args.current_model_dir = os.path.join(self.save_path, 'models')
        os.makedirs(self.args.current_model_dir, exist_ok=True)
        print(f"Results will be saved to: {self.save_path} (Run #{next_index}, Seed: {args.seed})")
        print(f"Models will be saved to: {self.args.current_model_dir}")

        self.agents = self._init_agents()
        self.buffer = Buffer(args)
        self.pso_manager = PSOManager(args, self.agents)

        self.returns = []
        self.evaluation_steps = []
        self.pso_opportunities = 0
        self.pso_injections = 0
        self.training_elapsed_seconds = None
        self._save_hyperparams()
        self._init_reward_csv()
        self._init_evaluation_csv()
        self._init_pso_csv()

    @staticmethod
    def _json_serializable(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (list, tuple)):
            return [Runner._json_serializable(item) for item in value]
        if isinstance(value, dict):
            return {
                str(key): Runner._json_serializable(item)
                for key, item in value.items()
            }
        return str(value)

    def _save_hyperparams(self):
        argument_names = getattr(
            self.args,
            '_argument_names',
            [key for key in vars(self.args) if not key.startswith('_')]
        )
        parameters = {
            name: getattr(self.args, name)
            for name in argument_names
        }

        config_path = os.path.join(self.save_path, 'hyperparams.json')
        with open(config_path, mode='w', encoding='utf-8') as config_file:
            json.dump(
                self._json_serializable(parameters),
                config_file,
                indent=2,
                ensure_ascii=False
            )
        print(f"Run configuration saved to: {config_path}")
    # [FIX-RUN-METADATA] END

    def _init_reward_csv(self):
        self.reward_csv_path = os.path.join(self.save_path, 'training_rewards.csv')
        self.episode_rewards = []

        with open(self.reward_csv_path, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Episode', 'Total Reward', 'Average Reward', 'Timestamp'])

    def _record_episode_reward(self, episode_idx, total_reward, avg_reward):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.reward_csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode_idx, total_reward, avg_reward, timestamp])

        self.episode_rewards.append({
            'episode': episode_idx,
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'timestamp': timestamp
        })

    def _init_evaluation_csv(self):
        """Initialize machine-readable routine evaluation history."""
        self.evaluation_csv_path = os.path.join(self.save_path, 'evaluations.csv')
        with open(self.evaluation_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Train Step', 'Mean Team Step Reward', 'Evaluation Episodes'
            ])

    def _record_evaluation(self, train_step, mean_return, episode_count):
        with open(self.evaluation_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([train_step, mean_return, episode_count])

    def _init_pso_csv(self):
        """Initialize one-row-per-gate-decision PSO audit log."""
        self.pso_csv_path = os.path.join(self.save_path, 'pso_injections.csv')
        with open(self.pso_csv_path, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Train Step', 'Seed', 'Threshold', 'Swarm Size',
                'Evolve Interval', 'Selection Baseline',
                'Selection Candidate', 'Gate Baseline', 'Gate Candidate',
                'Relative Improvement', 'Candidate Agent', 'Injected',
                'Selection Episodes', 'Gate Episodes'
            ])

    def _record_pso_event(self, train_step, selection_baseline, event):
        with open(self.pso_csv_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                train_step,
                self.args.seed,
                self.args.pso_inject_threshold,
                self.args.pso_swarm_size,
                self.args.evolve_interval,
                selection_baseline,
                event['selection_fitness'],
                event['baseline_fitness'],
                event['candidate_fitness'],
                event['relative_improvement'],
                event['agent_idx'],
                int(event['injected']),
                self.args.pso_eval_episodes,
                self.args.pso_gate_episodes,
            ])

    def _set_seeds(self, seed):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

    @contextmanager
    def _preserve_random_state(self):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @staticmethod
    def _draw_seeds(rng, count):
        return rng.randint(0, 2 ** 31 - 1, size=count).tolist()

    def _draw_pso_seeds(self, count):
        """Scenarios used only for PSO candidate selection."""
        return self._draw_seeds(self.pso_seed_rng, count)

    def _draw_gate_seeds(self, count):
        """Held-out scenarios used only by the injection gate."""
        return self._draw_seeds(self.gate_seed_rng, count)

    def _draw_evaluation_seeds(self, count):
        """Scenarios used only for reported routine evaluations."""
        return self._draw_seeds(self.eval_seed_rng, count)

    @staticmethod
    def _reward_scalar(value):
        return float(np.asarray(value, dtype=np.float64).reshape(-1)[0])

    @staticmethod
    def _actor_action(actor, observation):
        inputs = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor(inputs).squeeze(0).cpu().numpy()
        return np.clip(action, 0.0, 1.0)
    # [FIX-EVAL-ISOLATION] END

    def _get_next_run_index(self):
        existing_dirs = [d for d in os.listdir(self.base_dir)
                         if os.path.isdir(os.path.join(self.base_dir, d))]

        result_dirs = [d for d in existing_dirs if d.startswith('result')]

        if not result_dirs:
            return 1

        indices = []
        for d in result_dirs:
            try:
                num_str = d.replace('result', '')
                if '_' in num_str:
                    num_str = num_str.split('_')[0]
                if num_str.isdigit():
                    indices.append(int(num_str))
            except ValueError:
                continue

        return max(indices) + 1 if indices else 1

    def _resolve_load_model_dir(self):
        def has_latest_checkpoints(models_dir):
            return all(
                os.path.exists(os.path.join(models_dir, f'agent_{agent_idx}', filename))
                for agent_idx in range(self.args.n_agents)
                for filename in ('actor_params.pkl', 'critic_params.pkl')
            )

        if self.args.model_dir:
            candidate = os.path.expanduser(self.args.model_dir)
            if not os.path.isabs(candidate):
                candidate = os.path.join(self.args.save_dir, candidate)
            candidate = os.path.abspath(candidate)
            nested_models = os.path.join(candidate, 'models')
            if os.path.isdir(nested_models):
                candidate = nested_models
            if not has_latest_checkpoints(candidate):
                raise FileNotFoundError(
                    f"The specified model directory does not contain complete latest checkpoints: {candidate}"
                )
            return candidate

        result_dirs = []
        for directory_name in os.listdir(self.base_dir):
            if not directory_name.startswith('result'):
                continue
            suffix = directory_name.replace('result', '', 1)
            if suffix.isdigit():
                result_dirs.append((int(suffix), directory_name))

        for _, directory_name in sorted(result_dirs, reverse=True):
            candidate = os.path.join(self.base_dir, directory_name, 'models')
            if has_latest_checkpoints(candidate):
                return os.path.abspath(candidate)

        raise FileNotFoundError(
            f"No completed run checkpoint was found under {self.base_dir}. "
            "Run fresh training first or provide --model-dir."
        )

    def _init_agents(self):
        agents = []
        for i in range(self.args.n_agents):
            agent = Agent(i, self.args)
            agents.append(agent)
        return agents

    def run(self):
        training_started_at = time.perf_counter()
        step_rewards = []
        episode_idx = 0
        # self.pso_manager.evaluate_particles(self)
        for time_step in tqdm(range(self.args.time_steps)):
            # reset the environment
            if time_step % self.episode_limit == 0:
                s = self.env.reset()
                if step_rewards:
                    episode_rewards = step_rewards[-self.episode_limit:]
                    total_reward = np.sum(episode_rewards)
                    avg_reward = np.mean(episode_rewards)
                    self._record_episode_reward(episode_idx, total_reward, avg_reward)

                    tqdm.write(
                        f"Episode {episode_idx} | Total Reward: {total_reward:.2f} | Avg Reward: {avg_reward:.2f} | Steps: {time_step}")
                    episode_idx += 1

            u = []
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(self.agents):
                    action = agent.select_action(s[agent_id], self.noise, self.epsilon)
                    u.append(action)
                    actions.append(action)

            s_next, r, done, info = self.env.step(actions)
            self.buffer.store_episode(
                s[:self.args.n_agents],
                u,
                r[:self.args.n_agents],
                s_next[:self.args.n_agents],
                done[:self.args.n_agents]
            )
            s = s_next

            team_step_reward = sum(
                self._reward_scalar(r_i)
                for r_i in r[:self.args.n_agents]
            )
            step_rewards.append(team_step_reward)

            if self.buffer.current_size >= self.args.batch_size:
                transitions = self.buffer.sample(self.args.batch_size)
                for agent in self.agents:
                    other_agents = self.agents.copy()
                    other_agents.remove(agent)
                    agent.learn(transitions, other_agents)

            pso_ready = self.buffer.current_size >= self.args.batch_size
            if pso_ready and (time_step + 1) % self.evolve_interval == 0:
                self._pso_step(time_step + 1)

            # Evaluate at exact completed-step boundaries.  With an episode
            # length of 52, --evaluate-rate 52 now evaluates after every full
            # episode rather than after the first step of the next episode.
            if (time_step + 1) % self.args.evaluate_rate == 0:
                avg_return = self.evaluate()
                # 维度处理
                if isinstance(avg_return, np.ndarray):
                    avg_return = avg_return.item()

                self.returns.append(avg_return)
                self.evaluation_steps.append(time_step + 1)
                self._record_evaluation(
                    time_step + 1,
                    avg_return,
                    self.args.evaluate_episodes
                )
                self._save_progress(time_step + 1)
            self.noise = max(0.05, self.noise * 0.9995)
            self.epsilon = max(0.05, self.epsilon * 0.9995)
        # The last episode has no subsequent reset at which the legacy logger
        # could flush it, so persist it explicitly here.
        if step_rewards:
            final_episode_steps = self.args.time_steps % self.episode_limit
            final_episode_steps = final_episode_steps or self.episode_limit
            final_episode_rewards = step_rewards[-final_episode_steps:]
            self._record_episode_reward(
                episode_idx,
                np.sum(final_episode_rewards),
                np.mean(final_episode_rewards)
            )

        # Always report the final checkpoint on the isolated routine-evaluation
        # stream, rather than treating the last periodic evaluation as final.
        if self.evaluation_steps and self.evaluation_steps[-1] == self.args.time_steps:
            final_return = self.returns[-1]
        else:
            final_return = self.evaluate()
            self.returns.append(final_return)
            self.evaluation_steps.append(self.args.time_steps)
            self._record_evaluation(
                self.args.time_steps,
                final_return,
                self.args.evaluate_episodes
            )
        self.training_elapsed_seconds = time.perf_counter() - training_started_at

        self._save_final_models()
        self._save_final_results()
        # self._save_models(self.args.time_steps)

        return self.returns

    def _evaluate_team_policy(self, scenario_seeds, agent_idx=None,
                              policy_state=None, net_template=None,
                              episode_len=None):
        evaluation_len = episode_len or self.episode_limit
        candidate_actor = None
        if agent_idx is not None:
            candidate_actor = copy.deepcopy(net_template)
            candidate_actor.load_state_dict(policy_state)
            candidate_actor.eval()

        episode_scores = []
        with self._preserve_random_state():
            for seed in scenario_seeds:
                random.seed(int(seed))
                np.random.seed(int(seed))
                torch.manual_seed(int(seed))

                states = self.eval_env.reset()
                episode_team_return = 0.0

                for _ in range(evaluation_len):
                    actions = []
                    for current_agent_idx, agent in enumerate(self.agents):
                        actor = (
                            candidate_actor
                            if current_agent_idx == agent_idx
                            else agent.policy.actor_network
                        )
                        actions.append(
                            self._actor_action(actor, states[current_agent_idx])
                        )

                    next_states, rewards, dones, _ = self.eval_env.step(actions)
                    episode_team_return += sum(
                        self._reward_scalar(reward)
                        for reward in rewards[:self.args.n_agents]
                    )
                    states = next_states
                    if np.all(dones[:self.args.n_agents]):
                        break

                episode_scores.append(episode_team_return / evaluation_len)

        return float(np.mean(episode_scores))

    def _pso_step(self, current_time_step):

        scenario_seeds = self._draw_pso_seeds(self.args.pso_eval_episodes)
        current_performance = self._evaluate_team_policy(scenario_seeds)
        self.pso_manager.prepare_evolution_round()
        self.pso_manager.evaluate_particles(self, scenario_seeds)

        self.pso_manager.update_particles(current_time_step)
        candidate = self.pso_manager.get_best_candidate()

        # The selected maximum is re-evaluated on held-out gate scenarios.
        # This makes swarm-size comparisons much less vulnerable to winner's
        # curse from selection noise.
        gate_seeds = self._draw_gate_seeds(self.args.pso_gate_episodes)
        gate_current_performance = self._evaluate_team_policy(gate_seeds)
        gate_candidate_performance = None
        if candidate is not None:
            agent_idx = candidate['agent_idx']
            gate_candidate_performance = self._evaluate_individual(
                agent_idx,
                candidate['policy_state'],
                self.pso_manager.swarms[agent_idx]['net_template'],
                gate_seeds
            )

        event = self.pso_manager.inject_best_policy(
            gate_current_performance,
            self.args.pso_inject_threshold,
            candidate=candidate,
            candidate_performance=gate_candidate_performance
        )
        self.pso_opportunities += 1
        if event['injected']:
            self.pso_injections += 1
        self._record_pso_event(current_time_step, current_performance, event)

    def _evaluate_individual(self, agent_idx, policy_state, net_template,
                             scenario_seeds):

        return self._evaluate_team_policy(
            scenario_seeds,
            agent_idx=agent_idx,
            policy_state=policy_state,
            net_template=net_template,
            episode_len=self.episode_limit
        )

    def evaluate(self):
        scenario_seeds = self._draw_evaluation_seeds(self.args.evaluate_episodes)
        avg_return = self._evaluate_team_policy(
            scenario_seeds,
            episode_len=self.args.evaluate_episode_len
        )
        print(
            f"Evaluation Completed: "
            f"Mean Team Step Reward = {avg_return:.2f} "
            f"(Avg over {len(scenario_seeds)} episodes)"
        )
        return avg_return
    # [FIX-METRIC-UNIFICATION] END

    def _save_final_models(self):
        for agent in self.agents:
            agent.policy.save_model("final")
    def _save_progress(self, time_step):
        np.save(os.path.join(self.save_path, f'returns_step_{time_step}.npy'), self.returns)

        plt.figure(figsize=(10, 6))
        plt.plot(self.evaluation_steps, self.returns, 'b-o', lw=2)
        plt.xlabel('Training Environment Steps')
        plt.ylabel('Mean Team Step Reward')
        plt.title(f'Training Progress - Run #{self.run_index} (Time Step: {time_step})')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, f"progress_{time_step}.png"))
        plt.close()

    def _save_models(self, time_step):
        model_dir = os.path.join(self.save_path, f"models_step_{time_step}")
        os.makedirs(model_dir, exist_ok=True)

        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(),
                       os.path.join(model_dir, f"agent{i}_actor.pt"))
            torch.save(agent.critic.state_dict(),
                       os.path.join(model_dir, f"agent{i}_critic.pt"))
        print(f"Models saved at step: {time_step}")

    def _save_final_results(self):
        np.save(os.path.join(self.save_path, 'final_returns.npy'), self.returns)

        plt.figure(figsize=(10, 6))
        plt.plot(self.evaluation_steps, self.returns, 'b-o', lw=2)
        plt.xlabel('Training Environment Steps')
        plt.ylabel('Mean Team Step Reward')
        plt.title(f'Final Training Progress - Run #{self.run_index}')
        plt.grid(True)
        plt.savefig(os.path.join(self.save_path, "final_progress.png"))
        plt.close()

        if len(self.returns) >= 2:
            step_span = self.evaluation_steps[-1] - self.evaluation_steps[0]
            normalized_auc = (
                float(np.trapz(self.returns, self.evaluation_steps) / step_span)
                if step_span > 0 else float(self.returns[-1])
            )
        elif self.returns:
            normalized_auc = float(self.returns[-1])
        else:
            normalized_auc = float('nan')

        acceptance_rate = (
            self.pso_injections / self.pso_opportunities
            if self.pso_opportunities else 0.0
        )
        summary = {
            'seed': self.args.seed,
            'pso_inject_threshold': self.args.pso_inject_threshold,
            'pso_swarm_size': self.args.pso_swarm_size,
            'evolve_interval': self.args.evolve_interval,
            'time_steps': self.args.time_steps,
            'evaluate_episodes': self.args.evaluate_episodes,
            'final_mean_team_step_reward': (
                float(self.returns[-1]) if self.returns else None
            ),
            'normalized_learning_curve_auc': normalized_auc,
            'pso_opportunities': self.pso_opportunities,
            'pso_injections': self.pso_injections,
            'injection_acceptance_rate': acceptance_rate,
            'training_elapsed_seconds': self.training_elapsed_seconds,
        }
        with open(
            os.path.join(self.save_path, 'summary.json'),
            mode='w',
            encoding='utf-8'
        ) as summary_file:
            json.dump(summary, summary_file, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {self.save_path}")
