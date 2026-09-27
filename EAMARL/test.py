import os
import json
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import torch
import numpy as np
from tqdm import tqdm
from common.arguments import get_args
from common.utils import make_env
from agent import Agent
import matplotlib.pyplot as plt
import random

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
        return [_json_serializable(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _json_serializable(item)
            for key, item in value.items()
        }
    return str(value)

def _argument_parameters(args):
    argument_names = getattr(
        args,
        '_argument_names',
        [key for key in vars(args) if not key.startswith('_')]
    )
    return {
        name: getattr(args, name)
        for name in argument_names
    }


def _run_base_dir(args):
    return os.path.join(
        args.save_dir,
        args.scenario_name,
        args.run_subdir
    )


def _resolve_test_model_dir(args):
    checkpoint_suffix = '_final' if args.test_checkpoint == 'final' else ''
    required_files = (
        f'actor_params{checkpoint_suffix}.pkl',
        f'critic_params{checkpoint_suffix}.pkl'
    )

    def has_required_checkpoints(models_dir):
        return all(
            os.path.exists(
                os.path.join(models_dir, f'agent_{agent_idx}', filename)
            )
            for agent_idx in range(args.n_agents)
            for filename in required_files
        )

    if args.model_dir:
        candidate = os.path.expanduser(args.model_dir)
        if not os.path.isabs(candidate):
            candidate = os.path.join(args.save_dir, candidate)
        candidate = os.path.abspath(candidate)
        nested_models = os.path.join(candidate, 'models')
        if os.path.isdir(nested_models):
            candidate = nested_models
        if not has_required_checkpoints(candidate):
            raise FileNotFoundError(
                "The specified model directory does not contain complete "
                f"{args.test_checkpoint} checkpoints for all agents: {candidate}"
            )
        return candidate

    base_dir = _run_base_dir(args)
    if not os.path.isdir(base_dir):
        raise FileNotFoundError(
            f"No training run directory was found at {base_dir}. "
            "Train the model first or provide --model-dir."
        )

    result_dirs = []
    for directory_name in os.listdir(base_dir):
        if not directory_name.startswith('result'):
            continue
        suffix = directory_name.replace('result', '', 1)
        if suffix.isdigit():
            result_dirs.append((int(suffix), directory_name))

    for _, directory_name in sorted(result_dirs, reverse=True):
        candidate = os.path.join(base_dir, directory_name, 'models')
        if has_required_checkpoints(candidate):
            return os.path.abspath(candidate)

    raise FileNotFoundError(
        f"No resultN run containing complete {args.test_checkpoint} "
        f"checkpoints was found under {base_dir}. "
        "Train the model first or provide --model-dir."
    )
# [FIX-RUN-SCOPED-MODELS] END

def load_trained_agents(args, env):
    agents = []
    for agent_id in range(args.n_agents):
        agent = Agent(agent_id, args)

        model_dir = os.path.join(args.model_dir, f'agent_{agent_id}')

        checkpoint_suffix = '_final' if args.test_checkpoint == 'final' else ''
        actor_path = os.path.join(model_dir, f"actor_params{checkpoint_suffix}.pkl")
        if os.path.exists(actor_path):
            agent.policy.actor_network.load_state_dict(
                torch.load(actor_path, map_location='cpu')
            )
            print(f"Agent {agent_id} loaded actor from {actor_path}")
        else:
            raise FileNotFoundError(
                f"Actor checkpoint not found at {actor_path}. "
                "Train the model first or select --test-checkpoint latest."
            )

        critic_path = os.path.join(model_dir, f"critic_params{checkpoint_suffix}.pkl")
        if os.path.exists(critic_path):
            agent.policy.critic_network.load_state_dict(
                torch.load(critic_path, map_location='cpu')
            )
            print(f"Agent {agent_id} loaded critic from {critic_path}")

        agents.append(agent)
    return agents
def set_seeds(seed, env):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def test_policy(args, env, agents):
    episode_returns = []

    print(f"\n🚀 开始策略测试，共 {args.test_episodes} 个 episode ...")

    for ep in range(args.test_episodes):
        states = env.reset()
        episode_return = 0.0

        for step in range(args.max_episode_len):
            actions = []
            with torch.no_grad():
                for agent_id, agent in enumerate(agents):
                    action = agent.select_action(states[agent_id], 0, 0)
                    actions.append(action)

            next_states, rewards, done, _ = env.step(actions)

            step_total_reward = np.sum([float(r) for r in rewards])
            episode_return += step_total_reward

            states = next_states

            if all(done):
                break

        avg_step_reward = episode_return / args.max_episode_len
        episode_returns.append(avg_step_reward)

        print(
            f"Test Episode {ep + 1}: "
            f"Total Reward = {episode_return:.2f} | "
            f"Avg Step Reward = {avg_step_reward:.2f}"
        )

    avg_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    var_return = np.var(episode_returns)
    min_return = np.min(episode_returns)
    max_return = np.max(episode_returns)

    print("\n" + "=" * 60)
    print(f"✅ Testing Completed (consistent with evaluate())")
    print(f"  Episodes             : {args.test_episodes}")
    print(f"  Mean Avg Step Reward : {avg_return:.2f}")
    print(f"  Std Dev (σ)          : {std_return:.2f}")
    print(f"  Variance (σ²)        : {var_return:.4f}")  # ✅ 新增
    print(f"  Min Reward           : {min_return:.2f}")
    print(f"  Max Reward           : {max_return:.2f}")
    print("=" * 60)

    save_test_results(args, episode_returns, list(range(args.test_episodes)))

    return episode_returns


def save_test_results(args, returns, lengths):
    result_dir = os.path.join(
        args.save_dir,
        args.scenario_name,
        args.run_subdir,
        'test-results'
    )
    os.makedirs(result_dir, exist_ok=True)

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    result_file = os.path.join(result_dir, f"test_results_{timestamp}.npz")

    parameters = _argument_parameters(args)
    config_path = os.path.join(result_dir, f"test_config_{timestamp}.json")
    with open(config_path, mode='w', encoding='utf-8') as config_file:
        json.dump(
            _json_serializable(parameters),
            config_file,
            indent=2,
            ensure_ascii=False
        )
    print(f"Test configuration saved to: {config_path}")

    # 保存数据
    np.savez(result_file,
             returns=np.array(returns),
             lengths=np.array(lengths),
             args=parameters)

    print(f"Test results-certain saved to: {result_file}")

    plot_test_results(returns, lengths, result_dir, timestamp)


def plot_test_results(returns, lengths, save_dir, timestamp):
    plt.figure(figsize=(18, 12))
    episodes = np.arange(1, len(returns) + 1)
    returns = np.array(returns)
    lengths = np.array(lengths)

    plt.subplot(2, 3, 1)
    plt.plot(episodes, returns, 'b-', alpha=0.6, label='Episode Avg Reward')
    window = max(3, len(returns)//10)
    smooth_returns = np.convolve(returns, np.ones(window)/window, mode='valid')
    plt.plot(np.arange(1, len(smooth_returns)+1), smooth_returns, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    plt.xlabel('Episode')
    plt.ylabel('Average Step Reward')
    plt.title('Episode Rewards (with Moving Average)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 2)
    plt.hist(returns, bins=15, color='skyblue', edgecolor='black')
    plt.xlabel('Average Step Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution')
    plt.grid(True)

    plt.subplot(2, 3, 3)
    plt.boxplot(returns, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightyellow', color='orange'),
                medianprops=dict(color='red', linewidth=2))
    plt.title('Reward Boxplot')
    plt.ylabel('Average Step Reward')
    plt.grid(True, axis='y')

    plt.subplot(2, 3, 4)
    z = np.polyfit(episodes, returns, 1)
    p = np.poly1d(z)
    plt.plot(episodes, returns, 'o', alpha=0.5, label='Data')
    plt.plot(episodes, p(episodes), 'r-', linewidth=2, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    plt.xlabel('Episode')
    plt.ylabel('Average Step Reward')
    plt.title('Reward Trend (Linear Fit)')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 3, 5)
    plt.plot(episodes, lengths, 'g-o', alpha=0.6)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Episode Lengths')
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.scatter(lengths, returns, c='purple', alpha=0.7)
    plt.xlabel('Episode Length')
    plt.ylabel('Average Step Reward')
    plt.title('Reward vs Episode Length')
    plt.grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, f"test_results_plot_{timestamp}_enhanced.png")
    plt.savefig(plot_path, dpi=200)
    print(f"✅ Enhanced test results plot saved to: {plot_path}")
    plt.close()


if __name__ == '__main__':

    args = get_args()
    args.test = True
    env, args = make_env(args)
    args.model_dir = _resolve_test_model_dir(args)
    args.current_model_dir = args.model_dir
    print(f"Models will be loaded from: {args.model_dir}")
    seed = args.seed
    set_seeds(seed, env)
    agents = load_trained_agents(args, env)
    test_policy(args, env, agents)


