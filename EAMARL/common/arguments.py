import argparse
import os

"""
Here are the param for the training

"""

def get_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Environment
    parser.add_argument("--scenario-name", type=str, default="test", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=52, help="maximum episode length")
    parser.add_argument("--time-steps", type=int, default=104000, help="number of time steps")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    parser.add_argument("--lr-actor", type=float, default=5e-4, help="learning rate of actor")
    parser.add_argument("--lr-critic", type=float, default=5e-4, help="learning rate of critic")
    parser.add_argument("--epsilon", type=float, default=0.5, help="epsilon greedy")
    parser.add_argument("--noise_rate", type=float, default=0.3, help="noise rate for sampling from a standard normal distribution ")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--tau", type=float, default=0.01, help="parameter for updating the target network")
    parser.add_argument("--buffer-size", type=int, default=int(1e6), help="number of transitions can be stored in buffer")
    parser.add_argument("--batch-size", type=int, default=256, help="number of episodes to optimize at the same time")
    parser.add_argument("--pso-inject-threshold", type=float, default=0.05,
                        help="minimum relative candidate improvement required for policy injection")
    parser.add_argument("--pso-swarm-size", type=int, default=10,
                        help="number of PSO particles maintained for each agent")
    parser.add_argument("--evolve-interval", type=int, default=52,
                        help="number of training environment steps between PSO rounds")
    parser.add_argument("--pso-eval-episodes", type=int, default=3,
                        help="episodes used to select the best PSO candidate")
    parser.add_argument("--pso-gate-episodes", type=int, default=10,
                        help="independent episodes used to decide whether to inject the selected candidate")
    # Checkpointing
    parser.add_argument("--save-dir", type=str, default=os.path.join(project_root, "model"),
                        help="directory in which training state and model should be saved")
    parser.add_argument("--run-subdir", type=str, default="env-uncertain0.5",
                        help="shared subdirectory for training curves and test results")
    parser.add_argument("--save-rate", type=int, default=4160, help="save model once every time this many episodes are completed")
    parser.add_argument("--save-interval", type=int, default=10400,
                        help="save model every N steps (0 = don't save)")
    parser.add_argument("--model-dir", type=str, default="",
                        help="source resultN/models directory for resume, evaluate, or test")
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume training from the latest actor_params.pkl and critic_params.pkl')
    parser.add_argument('--test', action='store_true', default=False,
                        help='run in test mode (load trained models)')
    parser.add_argument('--test_episodes', type=int, default=50,
                        help='number of episodes for testing')
    parser.add_argument('--test-checkpoint', choices=['final', 'latest'], default='final',
                        help='checkpoint used by test.py: final or latest periodic checkpoint')
    # Evaluate
    parser.add_argument("--evaluate-episodes", type=int, default=15, help="number of episodes for evaluating")
    parser.add_argument("--evaluate-episode-len", type=int, default=52, help="length of episodes for evaluating")
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="evaluate the latest checkpoint instead of training")
    parser.add_argument("--evaluate-rate", type=int, default=1040,
                        help="evaluate every N training environment steps (default: every 5 episodes)")
    argument_names = [
        action.dest
        for action in parser._actions
        if action.dest != 'help'
    ]
    args = parser.parse_args()
    args._argument_names = argument_names

    return args
