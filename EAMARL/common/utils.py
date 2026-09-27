import numpy as np
import inspect
import functools

from common.envcore import EnvCore


def store_args(method):
    """Stores provided method args as instance attributes.
    """
    argspec = inspect.getfullargspec(method)
    defaults = {}
    if argspec.defaults is not None:
        defaults = dict(
            zip(argspec.args[-len(argspec.defaults):], argspec.defaults))
    if argspec.kwonlydefaults is not None:
        defaults.update(argspec.kwonlydefaults)
    arg_names = argspec.args[1:]

    @functools.wraps(method)
    def wrapper(*positional_args, **keyword_args):
        self = positional_args[0]
        # Get default arg values
        args = defaults.copy()
        # Add provided arg values
        for name, value in zip(arg_names, positional_args[1:]):
            args[name] = value
        args.update(keyword_args)
        self.__dict__.update(args)
        return method(*positional_args, **keyword_args)

    return wrapper


def make_env(args):
    class CustomMultiAgentEnv:
        def __init__(self, args):
            # 初始化自定义环境
            initial_states = {
                "collection_agent": 0,
                "disassembly_agent": 0,
                "recycling_agent": 0,
                "factory_agent": 0,
                "distributor_agent": 0
            }
            self.core_env = EnvCore(
                initial_states,
                max_time_steps=args.max_episode_len
            )

            self.n = 5
            self.observation_space = [self.ObservationSpace(2) for _ in range(5)]
            self.action_space = [self.ActionSpace(1) for _ in range(5)]

            args.n_players = self.n
            args.n_agents = self.n
            args.obs_shape = [2] * 5
            args.action_shape = [1] * 5
            args.high_action = 1.0
            args.low_action = 0.0

        class ObservationSpace:
            def __init__(self, dim):
                self.shape = (dim,)
                # print("维度",dim)
                # print(self.shape)
        class ActionSpace:
            def __init__(self, dim):
                self.n = dim
                # print("维度", dim)
                # print(self.n)

        def reset(self):
            raw_obs = self.core_env.reset()
            return [np.array(agent_obs, dtype=np.float32).flatten() for agent_obs in raw_obs]

        def step(self, actions):

            obs, rewards, dones, infos = self.core_env.step(actions)

            formatted_obs = [np.array(agent_obs, dtype=np.float32).flatten() for agent_obs in obs]

            formatted_rewards = [np.array([reward], dtype=np.float32) for reward in rewards]

            formatted_dones = dones

            return formatted_obs, formatted_rewards, formatted_dones, infos

        def render(self, mode='human'):

            return self.core_env.render(mode)
        @property
        def agents(self):
            return [f"agent_{i}" for i in range(5)]


    env = CustomMultiAgentEnv(args)

    return env, args

