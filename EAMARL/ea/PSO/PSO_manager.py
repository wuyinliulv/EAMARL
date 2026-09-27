# pso_manage.py
import copy
import numpy as np
import torch
import random


def _soft_update_target_network(target, source, tau=0.01):
    """软更新目标网络参数"""
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * param.data
        )

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class PSOManager:
    def __init__(self, args, maddpg_agents):
        self.args = args
        self.agents = maddpg_agents
        self.swarm_size = args.pso_swarm_size
        if self.swarm_size < 1:
            raise ValueError("pso_swarm_size must be at least 1")
        self.initial_inertia = 0.9
        self.final_inertia = 0.4
        self.current_inertia = self.initial_inertia
        self.c1 = 1.7
        self.c2 = 1.3
        self.clamp_velocity = True
        self.velocity_clamp = 0.5

        # 初始化粒子群
        self.swarms = []
        for agent in self.agents:
            net_template = copy.deepcopy(agent.policy.actor_network)

            particles = self._init_particles(net_template)
            velocities = [self._zero_vector(state_dict) for state_dict in particles]

            particle_fitness = np.full(self.swarm_size, -np.inf)
            swarm = {
                'particles': particles,
                'velocities': velocities,
                'pbest': copy.deepcopy(particles),
                'pbest_fitness': np.full(self.swarm_size, -np.inf),
                'gbest': None,
                'gbest_fitness': -np.inf,
                'net_template': net_template,
                'particle_fitness': particle_fitness ,
            }
            self.swarms.append(swarm)

    def _init_particles(self, network_template):
        particles = []
        base_state = copy.deepcopy(network_template.state_dict())

        for _ in range(self.swarm_size):

            particle_state = copy.deepcopy(base_state)

            for key in particle_state:
                noise = torch.randn_like(particle_state[key]) * 0.1
                particle_state[key] += noise

            particles.append(particle_state)
        return particles

    def _zero_vector(self, state_dict):

        return {k: torch.zeros_like(v) for k, v in state_dict.items()}


    def prepare_evolution_round(self):

        for agent_idx, swarm in enumerate(self.swarms):
            current_actor = copy.deepcopy(
                self.agents[agent_idx].policy.actor_network.state_dict()
            )
            swarm['particles'][0] = current_actor
            swarm['velocities'][0] = self._zero_vector(current_actor)
            swarm['particle_fitness'].fill(-np.inf)
            swarm['pbest_fitness'].fill(-np.inf)
            swarm['gbest'] = None
            swarm['gbest_fitness'] = -np.inf

    def evaluate_particles(self, runner, scenario_seeds):

        print("\n[PSO] Evaluating particles...")

        for agent_idx, swarm in enumerate(self.swarms):
            for part_idx in range(self.swarm_size):

                particle_fitness = runner._evaluate_individual(
                    agent_idx,
                    swarm['particles'][part_idx],
                    swarm['net_template'],
                    scenario_seeds
                )
                swarm['particle_fitness'][part_idx] = particle_fitness

                pbest_fitness = runner._evaluate_individual(
                    agent_idx,
                    swarm['pbest'][part_idx],
                    swarm['net_template'],
                    scenario_seeds
                )

                if particle_fitness >= pbest_fitness:
                    swarm['pbest_fitness'][part_idx] = particle_fitness
                    swarm['pbest'][part_idx] = copy.deepcopy(swarm['particles'][part_idx])
                else:
                    swarm['pbest_fitness'][part_idx] = pbest_fitness

                if swarm['pbest_fitness'][part_idx] > swarm['gbest_fitness']:
                    swarm['gbest_fitness'] = swarm['pbest_fitness'][part_idx]
                    swarm['gbest'] = copy.deepcopy(swarm['pbest'][part_idx])
                    print(f"Agent {agent_idx} new global best: {swarm['gbest_fitness']:.4f}")

        for agent_idx, swarm in enumerate(self.swarms):
            fitness_str = ", ".join([f"{f:.2f}" for f in swarm['pbest_fitness']])
            print(f"Agent {agent_idx} particle fitness: [{fitness_str}]")

        print(f"[PSO] Evaluation complete. Best fitness: {max(s['gbest_fitness'] for s in self.swarms):.4f}\n")
    # [FIX-PSO-CONSISTENCY] END


    def update_particles(self, current_time_step):
        print("\n[PSO] Updating particles...")

        progress = min(1.0, current_time_step / (self.args.time_steps * 0.8))

        self.current_inertia = (self.initial_inertia -
                                progress * (self.initial_inertia - self.final_inertia))

        current_c1 = self.c1
        current_c2 = self.c2

        for swarm in self.swarms:
            fitness_std = np.std(swarm['pbest_fitness'])
            fitness_mean = np.abs(np.mean(swarm['pbest_fitness']))

            if fitness_mean > 1e-6 and fitness_std / fitness_mean < 0.1:
                current_c1 = min(2.2, current_c1 * 1.05)
                current_c2 = max(0.8, current_c2 * 0.95)
                print(f"Low diversity detected: c1={current_c1:.2f}, c2={current_c2:.2f}")

            for part_idx in range(self.swarm_size):
                current = swarm['particles'][part_idx]
                velocity = swarm['velocities'][part_idx]
                pbest = swarm['pbest'][part_idx]
                gbest = swarm['gbest']

                new_velocity = {}
                new_position = {}

                for key in current.keys():
                    v = velocity[key]
                    x = current[key]
                    p = pbest[key]
                    g = gbest[key]

                    r1 = random.random()
                    r2 = random.random()

                    v_update = self.current_inertia * v
                    v_update += current_c1 * r1 * (p - x)
                    v_update += current_c2 * r2 * (g - x)

                    if self.clamp_velocity:
                        v_update = torch.clamp(v_update, -self.velocity_clamp, self.velocity_clamp)

                    new_pos = x + v_update

                    new_velocity[key] = v_update
                    new_position[key] = new_pos

                swarm['velocities'][part_idx] = new_velocity
                swarm['particles'][part_idx] = new_position


        print(f"Updated particles with inertia={self.current_inertia:.2f}, c1={current_c1:.2f}, c2={current_c2:.2f}")

    def get_best_candidate(self):
        """Return the best candidate selected on this round's PSO scenarios."""
        valid_candidates = [
            (agent_idx, swarm)
            for agent_idx, swarm in enumerate(self.swarms)
            if swarm['gbest'] is not None and np.isfinite(swarm['gbest_fitness'])
        ]
        if not valid_candidates:
            return None

        agent_idx, swarm = max(
            valid_candidates,
            key=lambda item: item[1]['gbest_fitness']
        )
        return {
            'agent_idx': agent_idx,
            'policy_state': copy.deepcopy(swarm['gbest']),
            'selection_fitness': float(swarm['gbest_fitness']),
        }

    def inject_best_policy(self, current_performance, threshold=0.05,
                           candidate=None, candidate_performance=None):
        """Inject a selected candidate using an independently measured score.

        Returning a small, serializable event dictionary lets Runner persist
        every accepted and rejected gate decision for later analysis.
        """
        print("\n======= PSO Policy Injection =======")
        injected = False
        improvement = float('nan')
        agent_idx = None

        if candidate is None:
            candidate = self.get_best_candidate()

        if candidate is not None:
            agent_idx = candidate['agent_idx']
            swarm = self.swarms[agent_idx]
            if candidate_performance is None:
                candidate_performance = candidate['selection_fitness']

            # Both scores must be measured on the same independent gate
            # scenarios.  This avoids a larger swarm passing the gate merely
            # because its maximum was selected from more noisy candidates.
            improvement = (
                candidate_performance - current_performance
            ) / max(1e-6, abs(current_performance))
            if improvement > threshold:
                best_net = copy.deepcopy(swarm['net_template'])
                best_net.load_state_dict(candidate['policy_state'])

                self.agents[agent_idx].policy.actor_network.load_state_dict(best_net.state_dict())
                self.agents[agent_idx].policy.actor_optim.state.clear()
                print(f"Injected PSO policy to agent {agent_idx} (+{improvement * 100:.1f}%)")

                _soft_update_target_network(
                    self.agents[agent_idx].policy.actor_target_network,
                    self.agents[agent_idx].policy.actor_network,
                    tau=0.01
                )
                injected = True
            else:
                print(
                    f"Skipped injection for best candidate agent {agent_idx} "
                    f"(improvement: {improvement * 100:.1f}% < {threshold * 100}%)"
                )

        if not injected:
            print("No PSO injections performed this round")

        print("======= Injection Complete =======\n")
        return {
            'injected': injected,
            'agent_idx': agent_idx,
            'baseline_fitness': float(current_performance),
            'candidate_fitness': (
                float(candidate_performance)
                if candidate_performance is not None else float('nan')
            ),
            'relative_improvement': float(improvement),
            'selection_fitness': (
                float(candidate['selection_fitness'])
                if candidate is not None else float('nan')
            ),
            'threshold': float(threshold),
        }
