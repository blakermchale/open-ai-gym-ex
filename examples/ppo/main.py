import ray
import ray.rllib.agents.ppo as ppo
import gym
from open_ai_gym_ex import WandBMonitor
from argparse import ArgumentParser
import tempfile
import os
from pathlib import Path

ray.shutdown()
ray.init(ignore_reinit_error=True)

# print(f"Dashboard URL: http://{ray.get_webui_url()}")


parser = ArgumentParser()
parser.add_argument("--env", default="CartPole-v1", choices=[env_spec.id for env_spec in gym.envs.registry.all()], help="environmnt to use")
args = parser.parse_args()


TMP_CHECKPOINT = tempfile.TemporaryDirectory(prefix=f"ppo_{args.env}")

DIR_RESULTS = Path(os.getenv("HOME"), "ray_results")

config = ppo.DEFAULT_CONFIG.copy()
config["env"] = args.env
config["log_level"] = "WARN"
config["framework"] = "torch"

agent = ppo.PPOTrainer(config)
env = gym.make(args.env)
agent = WandBMonitor(agent, project="gym_examples", name=f"ppo_{args.env}", save_code=True, sync_tensorboard=True)


EPISODES = 30
STEPS = 200

# obs = env.reset()
for ep in range(EPISODES):
    agent.train()
    print(f"Done with episode {ep}")
    # for step in range(STEPS):
        # agent.train()
        # a = agent.compute_action(obs)
        # obs, rew, done, info = env.step(a)
        # if done:
        #     print(f"Done with episode {ep}")
        #     obs = env.reset()
        #     break

# EPISODES = 30
# s = "{:3d} reward {:6.2f}/{:6.2f}/{:6.2f} len {:6.2f} saved {}"
# for ep in range(EPISODES):
    # result = agent.train()
    # file_name = agent.save(TMP_CHECKPOINT.name)

    # print(s.format(
    #     ep + 1,
    #     result["episode_reward_min"],
    #     result["episode_reward_mean"],
    #     result["episode_reward_max"],
    #     result["episode_len_mean"],
    #     file_name
    # ))


