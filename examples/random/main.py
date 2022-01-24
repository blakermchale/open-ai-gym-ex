import gym
from open_ai_gym_ex import WandBMonitor
from argparse import ArgumentParser


parser = ArgumentParser()
parser.add_argument("--env", default="CartPole-v1", choices=[env_spec.id for env_spec in gym.envs.registry.all()], help="environmnt to use")
args = parser.parse_args()


EPISODES = 5
STEPS = 200

env = gym.make(args.env)
env = WandBMonitor(env, project="gym_examples", name=f"rand_{args.env}", save_code=True, sync_tensorboard=True)

obs = env.reset()
for ep in range(EPISODES):
    for step in range(STEPS):
        a = env.action_space.sample()
        obs, rew, done, info = env.step(a)
        if done:
            print(f"Done with episode {ep}")
            obs = env.reset()
            break
