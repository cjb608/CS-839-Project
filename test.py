import argparse
import baseline_env
import modified_env
from stable_baselines3 import PPO


def test(args):
    if args.env == 'baseline':
        env = baseline_env.DrawEnv()
    elif args.env == 'modified':
        env = modified_env.DrawEnv()
    else:
        print('Invalid environment!')
        exit()

    model = PPO.load(args.model_path)

    state = env.reset()
    done = False
    while not done:
        action, _states = model.predict(state, deterministic=True)
        next_state, reward, done, _ = env.step(action)
        env.render()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test Draw with PPO')

    parser.add_argument('-env', dest='env', default='baseline', type=str,
                        help='What environment to use for testing (Options: baseline, modified).')

    parser.add_argument('-load_model_path', dest='model_path', default='model', type=str,
                        help='Where the trained model is stored.')

    args = parser.parse_args()

    test(args)
