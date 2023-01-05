import argparse
import baseline_env
import modified_env
from stable_baselines3 import PPO


def train(args):
    if args.env == 'baseline':
        env = baseline_env.DrawEnv()
    elif args.env == 'modified':
        env = modified_env.DrawEnv()
    else:
        print('Invalid environment!')
        exit()

    model = PPO("MlpPolicy", env, learning_rate=args.lr, n_steps=args.n_steps, batch_size=args.batch_size,
                gamma=args.gamma, verbose=1, tensorboard_log=args.tb_log)
    model.learn(total_timesteps=args.time_steps, progress_bar=True)
    model.save(args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Draw with PPO')

    parser.add_argument('-env', dest='env', default='baseline', type=str,
                        help='What environment to use for training (Options: baseline, modified).')
    parser.add_argument('-lr', dest='lr', default=0.001, type=float,
                        help='Learning rate of training.')
    parser.add_argument('-n_steps', dest='n_steps', default=512, type=int,
                        help='Number of steps to run per PPO update.')
    parser.add_argument('-batch_size', dest='batch_size', default=64, type=int,
                        help='PPO training minibatch size.')
    parser.add_argument('-gamma', dest='gamma', default=0.99, type=float,
                        help='Discount factor for PPO training')
    parser.add_argument('-tensorboard_log', dest='tb_log', default='./ppo_draw/', type=str,
                        help='Where to save the tensorboard log files.')
    parser.add_argument('-time_steps', dest='time_steps', default=1000000, type=int,
                        help='Number of PPO training time steps.')
    parser.add_argument('-save_model_path', dest='model_path', default='model', type=str,
                        help='Where to save the trained model.')

    args = parser.parse_args()

    train(args)
