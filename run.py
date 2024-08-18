from dqn_agent import DQNAgent
from tetris import Tetris
from datetime import datetime
from statistics import mean
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Run DQN with Tetris
def dqn():
    env = Tetris()
    episodes = 2000
    max_steps = None
    epsilon_stop_episode = 1500
    mem_size = 20000
    discount = 0.95
    batch_size = 512
    epochs = 1
    render_every = 50
    log_every = 50
    replay_start_size = 2000
    train_every = 1
    n_neurons = [32, 32]
    render_delay = None
    activations = ['relu', 'relu', 'linear']

    agent = DQNAgent(env.get_state_size(),
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    log_dir = f'logs/tetris-nn={str(n_neurons)}-mem={mem_size}-bs={batch_size}-e={epochs}-{datetime.now().strftime("%Y%m%d-%H%M%S")}'
    writer = SummaryWriter(log_dir)

    scores = []
    all_rewards = []
    all_losses = []
    epsilon_values = []

    for episode in tqdm(range(episodes)):
        current_state = env.reset()
        done = False
        steps = 0
        total_reward = 0

        render = render_every and episode % render_every == 0

        while not done and (not max_steps or steps < max_steps):
            next_states = env.get_next_states()
            best_state = agent.best_state(next_states.values())
            
            best_action = None
            for action, state in next_states.items():
                if state == best_state:
                    best_action = action
                    break

            reward, done = env.play(best_action[0], best_action[1], render=render, render_delay=render_delay)
            total_reward += reward
            
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = next_states[best_action]
            steps += 1

        scores.append(env.get_game_score())
        all_rewards.append(total_reward)
        epsilon_values.append(agent.epsilon)

        if episode % train_every == 0:
            loss = agent.train(batch_size=batch_size, epochs=epochs)
            if loss is not None:
                all_losses.append(loss)

        if log_every and episode and episode % log_every == 0:
            avg_score = mean(scores[-log_every:])
            avg_reward = mean(all_rewards[-log_every:])
            avg_loss = mean(all_losses[-log_every:]) if all_losses else 0
            epsilon = np.mean(epsilon_values[-log_every:])

            writer.add_scalar('Average Score', avg_score, episode)
            writer.add_scalar('Average Reward', avg_reward, episode)
            writer.add_scalar('Average Loss', avg_loss, episode)
            writer.add_scalar('Epsilon', epsilon, episode)

    # Close the TensorBoard writer after all episodes are complete
    writer.close()

if __name__ == "__main__":
    dqn()
