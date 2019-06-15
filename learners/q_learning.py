import numpy as np
from IPython.display import clear_output
import time

class QLearning:
    """
    QLearning reinforcement learning agent.

    Arguments:
      epsilon - (float) The probability of randomly exploring the action space
        rather than exploiting the best action.
      discount - (float) The discount factor. Controls the perceived value of
        future reward relative to short-term reward.
      adaptive - (bool) Whether to use an adaptive policy for setting
        values of epsilon during training
    """

    def __init__(self, epsilon=0.2, discount=0.95, adaptive=False):
        self.epsilon = epsilon
        self.discount = discount
        self.adaptive = adaptive

    def fit(self, env, steps=1000):
        state_action_values = np.zeros(
            (env.observation_space.n, env.action_space.n))
        N_actions_performed = np.zeros((env.action_space.n, ), dtype=int)
        state = env.reset()
        rewards = np.zeros((100, ))

        s = np.floor(steps / 100)
        s_count = 0
        reward_sum = 0
        idx = 0

        for step in range(steps):
            epsilon = self._get_epsilon(step / steps)
            # generate random num
            p = np.random.random()
            # check probability
            action = env.action_space.sample(
            )  # your agent here (this takes random actions)
            if p >= epsilon and len(set(state_action_values[state])) != 1:
                action = np.argmax(state_action_values[state])
            # take action and observe R, S'
            observation, reward, done, info = env.step(action)
            # update values
            N_actions_performed[action] += 1
            state_action_values[state][
                action] += 1 / N_actions_performed[action] * (
                    reward +
                    self.discount * max(state_action_values[observation]) -
                    state_action_values[state][action])
            reward_sum += reward
            # set next state
            state = observation
            # check s
            s_count += 1
            if s == s_count:
                rewards[idx] = reward_sum / (step + 1)
                s_count = 0
                idx += 1

            if done:
                state = env.reset()

        return state_action_values, rewards

    def predict(self, env, state_action_values):
        states, actions, rewards = [], [], []

        state = env.reset()
        env.render()
        time.sleep(0.3)

        while True:
            clear_output(wait=True)
            action = np.argmax(state_action_values[state])
            # take action and observe R, S'
            observation, reward, done, info = env.step(action)
            env.render()
            print(observation)
            time.sleep(0.3)
            # set next state
            state = observation
            # record data
            states.append(observation)
            actions.append(action)
            rewards.append(reward)

            if done:
                break

        return np.array(states), np.array(actions), np.array(rewards)

    def _get_epsilon(self, progress):
        return self._adaptive_epsilon(
            progress) if self.adaptive else self.epsilon

    def _adaptive_epsilon(self, progress):
        return (1 - progress) * self.epsilon
