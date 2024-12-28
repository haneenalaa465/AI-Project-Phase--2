import numpy as np
from collections import defaultdict
from problem import Problem
from task import Task


class QLearningAgent:
    def __init__(self, problem, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.problem = problem
        self.alpha = alpha  
        self.gamma = gamma 
        self.epsilon = epsilon 
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.training_rewards = []

    def get_q_value(self, state, action):
        return self.q_table[state][action.getID()]

    def choose_action(self, state, available_actions):
        if not available_actions:
            return None

        if np.random.random() < self.epsilon:
            return np.random.choice(available_actions)
        
        q_values = {action: self.get_q_value(state, action) 
                   for action in available_actions}
        return max(q_values.items(), key=lambda x: x[1])[0]

    def update(self, state, action, reward, next_state, next_actions):
        if next_actions:
            next_max = max(self.get_q_value(next_state, a) 
                          for a in next_actions)
        else:
            next_max = 0
            
        current_q = self.get_q_value(state, action)
        new_q = current_q + self.alpha * (
            reward + self.gamma * next_max - current_q)
        self.q_table[state][action.getID()] = new_q

    def train(self, episodes=100):
        for episode in range(episodes):
            self.problem.reset()
            episode_reward = 0
            
            while not self.problem.is_terminal():
                current_state = self.problem.get_state_representation()
                available_actions = self.problem.get_available_actions()
                
                if not available_actions:
                    break
                    
                action = self.choose_action(current_state, available_actions)
                reward = self.problem.step(action)
                episode_reward += reward
                
                next_state = self.problem.get_state_representation()
                next_actions = self.problem.get_available_actions()
                
                self.update(current_state, action, reward, next_state, next_actions)
            
            self.training_rewards.append(episode_reward)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-10:])
                print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")

    def get_optimal_schedule(self):
        self.problem.reset()
        schedule = []
        
        while not self.problem.is_terminal():
            state = self.problem.get_state_representation()
            available_actions = self.problem.get_available_actions()
            
            if not available_actions:
                break
                
            action = max(available_actions,
                        key=lambda a: self.get_q_value(state, a))
            schedule.append(action)
            self.problem.step(action)
            
        return schedule
