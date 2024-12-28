import numpy as np
import random
from task import Task
import copy

class QLearningAgent:
    def __init__(self, problem, alpha=0.1, gamma=0.9, epsilon=0.1):  # Fixed double underscore in __init__
        self.problem = problem
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self):
        # Get the current state as a tuple of task IDs
        state = [task.getID() for task in self.problem.tasks]
        return tuple(state)

    def choose_action(self, state):
        # Find available actions (tasks with no dependencies)
        available_actions = [task for task in self.problem.tasks if not task.getDependencies()]
        if not available_actions:
            return None  # No valid actions available
        
        # Epsilon-greedy policy
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)

        # Calculate Q-values for available actions
        q_values = {
            task: self.q_table.get((state, task.getID()), 0) for task in available_actions
        }

        # Select the best action based on Q-values
        best_action = max(q_values, key=q_values.get)
        return best_action

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action.getID()), 0)
        next_q_values = [
            self.q_table.get((next_state, task.getID()), 0)
            for task in self.problem.tasks if not task.getDependencies()
        ]

        # Bellman equation for Q-value update
        next_q = max(next_q_values, default=0)
        self.q_table[(state, action.getID())] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def train(self, episodes=10):
        # Create a deep copy of tasks for training
        tasks_copy = [Task(
            task.getID(),
            task.getDescription(),
            task.getDuration(),
            task.getDeadline(),
            list(task.getDependencies()) if isinstance(task.getDependencies(), list) else task.getDependencies()
        ) for task in self.problem.tasks]
            
        for episode in range(episodes):
            print(f"Episode {episode + 1}")

            # Reinitialize the problem with copied tasks
            self.problem.__init__(tasks_copy, self.problem.init_state)

            state = self.get_state()

            while not self.problem.goal_state():
                action = self.choose_action(state)
                if action is None:
                    print("No valid actions available.")
                    break
                self.problem.action()
                reward = self.problem.step_cost().get(action, 0)

                next_state = self.get_state()

                print(f"State: {state}, Action: {action}, Reward: {reward}, Next State: {next_state}")
                self.update_q_value(state, action, reward, next_state)
                state = next_state

    def get_policy(self):
        # Generate the optimal policy based on Q-values
        policy = []
        tasks_copy = [Task(
            task.getID(),
            task.getDescription(),
            task.getDuration(),
            task.getDeadline(),
            list(task.getDependencies()) if isinstance(task.getDependencies(), list) else task.getDependencies()
        ) for task in self.problem.tasks]

        # Reinitialize the problem for policy extraction
        self.problem.__init__(tasks_copy, self.problem.init_state)

        state = self.get_state()

        while not self.problem.goal_state():
            action = self.choose_action(state)
            if action is None:
                break
            policy.append(action)
            self.problem.action()
            state = self.get_state()

        return policy