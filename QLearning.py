import numpy as np
import random
from task import Task

class QLearningAgent:
    def __init__(self, problem, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.problem = problem
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_state(self):
        state = []
        for task in self.problem.tasks:
            state.append(task.getID())
        return tuple(state)

    def choose_action(self, state):
        available_actions = []
        for task in self.problem.tasks:
            if not task.getDependencies():
                available_actions.append(task)
        print("Available Tasks:\n", available_actions)
        if not available_actions:
            return None 
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_actions)
        q_values = {}
        for task in available_actions:
            q_value = self.q_table.get((state, task.getID()), 0)
            # print("Q val:", q_value)
            q_values[task] = q_value
        print("Q Table:\n", self.q_table)
        best_action = max(q_values, key=q_values.get)
        return best_action

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.q_table.get((state, action.getID()), 0)
        next_q_values = []
        for next_action in self.problem.tasks:
            if not next_action.getDependencies():
                next_q_value = self.q_table.get((next_state, next_action.getID()), 0)
                next_q_values.append(next_q_value)

        next_q = max(next_q_values, default=0)
        self.q_table[(state, action.getID())] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)

    def train(self, episodes=10):
        tasks_copy = []
        for task in self.problem.tasks:
            tasks_copy.append(Task(task.getID(), task.getDescription(), task.getDuration(),
                                    task.getDeadline(), task.getDependencies()))
            
        for episode in range(episodes):
            print(f"Episode {episode + 1}")

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
        policy = []
        tasks_copy = []
        for task in self.problem.tasks:
            tasks_copy.append(Task(task.getID(), task.getDescription(), task.getDuration(),
                                   task.getDeadline(), task.getDependencies()))
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
