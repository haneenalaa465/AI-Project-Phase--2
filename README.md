# AI Task Scheduling – Reinforcement Learning with Q-Learning

## 📚 Course
 - Artificial Intelligence (CSAI 301)
 - School of Computer Science and Artificial Intelligence
 - Zewail City of Science and Technology

## 📌 Objective
We extend our task scheduling problem to a Reinforcement Learning (RL) framework. The goal is to train an agent using Q-learning to discover an optimal scheduling policy that respects task dependencies and minimizes penalties due to deadline violations.

## 🧠 Problem Description

Each task has:
 - A duration and deadline
 - A set of dependencies that must be completed before it can begin

The RL agent learns to:
 - Schedule tasks in an order that respects all constraints
 - Maximize the cumulative reward based on how early tasks are completed relative to their deadlines


## 🤖 Reinforcement Learning Design

### 📌 State Representation

Each state includes:
 - Current time (today)
 - Set of unscheduled and scheduled task IDs

### 📌 Actions
An action corresponds to selecting a task to schedule that has no unmet dependencies.

### 📌 Reward Function
 - Positive reward for completing a task on or before its deadline
 - Negative reward for invalid actions or missed deadlines

### 📌 Algorithm
 - Q-Learning: Tabular RL method that updates Q-values using the Bellman equation:

 ## $Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_a Q(s', a) - Q(s,a) \right]$



## ▶️ Running the Project
```
python main.py
```
Console Output:
 - Trained agent for a number of episodes
 - Prints Q-values and policy analysis
 - Displays the optimal task schedule discovered by the RL agent

## 💡 Learning Outcomes
 - Translated a dependency-based scheduling problem into an RL environment
 - Designed and implemented a Q-learning agent
 - Tuned hyperparameters (learning rate, discount factor, exploration rate)
 - Gained insights into the value of exploration vs exploitation
 - Compared the learned RL policy to classical search solutions (from Phase I)
