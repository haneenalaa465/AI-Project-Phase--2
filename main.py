from task import Task
from problem import Problem
from QLearning import QLearningAgent

def main():
    # Define tasks
    tasks = [
        Task(1, "Task 1", 3, 5, []),
        Task(2, "Task 2", 2, 7, [1]),
        Task(3, "Task 3", 1, 6, []),
        Task(4, "Task 4", 4, 10, [2, 3])
    ]

    # Initialize problem
    problem = Problem(tasks, init_state=(tasks, 0))

    # Initialize Q-learning agent
    agent = QLearningAgent(problem)

    # Train the agent
    agent.train(episodes=10)

    # Get the optimal policy (schedule)
    policy = agent.get_policy()
    print("Optimal Schedule:", policy)

    # Print the schedule
    print("Schedule:")
    for task in policy:
        task.task_vis()

if __name__ == "__main__":
    main()
