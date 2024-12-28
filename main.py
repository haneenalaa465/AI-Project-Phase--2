from task import Task
from problem import Problem
from QLearning import QLearningAgent

def main():
    tasks = [
        Task(1, "Requirements Analysis", 3, 5, []),
        Task(2, "Design Phase", 2, 7, [1]),
        Task(3, "Infrastructure Setup", 1, 6, []),
        Task(4, "Implementation", 4, 10, [2, 3]),
        Task(5, "Testing", 2, 12, [4]),
        Task(6, "Deployment", 1, 13, [5])
    ]

    problem = Problem(tasks, init_state=(tasks, 0))
    agent = QLearningAgent(problem)

    print("Training the agent...")
    agent.train(episodes=10)

    agent.present_q_values(episode=10)
    agent.analyze_agent_behavior()

    optimal_schedule = agent.get_optimal_schedule()
    print("\nOptimal Schedule:")
    for i, task in enumerate(optimal_schedule, 1):
        print(f"{i}. {task.getDescription()} (ID: {task.getID()})")
        print(f"   Duration: {task.getDuration()}, Deadline: {task.getDeadline()}")

if __name__ == "__main__":
    main()
