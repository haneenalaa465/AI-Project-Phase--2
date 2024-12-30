class Problem:
    def __init__(self, tasks, init_state):
        self.tasks = tasks
        self.init_state = init_state
        self.length = len(tasks)
        self.schedule = []
        self.today = 0
        self.total_reward = 0

    def get_state_representation(self):
        completed_task_ids = []
        for task in self.schedule:
            completed_task_ids.append(task.getID())
        return (tuple(completed_task_ids), self.today)

    def get_available_actions(self):
        available = []
        scheduled_task_ids = []
        
        for task in self.schedule:
            scheduled_task_ids.append(task.getID())
        
        for task in self.tasks:
            if task in self.schedule:
                continue
                
            deps = task.getDependencies()
            all_deps_completed = True
            
            for dep in deps:
                if dep not in scheduled_task_ids:
                    all_deps_completed = False
                    break

            if all_deps_completed:
                available.append(task)
                
        return available

    def step(self, task):
        available_actions = self.get_available_actions()
        
        if task not in available_actions:
            return -float('inf')  # Invalid action penalty
        
        completion_time = self.today + task.getDuration()
        
        time_to_deadline = task.getDeadline() - completion_time
        
        reward = time_to_deadline
        
        self.schedule.append(task)
        self.today = completion_time
        self.total_reward += reward
        
        return reward

    def is_terminal(self):
        return len(self.schedule) == self.length

    def reset(self):
        self.schedule = []
        self.today = 0
        self.total_reward = 0