# from task import Task
class Problem:
    def __init__(self, tasks, init_state):
        self.tasks = tasks
        self.init_state = init_state
        self.length = len(tasks)
        self.schedule = []
        self.today = 0

    #return dictionary with all tasks w/o dependencies & their costs
    def step_cost(self):
        
        possible_routes = dict()
        for task in self.tasks:
            if not task.getDependencies():
                cost = (task.getDeadline() - (self.today + task.getDuration()))
                possible_routes[task] = cost
        return possible_routes

    # appends best task to schedule 
    def action(self):
        possible_routes = self.step_cost()
        if not possible_routes:  
            return
        selected_task = min(possible_routes, key=possible_routes.get)
        if selected_task in self.schedule:
            order = sorted(possible_routes.items(), key=lambda item: item[1])
            print("\norder:\n", order)
            selected_task = order[1][0]
            # print("\n\n\ntask", selected_task[0])
        self.today += selected_task.getDuration()
        self.schedule.append(selected_task)
        for task in self.tasks:
            if selected_task.getID() in task.getDependencies():
                deps = task.getDependencies()
                deps.remove(selected_task.getID())
                task.setDependencies(deps)


    # returns schedule    
    def result(self):
        return self.schedule

    # checks whether all tasks have been added
    def goal_state(self):
        return len(self.schedule) == self.length and bool(self.schedule)
