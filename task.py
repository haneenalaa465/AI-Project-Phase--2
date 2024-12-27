class Task:
    def __init__(self, ID, description, duration, deadline, dependencies):
        self.__ID = ID
        self.__description = description
        self.__duration = duration
        self._deadline = deadline
        self.__dependencies = dependencies

    def __repr__(self):
        return f"Task({self.__ID}, {self.__description}, {self.__duration}, {self._deadline}, {self.__dependencies})"

    def setID(self, ID):
        self.__ID = ID

    def getID(self):
        return self.__ID

    def setDescription(self, description):
        self.__description = description

    def getDescription(self):
        return self.__description

    def setDuration(self, duration):
        self.__duration = duration

    def getDuration(self):
        return self.__duration

    def setDeadline(self, deadline):
        self._deadline = deadline

    def getDeadline(self):
        return self._deadline

    def setDependencies(self, dependencies):
        self.__dependencies = dependencies

    def getDependencies(self):
        return self.__dependencies

    def task_vis(self):
        print("ID:", self.getID(), " Description:", self.getDescription(), " Duration:", self.getDuration(), 
            " Deadline:", self.getDeadline(), " Dependencies:", self.getDependencies())

    def __lt__(self, other):
        self_cost = self.getDuration() + max(0, self.getDeadline() - self.getDuration())
        other_cost = other.getDuration() + max(0, other.getDeadline() - other.getDuration())
        return self_cost < other_cost