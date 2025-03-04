


class Node:
    def __init__(self, level, question, state):
        self.level = level
        self.question = question
        self.state = state
        # self.score = score
    

class Tree:
    def __init__(self, condition, questions):
        self.condition = condition
        self.questions = questions

    def __str__(self):
        return f"Tree(condition={self.condition}, questions={self.questions})"
        
