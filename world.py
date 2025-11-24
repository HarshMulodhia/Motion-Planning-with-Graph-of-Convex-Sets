class Point():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class ConvexSet():
    def __init__(self, vertices):
        self.vertices = vertices

    def __repr__(self):
        return f"ConvexSet with {len(self.vertices)} vertices"
        

class World():

    def __init__(self, start=Point(), goal=Point()):
        self.start = start
        self.goal = goal
        self.obstacles = []
        self.convex_sets = []
