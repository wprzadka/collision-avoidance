import numpy as np


class ReciprocalVelocityObstacle:

    def __init__(self):
        pass

    def compute(self, agents: list):
        vel_diff = np.array([[a.velocity - b.velocity for a in agents] for b in agents])
