# ray.py
from vec3 import Vec3

class Ray:
    def __init__(self, origin: Vec3, direction: Vec3):
        self.origin = origin
        self.direction = direction.normalized()

    def at(self, t: float) -> Vec3:
        return self.origin + self.direction * t