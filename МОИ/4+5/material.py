# material.py
from vec3 import Vec3

class Material:
    def __init__(self, kd: Vec3 = None, ks: Vec3 = None, Le: Vec3 = None):
        self.kd = kd if kd is not None else Vec3(0.0, 0.0, 0.0)
        self.ks = ks if ks is not None else Vec3(0.0, 0.0, 0.0)
        self.Le = Le if Le is not None else Vec3(0.0, 0.0, 0.0)

    def is_emissive(self) -> bool:
        return self.Le.max_component() > 0.0