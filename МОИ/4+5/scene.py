# scene.py
from ray import Ray
from bvh import BVHNode
import random

class Scene:
    def __init__(self):
        self.triangles = []
        self.light_triangles = []
        self.bvh_root = None
        self._dirty = True

    def add_triangle(self, tri):
        self.triangles.append(tri)
        if tri.is_light:
            self.light_triangles.append(tri)
        self._dirty = True

    def finalize(self):
        self.build_bvh()

    def build_bvh(self):
        if self.triangles:
            self.bvh_root = BVHNode(self.triangles[:])
        else:
            self.bvh_root = None
        self._dirty = False

    def intersect(self, ray: Ray):
        if self._dirty:
            self.build_bvh()

        if self.bvh_root is None:
            return None

        return self.bvh_root.intersect(ray, 1e-4, 1e30)

    def intersect_any(self, ray: Ray, t_max=1e30):
        if self._dirty:
            self.build_bvh()

        if self.bvh_root is None:
            return False

        hit = self.bvh_root.intersect(ray, 1e-4, t_max)
        return hit is not None

    def has_lights(self):
        return len(self.light_triangles) > 0

    def sample_light(self):
        if not self.light_triangles:
            return None

        weights = [
            tri.area * tri.material.Le.max_component()
            for tri in self.light_triangles
        ]

        total = sum(weights)
        r = random.random() * total

        acc = 0.0
        for tri, w in zip(self.light_triangles, weights):
            acc += w
            if r <= acc:
                p = tri.sample_point()
                pdf = w / total / tri.area
                return tri, p, tri.normal, pdf

        tri = self.light_triangles[-1]
        p = tri.sample_point()
        pdf = weights[-1] / total / tri.area
        return tri, p, tri.normal, pdf