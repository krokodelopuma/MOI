# triangle.py
from vec3 import Vec3
from ray import Ray
from material import Material

class Triangle:
    _next_object_id = 0
    def __init__(self, v0: Vec3, v1: Vec3, v2: Vec3,
                 normal: Vec3, material: Material,
                 is_light: bool = False):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        self.material = material
        self.is_light = is_light
        self.object_id = Triangle._next_object_id
        Triangle._next_object_id += 1

        self.e1 = self.v1 - self.v0
        self.e2 = self.v2 - self.v0

        cross = self.e1.cross(self.e2)
        area2 = cross.length()
        self.area = 0.5 * area2

        if area2 > 1e-12:
            geom_n = cross.normalized()
        else:
            geom_n = Vec3(0.0, 0.0, 0.0)

        if normal is not None:
            ext = normal.normalized()
            if geom_n.dot(ext) < 0:
                ext = -ext
            self.normal = ext
        else:
            self.normal = geom_n

    def sample_point(self) -> Vec3:
        import random
        r1 = random.random()
        r2 = random.random()
        if r1 + r2 > 1.0:
            r1 = 1.0 - r1
            r2 = 1.0 - r2
        return self.v0 + self.e1 * r1 + self.e2 * r2

    def intersect(self, ray: Ray, t_min: float, t_max: float):
        pvec = ray.direction.cross(self.e2)
        det = self.e1.dot(pvec)

        if abs(det) < 1e-8:
            return None
        inv_det = 1.0 / det

        tvec = ray.origin - self.v0
        u = tvec.dot(pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return None

        qvec = tvec.cross(self.e1)
        v = ray.direction.dot(qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return None

        t = self.e2.dot(qvec) * inv_det
        if t < t_min or t > t_max:
            return None

        p = ray.at(t)
        n = self.normal

        if n.dot(ray.direction) > 0.0:
            n = -n

        return t, p, n, self.material, self