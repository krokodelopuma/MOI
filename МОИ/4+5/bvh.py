# bvh.py
from vec3 import Vec3

class AABB:
    def __init__(self, mn: Vec3, mx: Vec3):
        self.min = mn
        self.max = mx

    @staticmethod
    def from_triangle(tri):
        v0, v1, v2 = tri.v0, tri.v1, tri.v2
        mn = Vec3(min(v0.x, v1.x, v2.x),
                  min(v0.y, v1.y, v2.y),
                  min(v0.z, v1.z, v2.z))
        mx = Vec3(max(v0.x, v1.x, v2.x),
                  max(v0.y, v1.y, v2.y),
                  max(v0.z, v1.z, v2.z))
        return AABB(mn, mx)

    @staticmethod
    def union(a, b):
        mn = Vec3(min(a.min.x, b.min.x),
                  min(a.min.y, b.min.y),
                  min(a.min.z, b.min.z))
        mx = Vec3(max(a.max.x, b.max.x),
                  max(a.max.y, b.max.y),
                  max(a.max.z, b.max.z))
        return AABB(mn, mx)

    def hit(self, ray, t_min, t_max):
        for i, (mn, mx, o, d) in enumerate((
            (self.min.x, self.max.x, ray.origin.x, ray.direction.x),
            (self.min.y, self.max.y, ray.origin.y, ray.direction.y),
            (self.min.z, self.max.z, ray.origin.z, ray.direction.z),
        )):
            invD = 1.0 / d if d != 0 else 1e30
            t0 = (mn - o) * invD
            t1 = (mx - o) * invD
            if invD < 0:
                t0, t1 = t1, t0
            t_min = t0 if t0 > t_min else t_min
            t_max = t1 if t1 < t_max else t_max
            if t_max <= t_min:
                return False
        return True

class BVHNode:
    def __init__(self, triangles, depth=0, max_leaf=4):
        self.left = None
        self.right = None
        self.triangles = None
        self.box = None

        if not triangles:
            return

        box = AABB.from_triangle(triangles[0])
        for t in triangles[1:]:
            box = AABB.union(box, AABB.from_triangle(t))
        self.box = box

        if len(triangles) <= max_leaf:
            self.triangles = triangles
            return

        ext = Vec3(box.max.x - box.min.x,
                   box.max.y - box.min.y,
                   box.max.z - box.min.z)
        if ext.x >= ext.y and ext.x >= ext.z:
            axis = 0
        elif ext.y >= ext.x and ext.y >= ext.z:
            axis = 1
        else:
            axis = 2

        triangles.sort(key=lambda tri: (
            (tri.v0.x + tri.v1.x + tri.v2.x)/3 if axis == 0 else
            (tri.v0.y + tri.v1.y + tri.v2.y)/3 if axis == 1 else
            (tri.v0.z + tri.v1.z + tri.v2.z)/3
        ))

        mid = len(triangles) // 2
        left_tris = triangles[:mid]
        right_tris = triangles[mid:]

        self.left = BVHNode(left_tris, depth+1, max_leaf)
        self.right = BVHNode(right_tris, depth+1, max_leaf)

    def intersect(self, ray, t_min, t_max):
        if self.box is None or not self.box.hit(ray, t_min, t_max):
            return None

        hit_any = None
        closest = t_max

        if self.triangles is not None:
            for tri in self.triangles:
                res = tri.intersect(ray, t_min, closest)
                if res:
                    t, p, n, m, tref = res
                    closest = t
                    hit_any = (t, p, n, m, tref)
            return hit_any

        hit_left = self.left.intersect(ray, t_min, closest) if self.left else None
        if hit_left:
            t, _, _, _, _ = hit_left
            closest = t
            hit_any = hit_left

        hit_right = self.right.intersect(ray, t_min, closest) if self.right else None
        if hit_right:
            hit_any = hit_right

        return hit_any