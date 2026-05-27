# camera.py
import math
from vec3 import Vec3
from ray import Ray

class Camera:
    def __init__(self, lookfrom: Vec3, lookat: Vec3, vup: Vec3,
                 vfov_deg: float, aspect: float):
        theta = math.radians(vfov_deg)
        half_height = math.tan(theta / 2.0)
        half_width = aspect * half_height

        self.origin = lookfrom
        w = (lookfrom - lookat).normalized()
        u = vup.cross(w).normalized()
        v = w.cross(u)

        self.lower_left_corner = self.origin - u * half_width - v * half_height - w
        self.horizontal = u * (2.0 * half_width)
        self.vertical = v * (2.0 * half_height)

    def get_ray(self, s: float, t: float) -> Ray:
        direction = (self.lower_left_corner +
                     self.horizontal * s +
                     self.vertical * t -
                     self.origin)
        return Ray(self.origin, direction)