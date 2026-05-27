# vec3.py - вектор для path tracing
import math

class Vec3:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

    def __neg__(self):
        return Vec3(-self.x, -self.y, -self.z)

    def __add__(self, other):
        return Vec3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vec3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, Vec3):
            return Vec3(self.x * other.x, self.y * other.y, self.z * other.z)
        return Vec3(self.x * other, self.y * other, self.z * other)

    __rmul__ = __mul__

    def __truediv__(self, s):
        if abs(s) < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return Vec3(self.x / s, self.y / s, self.z / s)

    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other):
        return Vec3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )

    def length(self):
        return math.sqrt(self.dot(self))

    def length_squared(self):
        return self.dot(self)

    def normalized(self):
        l = self.length()
        if l < 1e-12:
            return Vec3(0.0, 0.0, 0.0)
        return self / l

    def max_component(self):
        return max(self.x, self.y, self.z)

    def min_component(self):
        return min(self.x, self.y, self.z)

    def is_black(self, eps=1e-6):
        return self.x < eps and self.y < eps and self.z < eps

    def clamp(self, lo=0.0, hi=1.0):
        return Vec3(
            max(lo, min(hi, self.x)),
            max(lo, min(hi, self.y)),
            max(lo, min(hi, self.z)),
        )

    def gamma_correct(self, gamma=2.2):
        inv = 1.0 / gamma
        return Vec3(
            self.x ** inv if self.x > 0 else 0.0,
            self.y ** inv if self.y > 0 else 0.0,
            self.z ** inv if self.z > 0 else 0.0,
        )

    def to_rgb8(self):
        r = int(max(0.0, min(1.0, self.x)) * 255.0 + 0.5)
        g = int(max(0.0, min(1.0, self.y)) * 255.0 + 0.5)
        b = int(max(0.0, min(1.0, self.z)) * 255.0 + 0.5)
        return r, g, b

    def __repr__(self):
        return f"Vec3({self.x:.6f}, {self.y:.6f}, {self.z:.6f})"