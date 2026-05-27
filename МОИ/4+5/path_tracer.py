import math
import random
from vec3 import Vec3
from ray import Ray

MAX_DEPTH = 8
N_LIGHT_SAMPLES = 4
FIREFLY_THRESHOLD = 10.0

def reflect(v, n):
    return v - n * (2 * v.dot(n))

def cosine_weighted_hemisphere(normal):
    r1 = random.random()
    r2 = random.random()
    phi = 2 * math.pi * r1
    x = math.cos(phi) * math.sqrt(r2)
    y = math.sin(phi) * math.sqrt(r2)
    z = math.sqrt(1 - r2)

    w = normal.normalized()
    a = Vec3(1, 0, 0) if abs(w.x) < 0.9 else Vec3(0, 1, 0)
    v = w.cross(a).normalized()
    u = v.cross(w)

    return (u * x + v * y + w * z).normalized()

def choose_scattering(mat):
    kd = mat.kd.max_component()
    ks = mat.ks.max_component()
    if kd + ks < 1e-6:
        return "none"
    return "diffuse" if random.random() < kd / (kd + ks) else "specular"

def direct_lighting_mis(scene, p, n, material, wo):
    if not scene.has_lights():
        return Vec3(0, 0, 0)

    L_total = Vec3(0, 0, 0)

    for _ in range(N_LIGHT_SAMPLES):
        tri, light_point, light_normal, pdf_light = scene.sample_light()
        if tri is None or pdf_light < 1e-8:
            continue

        wi = light_point - p
        dist2 = wi.dot(wi)
        if dist2 < 1e-8:
            continue
        dist = math.sqrt(dist2)
        wi = wi / dist

        cos_surface = max(0.0, n.dot(wi))
        cos_light = max(0.0, light_normal.dot(-wi))

        if cos_surface == 0.0 or cos_light == 0.0:
            continue

        pdf_brdf = cos_surface / math.pi
        w_mis = (pdf_light * pdf_light) / (pdf_light * pdf_light + pdf_brdf * pdf_brdf + 1e-8)

        eps = 1e-4
        shadow_origin = p + wi * eps
        shadow_ray = Ray(shadow_origin, wi)
        hit = scene.intersect(shadow_ray)

        visible = False
        if hit is not None:
            t_hit, _, _, hit_mat, hit_tri = hit
            if hit_tri.is_light and abs(t_hit - dist) < 1e-4:
                visible = True

        if visible and pdf_light > 0.0:
            Le = tri.material.Le
            brdf = material.kd * (1.0 / math.pi)
            contrib = Le * brdf * cos_surface * cos_light / (dist2 * pdf_light)
            L_total += contrib * w_mis

    return L_total / N_LIGHT_SAMPLES

def radiance(scene, ray, depth):
    if depth > MAX_DEPTH:
        return Vec3(0, 0, 0)

    hit = scene.intersect(ray)
    if hit is None:
        return Vec3(0, 0, 0)

    t, p, n, mat, tri = hit

    if n.dot(ray.direction) > 0:
        n = -n

    if mat.is_emissive():
        return mat.Le

    rr_prob = 0.9 if depth > 2 else 1.0
    if random.random() > rr_prob:
        return Vec3(0, 0, 0)
    rr_factor = 1.0 / rr_prob

    color = Vec3(0, 0, 0)

    event = choose_scattering(mat)

    if event == "diffuse":
        new_dir = cosine_weighted_hemisphere(n)
        cos_theta = max(0.0, n.dot(new_dir))
        if cos_theta <= 0.0:
            return color
        brdf = mat.kd * (1.0 / math.pi)
        pdf = cos_theta / math.pi

    elif event == "specular":
        new_dir = reflect(-ray.direction, n).normalized()
        brdf = mat.ks
        pdf = 1.0
        cos_theta = 1.0

    else:
        return color

    eps = 1e-4
    new_ray = Ray(p + new_dir * eps, new_dir)
    incoming = radiance(scene, new_ray, depth + 1)

    if pdf > 0.0:
        color += brdf * incoming * (cos_theta / pdf) * rr_factor

    if color.x > FIREFLY_THRESHOLD:
        color.x = FIREFLY_THRESHOLD
    if color.y > FIREFLY_THRESHOLD:
        color.y = FIREFLY_THRESHOLD
    if color.z > FIREFLY_THRESHOLD:
        color.z = FIREFLY_THRESHOLD

    return color

def trace_primary(scene, ray):
    hit = scene.intersect(ray)
    if hit is None:
        return (
            Vec3(0, 0, 0),
            Vec3(0, 0, 0),
            Vec3(0, 0, 0),
            0.0,
            Vec3(0, 0, 0),
            -1,
        )

    t, p, n, mat, tri = hit

    if n.dot(ray.direction) > 0:
        n = -n

    depth = t
    obj_id = getattr(tri, "object_id", -1)

    if mat.is_emissive():
        Le = mat.Le
        return Le, Le, Vec3(0, 0, 0), depth, n, obj_id

    wo = -ray.direction
    direct = direct_lighting_mis(scene, p, n, mat, wo)
    indirect = radiance(scene, ray, 1)   # начинаем с первого bounce
    total = direct + indirect

    indirect = Vec3(
        max(0.0, indirect.x),
        max(0.0, indirect.y),
        max(0.0, indirect.z),
    )

    return total, direct, indirect, depth, n, obj_id

