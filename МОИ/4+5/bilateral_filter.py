# bilateral_filter.py

import math
from vec3 import Vec3


def gaussian(x, sigma):
    return math.exp(-(x * x) / (2.0 * sigma * sigma))


def bilateral_filter_color(
        width,
        height,
        color_buffer,
        normal_buffer,
        depth_buffer,
        object_id_buffer,
        sigma_s=2.0,
        sigma_r=0.2,
        sigma_n=0.3,
        sigma_d=0.5
):
    out = [Vec3(0, 0, 0) for _ in range(width * height)]

    R = int(3 * sigma_s)
    spatial_kernel = {}
    sum_w_s = 0.0
    for dy in range(-R, R + 1):
        for dx in range(-R, R + 1):
            dist_xy = math.sqrt(dx * dx + dy * dy)
            w_s = gaussian(dist_xy, sigma_s)
            spatial_kernel[(dx, dy)] = w_s
            sum_w_s += w_s

    if sum_w_s < 1e-8:
        sum_w_s = 1.0

    for k in spatial_kernel:
        spatial_kernel[k] /= sum_w_s

    for y in range(height):
        for x in range(width):
            idx = y * width + x

            base_color = color_buffer[idx]
            base_normal = normal_buffer[idx]
            base_depth = depth_buffer[idx]
            base_obj = object_id_buffer[idx]

            if base_obj < 0:
                out[idx] = base_color
                continue

            n0 = base_normal.normalized()

            Wp = 0.0
            col_sum = Vec3(0, 0, 0)

            for dy in range(-R, R + 1):
                yy = y + dy
                if yy < 0 or yy >= height:
                    continue

                for dx in range(-R, R + 1):
                    xx = x + dx
                    if xx < 0 or xx >= width:
                        continue

                    j = yy * width + xx

                    if object_id_buffer[j] != base_obj:
                        continue

                    w_s_norm = spatial_kernel[(dx, dy)]

                    diff_c = (color_buffer[j] - base_color).length()
                    w_r = gaussian(diff_c, sigma_r)

                    n1 = normal_buffer[j].normalized()
                    cos_dn = max(-1.0, min(1.0, n0.dot(n1)))
                    dn = 1.0 - cos_dn
                    w_n = gaussian(dn, sigma_n)

                    dd = abs(depth_buffer[j] - base_depth)
                    w_d = gaussian(dd, sigma_d)

                    w_range = w_r * w_n * w_d
                    w = w_s_norm * w_range

                    Wp += w
                    col_sum += color_buffer[j] * w

            if Wp > 1e-8:
                out[idx] = col_sum / Wp
            else:
                out[idx] = base_color

    return out


def bilateral_median_filter(
        width,
        height,
        color_buffer,
        normal_buffer,
        depth_buffer,
        object_id_buffer,
        radius=3,
        iterations=2
):
    from statistics import median

    out = color_buffer[:]

    for _ in range(iterations):
        temp = [Vec3(0, 0, 0) for _ in range(width * height)]

        for y in range(height):
            for x in range(width):
                idx = y * width + x
                obj = object_id_buffer[idx]

                if obj < 0:
                    temp[idx] = out[idx]
                    continue

                rs, gs, bs = [], [], []

                for dy in range(-radius, radius + 1):
                    yy = y + dy
                    if yy < 0 or yy >= height:
                        continue

                    for dx in range(-radius, radius + 1):
                        xx = x + dx
                        if xx < 0 or xx >= width:
                            continue

                        j = yy * width + xx
                        if object_id_buffer[j] != obj:
                            continue

                        c = out[j]
                        rs.append(c.x)
                        gs.append(c.y)
                        bs.append(c.z)

                if rs:
                    temp[idx] = Vec3(median(rs), median(gs), median(bs))
                else:
                    temp[idx] = out[idx]

        # нормировка по объектам
        objects = {}
        for i, oid in enumerate(object_id_buffer):
            if oid < 0:
                continue
            objects.setdefault(oid, []).append(i)

        for oid, indices in objects.items():
            sum_original = Vec3(0, 0, 0)
            sum_filtered = Vec3(0, 0, 0)

            for i in indices:
                sum_original += color_buffer[i]
                sum_filtered += temp[i]

            kx = sum_original.x / sum_filtered.x if abs(sum_filtered.x) > 1e-8 else 1.0
            ky = sum_original.y / sum_filtered.y if abs(sum_filtered.y) > 1e-8 else 1.0
            kz = sum_original.z / sum_filtered.z if abs(sum_filtered.z) > 1e-8 else 1.0

            for i in indices:
                c = temp[i]
                temp[i] = Vec3(c.x * kx, c.y * ky, c.z * kz)

        out = temp

    return out
