# main.py
# 4
# 5 билатеральная фильтрация (среднее и медианное)

import sys
import time
import random
import statistics
from concurrent.futures import ProcessPoolExecutor
from PIL import Image

from vec3 import Vec3
from camera import Camera
from obj_loader import load_obj_to_scene
from path_tracer import trace_primary
from bilateral_filter import (
    bilateral_filter_color,
    bilateral_median_filter
)

def save_png(filename, width, height, pixels):
    img = Image.new("RGB", (width, height))
    data = []
    for c in pixels:
        r, g, b = c.to_rgb8()
        data.append((r, g, b))
    img.putdata(data)
    img.save(filename, "PNG")

def compute_scale(pixels):
    luminances = []
    for c in pixels:
        lum = 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z
        luminances.append(lum)
    median_lum = statistics.median(luminances)
    if median_lum < 1e-8:
        median_lum = 1.0
    target_median = 0.5
    scale = target_median / median_lum
    print(f"Median luminance = {median_lum:.6f}, Scale = {scale:.4f}")
    return scale

def postprocess(pixels, scale):
    out = []
    for c in pixels:
        cc = c * scale
        cc = cc.clamp(0.0, 1.0)
        cc = cc.gamma_correct(2.2)
        out.append(cc)
    return out

def render_row(y, width, height, spp, camera, scene):
    row_total = []
    row_direct = []
    row_indirect = []
    row_depth = []
    row_normal = []
    row_object = []

    for x in range(width):
        total = Vec3(0, 0, 0)
        direct = Vec3(0, 0, 0)
        indirect = Vec3(0, 0, 0)
        depth = 0.0
        normal = Vec3(0, 0, 0)
        object_id = -1

        for s in range(spp):
            u = (x + random.random()) / width
            v = (y + random.random()) / height
            # Исправление переворота: v = 1.0 - v
            ray = camera.get_ray(u, 1.0 - v)
            tcol, dcol, icol, dep, nrm, oid = trace_primary(scene, ray)
            total += tcol
            direct += dcol
            indirect += icol
            depth += dep
            normal += nrm
            if s == 0:
                object_id = oid

        total /= spp
        direct /= spp
        indirect /= spp
        depth /= spp
        normal = (normal / spp).normalized()

        row_total.append(total)
        row_direct.append(direct)
        row_indirect.append(indirect)
        row_depth.append(depth)
        row_normal.append(normal)
        row_object.append(object_id)

    return (y, row_total, row_direct, row_indirect,
            row_depth, row_normal, row_object)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py Low-Poly_Models.obj")
        return

    obj_file = sys.argv[1]
    width = 1000
    height = 1000
    samples_per_pixel = 64

    print("Loading scene")
    scene = load_obj_to_scene(obj_file)

    lookfrom = Vec3(0.0, 3.0, 10.0)
    lookat = Vec3(0.0, 1.0, 0.0)
    vup = Vec3(0.0, 1.0, 0.0)
    fov = 60.0
    aspect = width / height

    camera = Camera(lookfrom, lookat, vup, fov, aspect)

    total_pixels = [Vec3(0,0,0) for _ in range(width*height)]
    direct_pixels = [Vec3(0,0,0) for _ in range(width*height)]
    indirect_pixels = [Vec3(0,0,0) for _ in range(width*height)]
    depth_buffer = [0.0 for _ in range(width*height)]
    normal_buffer = [Vec3(0,0,0) for _ in range(width*height)]
    object_id_buffer = [-1 for _ in range(width*height)]

    print("ЗАДАНИЕ 4")
    start_time = time.time()

    with ProcessPoolExecutor() as executor:
        futures = []
        for y in range(height):
            futures.append(executor.submit(render_row, y, width, height,
                                           samples_per_pixel, camera, scene))

        done = 0
        for f in futures:
            y, rt, rd, ri, rdep, rnorm, robj = f.result()
            for x in range(width):
                idx = y * width + x
                total_pixels[idx] = rt[x]
                direct_pixels[idx] = rd[x]
                indirect_pixels[idx] = ri[x]
                depth_buffer[idx] = rdep[x]
                normal_buffer[idx] = rnorm[x]
                object_id_buffer[idx] = robj[x]
            done += 1
            if done % max(1, height//10) == 0:
                print(f"{int(100*done/height)}%")

    end_time = time.time()
    print(f"Рендеринг завершён за {end_time - start_time:.2f} сек")

    scale_raw = compute_scale(total_pixels)
    total_pp = postprocess(total_pixels, scale_raw)

    save_png("01_общая_яркость.png", width, height, total_pp)


    print("ЗАДАНИЕ 5")

    filtered_indirect_avg = bilateral_filter_color(
        width, height, indirect_pixels,
        normal_buffer, depth_buffer, object_id_buffer,
        sigma_s=3.0, sigma_r=0.1, sigma_n=0.2, sigma_d=0.3
    )
    filtered_indirect_median = bilateral_median_filter(
        width, height, indirect_pixels,
        normal_buffer, depth_buffer, object_id_buffer, radius=4
    )

    final_avg = []
    final_median = []
    for i in range(width*height):
        final_avg.append(direct_pixels[i] + filtered_indirect_avg[i])
        final_median.append(direct_pixels[i] + filtered_indirect_median[i])

    scale_final = compute_scale(final_avg)
    final_avg_pp = postprocess(final_avg, scale_final)
    final_median_pp = postprocess(final_median, scale_final)

    save_png("02_билатеральное_среднее.png", width, height, final_avg_pp)
    save_png("03_билатеральное_медианное.png", width, height, final_median_pp)

    print("Готово. Файлы:")
    print("01_общая_яркость.png")
    print("02_билатеральное_среднее.png")
    print("03_билатеральное_медианное.png")

if __name__ == "__main__":
    main()