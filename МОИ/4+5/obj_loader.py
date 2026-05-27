# obj_loader.py
from vec3 import Vec3
from material import Material
from triangle import Triangle
from scene import Scene

def load_obj_to_scene(filename: str) -> Scene:
    scene = Scene()

    vertices = []
    normals = []
    materials = {}
    current_mtl = None
    current_object_id = 0

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append(Vec3(float(x), float(y), float(z)))
                continue

            if line.startswith("vn "):
                _, x, y, z = line.split()[:4]
                normals.append(Vec3(float(x), float(y), float(z)))
                continue

            if line.startswith("newmtl "):
                name = line.split()[1]
                materials[name] = Material()
                current_mtl = name
                continue

            if line.startswith("o "):
                current_object_id += 1
                continue

            if line.startswith("usemtl "):
                name = line.split()[1]
                current_mtl = name
                if name not in materials:
                    materials[name] = Material()
                continue

            if line.startswith("Kd "):
                if current_mtl is None:
                    continue
                _, r, g, b = line.split()[:4]
                kd = Vec3(float(r), float(g), float(b))
                mat = materials[current_mtl]

                if "Light" in current_mtl:
                    mat.Le = kd
                    mat.kd = Vec3(0, 0, 0)
                    mat.ks = Vec3(0, 0, 0)
                else:
                    mat.kd = kd
                continue

            if line.startswith("Ks "):
                if current_mtl is None:
                    continue
                _, r, g, b = line.split()[:4]
                ks = Vec3(float(r), float(g), float(b))
                mat = materials[current_mtl]

                if "Light" not in current_mtl:
                    mat.ks = ks
                continue

            if line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) != 3:
                    continue

                v_idx = []
                n_idx = []
                valid = True

                for p in parts:
                    tokens = p.split("/")

                    if not tokens[0].isdigit():
                        valid = False
                        break
                    vi = int(tokens[0]) - 1
                    if not (0 <= vi < len(vertices)):
                        valid = False
                        break
                    v_idx.append(vi)

                    ni = None
                    if len(tokens) >= 3 and tokens[2].isdigit():
                        tmp = int(tokens[2]) - 1
                        if 0 <= tmp < len(normals):
                            ni = tmp
                    n_idx.append(ni)

                if not valid:
                    continue

                v0 = vertices[v_idx[0]]
                v1 = vertices[v_idx[1]]
                v2 = vertices[v_idx[2]]

                if any(ni is None for ni in n_idx):
                    n = (v1 - v0).cross(v2 - v0).normalized()
                else:
                    n0 = normals[n_idx[0]]
                    n1 = normals[n_idx[1]]
                    n2 = normals[n_idx[2]]
                    n = ((n0 + n1 + n2) / 3.0).normalized()

                mat = materials.get(current_mtl, Material())
                is_light = mat.is_emissive()

                tri = Triangle(v0, v1, v2, n, mat, is_light=is_light)
                tri.object_id = current_object_id
                scene.add_triangle(tri)

    return scene