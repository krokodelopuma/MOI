import numpy as np
from matplotlib import pyplot as plt


def sample_points_in_triangle(V1, V2, V3, n=100000):
    u = np.random.rand(n)
    v = np.random.rand(n)
    r = np.sqrt(u)
    a = 1.0 - r
    b = r * (1.0 - v)
    c = r * v
    P = a[:, None] * V1 + b[:, None] * V2 + c[:, None] * V3
    return P, a, b, c


def barycentric_from_point(P, V1, V2, V3):
    """
    Вычисление барицентрических координат
    Работает как с одной точкой, так и с массивом точек
    """
    # Преобразуем в двумерный массив, если это одна точка
    P = np.atleast_2d(P)

    # Барицентрические координаты
    E1 = V2 - V1
    E2 = V3 - V1
    # Матрица базиса (3×2) каждый столбец - базисный вектор
    A = np.stack([E1, E2], axis=1)
    # Решаем для каждой точки P: R = b*E1 + c*E2 (базис)
    R = P - V1
    # Псевдообратная матрица: M = (A^T·A)^{-1}·A^T (размер 2×3)
    # Для всех точек сразу: [b,c]^T = M · (P - V1)^T
    At = A.T
    M = np.linalg.inv(At @ A) @ At  # размер (2,3)
    bg = (M @ R.T).T  # размер (N,2)
    b = bg[:, 0]
    c = bg[:, 1]
    a = 1.0 - b - c

    # Если была одна точка, возвращаем скаляры
    if len(a) == 1:
        return a[0], b[0], c[0]
    return a, b, c


def point_in_triangle(P, V1, V2, V3):
    """Проверка, находится ли точка внутри треугольника"""
    a, b, c = barycentric_from_point(P, V1, V2, V3)
    # Если это одна точка
    if isinstance(a, (int, float)):
        return a >= -1e-6 and b >= -1e-6 and c >= -1e-6
    # Если это массив точек
    return (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)


def distance_from_point_to_line(point, line_start, line_end):
    """Расстояние от точки до отрезка (в 2D)"""
    # Векторы
    line_vec = line_end - line_start
    point_vec = point - line_start

    # Длина отрезка
    line_len = np.linalg.norm(line_vec)
    if line_len == 0:
        return np.linalg.norm(point_vec)

    # Проекция точки на прямую
    line_unit = line_vec / line_len
    projection = np.dot(point_vec, line_unit)

    # Находим ближайшую точку на отрезке
    if projection <= 0:
        closest_point = line_start
    elif projection >= line_len:
        closest_point = line_end
    else:
        closest_point = line_start + projection * line_unit

    return np.linalg.norm(point - closest_point)


def is_circle_inside_triangle(center, radius, V1, V2, V3):
    """Проверка, что окружность полностью внутри треугольника"""
    # Проверяем, что центр внутри треугольника
    center_inside = point_in_triangle(np.array([center[0], center[1], 0]), V1, V2, V3)
    if not center_inside:
        return False

    # Проверяем расстояния до всех сторон
    dist1 = distance_from_point_to_line(center, V1[:2], V2[:2])
    dist2 = distance_from_point_to_line(center, V2[:2], V3[:2])
    dist3 = distance_from_point_to_line(center, V3[:2], V1[:2])

    # Окружность внутри, если расстояние до каждой стороны >= радиусу
    return dist1 >= radius - 1e-6 and dist2 >= radius - 1e-6 and dist3 >= radius - 1e-6


def generate_random_circles(V1, V2, V3, radius, num_circles=6):
    """Генерация случайных окружностей внутри треугольника"""
    circles = []
    max_attempts = 10000

    for i in range(num_circles):
        attempts = 0
        while attempts < max_attempts:
            # Генерируем случайную точку внутри треугольника
            u = np.random.rand()
            v = np.random.rand()
            if u + v > 1:
                u, v = 1 - u, 1 - v

            # Преобразуем в координаты
            center = u * V2[:2] + v * V3[:2] + (1 - u - v) * V1[:2]

            # Проверяем, что окружность полностью внутри треугольника
            if is_circle_inside_triangle(center, radius, V1, V2, V3):
                # Проверяем, что окружности не пересекаются
                overlap = False
                for existing_circle in circles:
                    dist = np.linalg.norm(center - existing_circle['center'])
                    if dist < 2 * radius:  # Пересекаются или касаются
                        overlap = True
                        break

                if not overlap:
                    circles.append({
                        'center': center,
                        'radius': radius,
                        'name': f'Окружность {i + 1}'
                    })
                    break

            attempts += 1

        if attempts == max_attempts:
            print(f"Предупреждение: Не удалось разместить окружность {i + 1}")
            break

    return circles


def count_points_in_circles(P, circles):
    """Подсчет точек в каждой окружности"""
    counts = []
    for circle in circles:
        center = circle['center']
        radius = circle['radius']
        distances = np.linalg.norm(P[:, :2] - center, axis=1)
        count = np.sum(distances < radius)
        counts.append(count)
    return counts


def print_statistics(P, V1, V2, V3, circles, counts):
    """Вывод статистики"""
    # Площадь треугольника
    triangle_area = 0.5 * np.linalg.norm(np.cross(V2 - V1, V3 - V1))
    circle_area = np.pi * circles[0]['radius'] ** 2
    expected_per_circle = len(P) * circle_area / triangle_area

    print("\n" + "=" * 80)
    print(f"   Вершины: V1{V1[:2]}, V2{V2[:2]}, V3{V3[:2]}")
    print(f"   Площадь треугольника: {triangle_area:.4f}")
    print(f"   Радиус каждой окружности: {circles[0]['radius']:.3f}")
    print(f"   Ожидаемое количество точек в одной окружности: {expected_per_circle:.1f}")
    print(f"{'Окружность':<20} {'Точек':<10} {'Процент':<10} {'Отклонение':<15} {'Статус':<15}")

    mean_count = np.mean(counts)
    std_count = np.std(counts)

    for i, (circle, cnt) in enumerate(zip(circles, counts)):
        percentage = cnt / len(P) * 100
        deviation = (cnt - expected_per_circle) / expected_per_circle * 100
        status = "норма" if abs(deviation) < 15 else "отклонение"
        print(f"{circle['name']:<20} {cnt:<10} {percentage:<10.2f} {deviation:+.1f}%{'':<8} {status}")

def visualize_triangle_with_circles(V1, V2, V3, P, circles):
    """Визуализация треугольника и окружностей"""
    fig = plt.figure(figsize=(15, 5))

    # 1. 2D визуализация
    ax1 = fig.add_subplot(1, 3, 1)

    # Рисуем треугольник
    triangle = np.array([V1[:2], V2[:2], V3[:2], V1[:2]])
    ax1.plot(triangle[:, 0], triangle[:, 1], 'r-', linewidth=2, label='Треугольник')

    # Рисуем точки
    sample_size = min(5000, len(P))
    ax1.scatter(P[:sample_size, 0], P[:sample_size, 1], s=0.5, alpha=0.5, c='blue', label='Случайные точки')

    # Рисуем окружности
    colors = plt.cm.tab10(np.linspace(0, 1, len(circles)))
    for i, circle in enumerate(circles):
        circle_patch = plt.Circle(circle['center'], circle['radius'],
                                  fill=False, color=colors[i], linewidth=2,
                                  linestyle='-', label=circle['name'])
        ax1.add_patch(circle_patch)
        ax1.scatter(circle['center'][0], circle['center'][1],
                    c=[colors[i]], s=50, marker='x', linewidths=2)

    # Отмечаем вершины
    ax1.scatter(V1[0], V1[1], c='red', s=100, marker='o', label='V₁')
    ax1.scatter(V2[0], V2[1], c='green', s=100, marker='o', label='V₂')
    ax1.scatter(V3[0], V3[1], c='orange', s=100, marker='o', label='V₃')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Равномерное распределение точек в треугольнике\nс 6 окружностями')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')

    # 2. 3D визуализация
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    # Рисуем треугольник в 3D
    ax2.plot_trisurf([V1[0], V2[0], V3[0]],
                     [V1[1], V2[1], V3[1]],
                     [V1[2], V2[2], V3[2]],
                     alpha=0.3, color='red')

    # Рисуем ребра треугольника
    edges = [[V1, V2], [V2, V3], [V3, V1]]
    for edge in edges:
        ax2.plot3D([edge[0][0], edge[1][0]],
                   [edge[0][1], edge[1][1]],
                   [edge[0][2], edge[1][2]], 'r-', linewidth=2)

    # Рисуем точки
    ax2.scatter(P[:sample_size, 0], P[:sample_size, 1], P[:sample_size, 2],
                s=0.5, alpha=0.5, c='blue')

    # Отмечаем вершины
    ax2.scatter([V1[0], V2[0], V3[0]],
                [V1[1], V2[1], V3[1]],
                [V1[2], V2[2], V3[2]],
                c=['red', 'green', 'orange'], s=100)

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D визуализация')

    # 3. Гистограмма
    ax3 = fig.add_subplot(1, 3, 3)
    counts = count_points_in_circles(P, circles)
    circle_names = [circle['name'] for circle in circles]

    # Ожидаемое значение
    triangle_area = 0.5 * np.linalg.norm(np.cross(V2 - V1, V3 - V1))
    circle_area = np.pi * circles[0]['radius'] ** 2
    expected = len(P) * circle_area / triangle_area

    bars = ax3.bar(range(len(counts)), counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axhline(expected, color='red', linestyle='--', linewidth=2, label=f'Ожидаемое: {expected:.0f}')
    ax3.axhline(np.mean(counts), color='green', linestyle=':', linewidth=2, label=f'Среднее: {np.mean(counts):.0f}')

    for bar, cnt in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, cnt + max(counts) * 0.01,
                 str(cnt), ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('Окружности')
    ax3.set_ylabel('Количество точек')
    ax3.set_title('Сравнение количества точек\nв окружностях')
    ax3.set_xticks(range(len(circle_names)))
    ax3.set_xticklabels(circle_names, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()



V1 = np.array([0.0, 0.0, 0.0])
V2 = np.array([10.0, 0.0, 0.0])
V3 = np.array([5.0, 9.0, 0.0])

n_points = 100000
P, a_gen, b_gen, g_gen = sample_points_in_triangle(V1, V2, V3, n=n_points)

a_rec, b_rec, g_rec = barycentric_from_point(P, V1, V2, V3)
inside = (a_rec >= -1e-6) & (b_rec >= -1e-6) & (g_rec >= -1e-6)
print(f"Доля точек внутри треугольника: {inside.mean():.6f}")
print(f"Всего сгенерировано точек: {n_points}")
print(f"Точек внутри треугольника: {inside.sum()}")

radius = 0.8

circles = generate_random_circles(V1, V2, V3, radius, num_circles=6)

if len(circles) == 6:
    counts = count_points_in_circles(P, circles)
    print_statistics(P, V1, V2, V3, circles, counts)
    visualize_triangle_with_circles(V1, V2, V3, P, circles)

