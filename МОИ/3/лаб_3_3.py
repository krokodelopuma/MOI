import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Параметры
N_POINTS = 100000
RNG = np.random.default_rng(42)  # для воспроизводимости


def generate_uniform_sphere_points(n_points):

    # Равномерное распределение по высоте
    z = RNG.uniform(-1, 1, n_points)
    # Равномерное распределение по азимуту
    phi = RNG.uniform(0, 2 * np.pi, n_points)
    # Декартовы координаты
    r_xy = np.sqrt(1 - z ** 2)
    x = r_xy * np.cos(phi)
    y = r_xy * np.sin(phi)

    return np.column_stack([x, y, z])


def count_points_in_cone(points, axis, half_angle_deg):
    """
    Подсчет точек внутри конуса с заданной осью и половинным углом
    """
    half_angle_rad = np.radians(half_angle_deg)
    cos_angle = np.cos(half_angle_rad)

    # Нормализуем ось конуса
    axis = axis / np.linalg.norm(axis)

    # Скалярное произведение с осью
    dot_products = points @ axis

    # Точка внутри конуса, если угол между вектором и осью < half_angle
    # cos(угла) = dot(p, axis) > cos(half_angle)
    inside = dot_products > cos_angle

    return np.sum(inside)


def draw_cone_boundary(ax, axis, half_angle_deg, color, alpha=0.3):
    """
    Рисование границы конуса на сфере
    """
    half_angle_rad = np.radians(half_angle_deg)
    axis = axis / np.linalg.norm(axis)

    # Строим ортонормированный базис в плоскости, перпендикулярной оси
    # Находим любой вектор, не коллинеарный оси
    if abs(axis[0]) < 0.9:
        temp = np.array([1, 0, 0])
    else:
        temp = np.array([0, 1, 0])

    u = np.cross(axis, temp)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)

    # Параметризуем окружность - границу конуса на сфере
    # Вектор от центра до точки на границе конуса:
    # direction = cos(α)·axis + sin(α)·(cos(t)·u + sin(t)·v)
    t = np.linspace(0, 2 * np.pi, 100)
    sin_alpha = np.sin(half_angle_rad)
    cos_alpha = np.cos(half_angle_rad)

    circle_points = []
    for tt in t:
        point = cos_alpha * axis + sin_alpha * (np.cos(tt) * u + np.sin(tt) * v)
        circle_points.append(point)

    circle_points = np.array(circle_points)
    ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2],
            color=color, linewidth=2, alpha=alpha)


def task3_sphere():

    # Генерация точек
    points = generate_uniform_sphere_points(N_POINTS)
    # Параметры конусов
    CONE_ANGLE_DEG = 25
    cos_angle = np.cos(np.radians(CONE_ANGLE_DEG))

    # Телесный угол конуса: Ω = 2π(1 - cos α)
    solid_angle = 2 * np.pi * (1 - cos_angle)
    expected_count = N_POINTS * solid_angle / (4 * np.pi)

    print(f"Половинный угол конуса: {CONE_ANGLE_DEG}°")
    print(f"Телесный угол конуса: {solid_angle:.4f} ср")
    print(f"Ожидаемое число точек в конусе: {expected_count:.1f}")
    print()

    # Определяем конусы в разных направлениях
    cones = [
        {"axis": [0, 0, 1], "name": "Северный полюс", "color": "crimson"},
        {"axis": [0, 0, -1], "name": "Южный полюс", "color": "darkorange"},
        {"axis": [1, 0, 0], "name": "Экватор (X+)", "color": "forestgreen"},
        {"axis": [0, 1, 0], "name": "Экватор (Y+)", "color": "royalblue"},
        {"axis": [-1, 0, 0], "name": "Экватор (X-)", "color": "purple"},
        {"axis": [1, 1, 0], "name": "Экваториальная диагональ", "color": "goldenrod"},
        {"axis": [1, 1, 1], "name": "Пространственная диагональ", "color": "hotpink"},
    ]

    # Подсчет точек в каждом конусе
    results = []
    for cone in cones:
        count = count_points_in_cone(points, cone["axis"], CONE_ANGLE_DEG)
        results.append(count)
        cone["count"] = count
        print(f"{cone['name']:25} : {count:5d} точек  (ожидалось ~{expected_count:.0f})")

    print()
    print(f"Отклонения: {np.std(results):.1f} (≈ {100 * np.std(results) / expected_count:.1f}%)")

    # Визуализация
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle("Равномерное распределение точек на сфере\n"
                 "Проверка через телесные углы",
                 fontsize=14, fontweight='bold')

    # 3D визуализация
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.set_title(f"Точки на сфере и границы конусов (α = {CONE_ANGLE_DEG}°)")

    # Отображаем все точки полупрозрачно
    # Берем подвыборку для лучшей видимости
    sample_idx = np.random.choice(N_POINTS, min(3000, N_POINTS), replace=False)
    ax1.scatter(points[sample_idx, 0],
                points[sample_idx, 1],
                points[sample_idx, 2],
                s=1, alpha=0.3, color='gray')

    # Рисуем границы конусов и выделяем точки внутри
    for cone in cones:
        # Граница конуса
        draw_cone_boundary(ax1, cone["axis"], CONE_ANGLE_DEG, cone["color"], alpha=0.6)

        # Точки внутри этого конуса (подвыборка)
        axis_norm = cone["axis"] / np.linalg.norm(cone["axis"])
        inside = points[sample_idx] @ axis_norm > cos_angle
        if np.any(inside):
            ax1.scatter(points[sample_idx][inside, 0],
                        points[sample_idx][inside, 1],
                        points[sample_idx][inside, 2],
                        s=8, alpha=0.6, color=cone["color"],
                        label=f"{cone['name']} (n={cone['count']})")

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)

    # Столбчатая диаграмма
    ax2 = fig.add_subplot(122)
    names = [cone["name"] for cone in cones]
    counts = results
    colors = [cone["color"] for cone in cones]

    bars = ax2.bar(range(len(names)), counts, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1)
    ax2.axhline(expected_count, color='red', linestyle='--',
                linewidth=2, label=f'Ожидаемое значение ({expected_count:.0f})')

    # Добавляем значения на столбцы
    for bar, count in zip(bars, counts):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_xlabel('Направление конуса')
    ax2.set_ylabel('Количество точек')
    ax2.set_title(f'Сравнение числа точек в конусах\n'
                  f'Телесный угол Ω = {solid_angle:.3f} ср\n'
                  f'Стандартное отклонение: {np.std(counts):.1f}')
    ax2.set_xticks(range(len(names)))
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


    return points, results


# Запуск
if __name__ == "__main__":
    points, counts = task3_sphere()