import numpy as np
import matplotlib.pyplot as plt


def generate_points_in_circle(Rc, center, n_points=100000, normal=np.array([0, 0, 1])):
    # Нормализуем вектор нормали
    N = normal / np.linalg.norm(normal)
    C = np.array(center)

    # Строим ортонормированный базис (U, V) в плоскости круга
    W = np.array([1.0, 0.0, 0.0]) if abs(N[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    U = np.cross(N, W)
    U /= np.linalg.norm(U)
    V = np.cross(N, U)

    # Генерация случайных точек
    u = np.random.rand(n_points)
    v = np.random.rand(n_points)

    r = Rc * np.sqrt(u)
    theta = 2 * np.pi * v

    # Координаты в плоскости круга
    P_local = (r * np.cos(theta))[:, None] * U + (r * np.sin(theta))[:, None] * V
    P = C + P_local

    return P


def count_points_in_circles(P, circles):
    """Подсчет точек в каждой окружности"""
    counts = []
    for circle in circles:
        center = circle['center']
        radius = circle['radius']
        distances = np.linalg.norm(P[:, :2] - center, axis=1)
        count = np.sum(distances < radius)
        counts.append(count)
    return np.array(counts)  # Возвращаем numpy массив


def print_statistics_circle(P, Rc, center, circles, counts):
    """Вывод статистики для круга"""
    # Площадь большого круга
    big_circle_area = np.pi * Rc ** 2
    small_circle_area = np.pi * circles[0]['radius'] ** 2
    expected_per_circle = len(P) * small_circle_area / big_circle_area
    print(f"Ожидаемое количество точек в одной окружности: {expected_per_circle:.1f}")

    print(f"\nРезультаты подсчета ({len(P)} точек):")
    print(f"{'Окружность':<20} {'Точек':<10} {'Процент':<10} {'Отклонение':<15} {'Статус':<15}")

    mean_count = np.mean(counts)
    std_count = np.std(counts)

    for i, (circle, cnt) in enumerate(zip(circles, counts)):
        percentage = cnt / len(P) * 100
        deviation = (cnt - expected_per_circle) / expected_per_circle * 100
        status = "норма" if abs(deviation) < 15 else "отклонение"
        print(f"{circle['name']:<20} {cnt:<10} {percentage:<10.4f} {deviation:+.1f}%{'':<8} {status}")



def visualize_circle_with_regions(P, Rc, center, circles):
    """Визуализация круга и малых окружностей"""
    fig = plt.figure(figsize=(15, 5))

    # 1. Основная визуализация
    ax1 = fig.add_subplot(1, 3, 1)

    sample_size = min(5000, len(P))
    ax1.scatter(P[:sample_size, 0], P[:sample_size, 1], s=0.5, alpha=0.5, c='blue', label='Случайные точки')

    # Рисуем большой круг
    big_circle = plt.Circle(center[:2], Rc, fill=False, edgecolor='black', linewidth=2, label='Круг')
    ax1.add_patch(big_circle)

    # Рисуем малые окружности
    colors = plt.cm.tab10(np.linspace(0, 1, len(circles)))
    for i, circle in enumerate(circles):
        circle_patch = plt.Circle(circle['center'], circle['radius'],
                                  fill=False, color=colors[i], linewidth=2,
                                  linestyle='-', label=circle['name'])
        ax1.add_patch(circle_patch)
        ax1.scatter(circle['center'][0], circle['center'][1],
                    c=[colors[i]], s=50, marker='x', linewidths=2)

    # Отмечаем центр большого круга
    ax1.scatter(center[0], center[1], c='red', s=100, marker='o', label='Центр круга')

    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Равномерное распределение точек в круге\nс 6 тестовыми окружностями')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    ax1.set_xlim(center[0] - Rc - 0.5, center[0] + Rc + 0.5)
    ax1.set_ylim(center[1] - Rc - 0.5, center[1] + Rc + 0.5)

    # 2. 3D визуализация
    ax2 = fig.add_subplot(1, 3, 2, projection='3d')

    # Создаем точки для 3D визуализации
    ax2.scatter(P[:sample_size, 0], P[:sample_size, 1], P[:sample_size, 2],
                s=0.5, alpha=0.5, c='blue')

    # Рисуем круг в 3D
    theta_3d = np.linspace(0, 2 * np.pi, 100)
    x_circle = center[0] + Rc * np.cos(theta_3d)
    y_circle = center[1] + Rc * np.sin(theta_3d)
    z_circle = np.zeros_like(theta_3d)
    ax2.plot(x_circle, y_circle, z_circle, 'r-', linewidth=2, label='Круг')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('3D визуализация')
    ax2.legend()

    # 3. Гистограмма
    ax3 = fig.add_subplot(1, 3, 3)

    # Ожидаемое значение
    expected = len(P) * (np.pi * circles[0]['radius'] ** 2) / (np.pi * Rc ** 2)

    bars = ax3.bar(range(len(counts)), counts, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axhline(expected, color='red', linestyle='--', linewidth=2, label=f'Ожидаемое: {expected:.0f}')
    ax3.axhline(np.mean(counts), color='green', linestyle=':', linewidth=2, label=f'Среднее: {np.mean(counts):.0f}')

    for bar, cnt in zip(bars, counts):
        ax3.text(bar.get_x() + bar.get_width() / 2, cnt + max(counts) * 0.01,
                 str(cnt), ha='center', va='bottom', fontsize=9)

    ax3.set_xlabel('Окружности')
    ax3.set_ylabel('Количество точек')
    ax3.set_title('Сравнение количества точек\nв тестовых окружностях')
    ax3.set_xticks(range(len(circle_names)))
    ax3.set_xticklabels(circle_names, rotation=45, ha='right', fontsize=8)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()


def generate_test_circles_in_circle(Rc, center, radius, num_circles=6):
    """Генерация тестовых окружностей внутри большого круга"""
    circles = []

    test_positions = [
        (0.0, 0.0),
        (Rc / 2, 0.0),
        (-Rc / 2, 0.0),
        (0.0, Rc / 2),
        (0.0, -Rc / 2),
        (Rc / 2, Rc / 2)
    ]

    for i, (dx, dy) in enumerate(test_positions[:num_circles]):
        # Проверяем, что окружность полностью внутри большого круга
        circle_center = np.array([center[0] + dx, center[1] + dy])
        dist_from_big_center = np.linalg.norm(circle_center - center[:2])

        # Окружность должна быть полностью внутри
        if dist_from_big_center + radius <= Rc:
            circles.append({
                'center': circle_center,
                'radius': radius,
                'name': f'Область {i + 1}'
            })

    return circles



Rc = 5.0
center = np.array([0.0, 0.0, 0.0])
n_points = 100000

P = generate_points_in_circle(Rc, center, n_points)
test_radius = 0.8
test_circles = generate_test_circles_in_circle(Rc, center, test_radius, num_circles=6)

if len(test_circles) == 6:
    counts = count_points_in_circles(P, test_circles)
    circle_names = [circle['name'] for circle in test_circles]
    print_statistics_circle(P, Rc, center, test_circles, counts)
    visualize_circle_with_regions(P, Rc, center, test_circles)
else:
    print(f" Не удалось разместить 6 окружностей, размещено только {len(test_circles)}")