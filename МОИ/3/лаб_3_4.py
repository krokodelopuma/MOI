import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)
n = 100000

u = np.random.rand(n)
v = np.random.rand(n)

theta = np.arcsin(u)
phi = 2 * np.pi * v

sin_theta = np.sin(theta)
cos_theta = np.cos(theta)

x = sin_theta * np.cos(phi)
y = sin_theta * np.sin(phi)
z = cos_theta

# Проверка
norms = np.linalg.norm(np.column_stack((x, y, z)), axis=1)
print(f"Все векторы единичные: {np.allclose(norms, 1)}")
print(f"Все z ≥ 0: {np.all(z >= 0)}")
print()


fig = plt.figure(figsize=(14, 5))

ax1 = fig.add_subplot(131, projection='3d')
sample = np.random.choice(n, 5000, replace=False)
ax1.scatter(x[sample], y[sample], z[sample],
            s=1, alpha=0.3, c=z[sample], cmap='viridis')
ax1.set_title('Косинусное распределение\nна полусфере')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_box_aspect([1, 1, 1])

ax2 = fig.add_subplot(132)
ax2.scatter(x[sample], y[sample], s=1, alpha=0.3, c=z[sample], cmap='viridis')
ax2.set_title('Вид сверху (проекция XY)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_aspect('equal')

ax3 = fig.add_subplot(133)
theta_rad = np.arccos(z)
ax3.hist(theta_rad, bins=100, density=True, alpha=0.7,
         label='метод', color='steelblue', edgecolor='black')

theta_lin = np.linspace(0, np.pi/2, 100)
p_theory = np.cos(theta_lin)
ax3.plot(theta_lin, p_theory, 'r-', lw=2, label='Теория')
ax3.set_xlabel(r'Полярный угол θ (рад)')
ax3.set_ylabel('Плотность')
ax3.set_title('Распределение угла θ')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cosine_method_correct.png', dpi=150)
plt.show()
