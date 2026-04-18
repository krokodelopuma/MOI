import numpy as np

# 1. Аналитическое решение
def true_integral():
    return (5**3 - 2**3) / 3
I_true = true_integral()


# 2. Простое интегрирование методом Монте‑Карло
def mc_simple(f, a, b, N):
    x = np.random.uniform(a, b, N)
    return (b - a) * np.mean(f(x))


# 3. Стратификация
def mc_stratified(f, a, b, N, step):
    bins = np.arange(a, b, step)
    if bins[-1] != b:
        bins = np.append(bins, b)

    total = 0
    samples_per_bin = N // (len(bins) - 1)

    for i in range(len(bins) - 1):
        x0 = bins[i]
        x1 = bins[i+1]
        xs = np.random.uniform(x0, x1, samples_per_bin)
        total += (x1 - x0) * np.mean(f(xs))

    return total


# 4. Выборка по значимости

def sample_pdf(pdf, inv_cdf, f, a, b, N):
    u = np.random.rand(N)
    x = inv_cdf(u)
    return np.mean(f(x) / pdf(x))


# p1(x) = x / C1
C1 = 21/2
p1 = lambda x: x / C1
inv1 = lambda u: np.sqrt(21*u + 4)

# p2(x) = x^2 / C2
C2 = 39
p2 = lambda x: x**2 / C2
inv2 = lambda u: (117*u + 8)**(1/3)

# p3(x) = x^3 / C3
C3 = 609/4
p3 = lambda x: x**3 / C3
inv3 = lambda u: (609*u + 16)**0.25


# 5. MIS
def mis(f, pdfs, invs, weights, N):
    M = len(pdfs)
    N_each = N // M
    total = 0

    for i in range(M):
        u = np.random.rand(N_each)
        x = invs[i](u)
        w = weights[i](x)
        total += np.mean(w * f(x) / pdfs[i](x))

    return total


def w_mean(pdf1, pdf2):
    return (
        lambda x: pdf1(x) / (pdf1(x) + pdf2(x)),
        lambda x: pdf2(x) / (pdf1(x) + pdf2(x))
    )

def w_square(pdf1, pdf2):
    return (
        lambda x: pdf1(x)**2 / (pdf1(x)**2 + pdf2(x)**2),
        lambda x: pdf2(x)**2 / (pdf1(x)**2 + pdf2(x)**2)
    )


# 6. Русская рулетка

def russian_roulette(f, a, b, N, R):
    x = np.random.uniform(a, b, N)
    alive = np.random.rand(N) > R
    weights = alive.astype(float) / (1 - R)
    return (b - a) * np.mean(f(x) * weights)




def print_block(title, values):
    print(title)
    print(f"{'N':<10} {'Приближение':<15} {'Погрешность':<15}")
    for N, approx in values:
        print(f"{N:<10} {approx:<15.6f} {abs(approx - I_true):<15.6f}")
    print()



Ns = [100, 1000, 10000, 100000]
f = lambda x: x**2
a, b = 2, 5

print(f"Точное значение интеграла: {I_true:.6f}\n")

# Простой MC
vals_simple = [(N, mc_simple(f, a, b, N)) for N in Ns]
print_block("Простой метод Монте-Карло", vals_simple)

# Стратификация
vals_s1 = [(N, mc_stratified(f, a, b, N, 1)) for N in Ns]
print_block("Стратификация (шаг 1)", vals_s1)

vals_s05 = [(N, mc_stratified(f, a, b, N, 0.5)) for N in Ns]
print_block("Стратификация (шаг 0.5)", vals_s05)

# Importance sampling
vals_p1 = [(N, sample_pdf(p1, inv1, f, a, b, N)) for N in Ns]
print_block("Выборка по значимости p(x)=x", vals_p1)

vals_p2 = [(N, sample_pdf(p2, inv2, f, a, b, N)) for N in Ns]
print_block("Выборка по значимости p(x)=x^2", vals_p2)

vals_p3 = [(N, sample_pdf(p3, inv3, f, a, b, N)) for N in Ns]
print_block("Выборка по значимости p(x)=x^3", vals_p3)

# MIS
w1 = w_mean(p1, p3)
vals_mis_mean = [(N, mis(f, [p1, p3], [inv1, inv3], w1, N)) for N in Ns]
print_block("Многократная важность (средняя плотность)", vals_mis_mean)

w2 = w_square(p1, p3)
vals_mis_square = [(N, mis(f, [p1, p3], [inv1, inv3], w2, N)) for N in Ns]
print_block("Многократная важность (средний квадрат плотности)", vals_mis_square)

# Russian roulette
for R in [0.5, 0.75, 0.95]:
    print(f"Русская рулетка (R={R})")
    print(f"{'N':<10} {'Приближение':<15} {'Погрешность':<15}")
    for N in Ns:
        approx = russian_roulette(f, a, b, N, R)
        error = abs(approx - I_true)
        print(f"{N:<10} {approx:<15.6f} {error:<15.6f}")
    print()

