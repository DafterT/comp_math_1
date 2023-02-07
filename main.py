"""
Для функции f(x) = 1/(1+x) по узлам xk = 0.1k (k=0,1,..., 10) построить
полином Лагранжа L(x) 10-й степени и сплайн-функцию S(x).
Вычислить значения всех трех функций в точках y_k = 0.05 +0.1k (k=0,1,..., 9).
Результаты отобразить графически.
Используя программу QUANC8, вычислить два интеграла:
от 2 до 5 (abs(x - tg(x)))^m dx, при m=-1 и m=-0.5
"""
import numpy as np
from scipy.integrate import quad
from scipy.interpolate import lagrange, CubicSpline


def parody_quanc8(func):
    integral = quad(func, 2, 5, limit=30)
    return integral


def interpolation():
    # Заполнили массив узлов интерполяции
    x_interpolation = np.arange(0, 1 + 0.1, 0.1)
    x_find = np.arange(0.05, 1, 0.1)
    # Заполнили f(x)
    y_interpolation = 1 / (1 + x_interpolation)
    y_real = 1 / (1 + x_find)
    # Строим интерполяционный полином Лагранжа
    lagrange_interpolation = lagrange(x_interpolation, y_interpolation)
    y_lagrange = lagrange_interpolation(x_find)
    # Строим сплайн интерполяцию
    spline_interpolation = CubicSpline(x_interpolation, y_interpolation)
    y_spline = spline_interpolation(x_find)
    print(y_lagrange - y_real, '\n', y_spline - y_real)


def main():
    interpolation()


if __name__ == '__main__':
    main()
