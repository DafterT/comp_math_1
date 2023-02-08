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


def pseudo_quanc8(func, start, end):
    return quad(func, start, end, limit=30)


def get_integrate_function(m):
    def integrate_function(x):
        return (np.abs(x - np.tan(x))) ** m

    return integrate_function


def do_integrate(m, approximation):
    inconvenient_point = 4.49340945790906
    function = get_integrate_function(m)
    integral = pseudo_quanc8(function, 2, inconvenient_point - approximation) + \
               pseudo_quanc8(function, inconvenient_point + approximation, 5)
    return integral[0]


def integration():
    print(do_integrate(-1.0, 2e-9))
    print(do_integrate(-0.5, 1e-6))


def main():
    interpolation()
    integration()


if __name__ == '__main__':
    main()
