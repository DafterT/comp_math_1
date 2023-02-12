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
import matplotlib.pyplot as plt
from prettytable import PrettyTable


# Функция для отрисовки таблицы
def print_interpolation_table(x_find, y_real, y_lagrange, y_spline):
    pt = PrettyTable()
    pt.add_column('x', np.round(x_find, 4))
    pt.add_column('real y', y_real)
    pt.add_column('lagrange y', y_lagrange)
    pt.add_column('spline y', y_spline)
    print(pt)


# Функция для отрисовки одного графика
def print_one_graph(x, y, title, id, count_graphs):
    plt.subplot(1, count_graphs, id)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.title(title)
    plt.plot(x, y, '-o')


# Функция для отрисовки всех графиков
def print_interpolation_graph(x_find, y_real, y_lagrange, y_spline):
    plt.figure(figsize=(15, 4))
    print_one_graph(x_find, y_real, 'Исходный график', 1, 3)
    print_one_graph(x_find, y_lagrange, 'График Лагранжем', 2, 3)
    print_one_graph(x_find, y_spline, 'График сплайном', 3, 3)
    plt.savefig("Graphs.jpg")
    plt.show()


# Функция для отрисовки погрешности
def print_interpolation_error(x_interpolation, x_find, y_lagrange_error, y_spline_error):
    plt.figure(figsize=(15, 4))
    # Добавляем узлы т.к. при интерполяции в них погрешность нулевая
    x = np.arange(0, 1 + 0.05, 0.05)
    # Добавляем нулевые узлы на график
    y_lagrange = [0 if i % 2 == 0 else y_lagrange_error[i // 2] for i in range(0, len(x_interpolation) + len(x_find))]
    y_spline = [0 if i % 2 == 0 else y_spline_error[i // 2] for i in range(0, len(x_interpolation) + len(x_find))]
    # Собственно сам график
    print_one_graph(x, y_lagrange, 'Погрешность Лагранжем', 1, 2)
    print_one_graph(x, y_spline, 'Погрешность сплайном', 2, 2)
    plt.savefig("Error.jpg")
    plt.show()


# Выполнение задачи интерполяции
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
    # Отрисовка
    print_interpolation_table(x_find, y_real, y_lagrange, y_spline)
    print_interpolation_graph(x_find, y_real, y_lagrange, y_spline)
    print_interpolation_error(x_interpolation, x_find, y_lagrange - y_real, y_spline - y_real)


# Функция, которая используется вместо quanc8
def pseudo_quanc8(func, start, end):
    return quad(func, start, end, limit=30)


# Получение функции интеграла
def get_integrate_function(m):
    def integrate_function(x):
        return (np.abs(x - np.tan(x))) ** m

    return integrate_function


# Выполнение задачи интегрирования
def do_integrate(approximation, integrate_function):
    inconvenient_point = 4.49340945790906
    integral = pseudo_quanc8(integrate_function, 2, inconvenient_point - approximation) + \
               pseudo_quanc8(integrate_function, inconvenient_point + approximation, 5)
    return integral[0]


# Функция для подсчета интегралов с различным приближением к бесконечности
def do_integration_with_various_approximations(approximation_array, m):
    """Эта функция считает значение интеграла с различным приближением к точке разрыва"""
    integrate_function = get_integrate_function(m)
    integration_return = []
    for i in approximation_array:
        integration_return.append(do_integrate(i, integrate_function))
    return integration_return


# Нарисовать таблицу с интегрированием
def print_integration_table(approximation, array_with_m_05, array_with_m_1):
    pt = PrettyTable()
    approximation_decimal = [f'{i:.0e}' for i in approximation]
    pt.add_column('Approximation', approximation_decimal)
    pt.add_column('Integral, m=-0.5', array_with_m_05)
    pt.add_column('Integral, m=-1.0', array_with_m_1)
    print(pt)


# Выполнение задачи интегрирования
def integration():
    approximation_array = [10 ** (-i) for i in range(20)]
    print_integration_table(
        approximation_array,
        do_integration_with_various_approximations(approximation_array, -0.5),
        do_integration_with_various_approximations(approximation_array, -1.0)
    )


def main():
    interpolation()
    integration()


if __name__ == '__main__':
    main()
