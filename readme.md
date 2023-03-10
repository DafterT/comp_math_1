## Первая лабораторная работа по вычислительной математике в СПбПУ
___
### 11 Вариант:

Для функции `f(x) = 1 / (1 + x)` по узлам `x_k = 0.1k (k = 0, 1, ..., 10)` построить
полином Лагранжа L(x) 10-й степени и сплайн-функцию S(x).
Вычислить значения всех трех функций в точках `y_k = 0.05 + 0.1k (k = 0, 1, ..., 9)`.
Результаты отобразить графически.

Используя программу QUANC8, вычислить два интеграла:
от 2 до 5 `(abs(x - tg(x)))^m dx`, при m = -1 и m = -0.5
___
Задачу можно разделить на 2:
1. Интерполяция Лагранжем и сплайнами
2. Подсчет значений интеграла, используя QUANC8

Для выполнения первой задачи будем использовать методы, встроенные 
в библиотеку `scipy`, а конкретно: 
* `interpolate.lagrange` - для подсчета Лагранжа
* `interpolate.CubicSpline` - для подсчета сплайна

Вывод информации о результатах: `prettytable` для вывода информации в консоль
и `matplotlib.pyplot` для отрисовки графиков.

Для второй задачи используем `scipy.integrate.quad` в качестве альтернативы QUANC8.
Так как наш интеграл имеет разрыв в точке ~4.49340945790906 придется считать сначала
слева от точки разрыва, потом справа.