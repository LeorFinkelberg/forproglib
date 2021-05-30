"""
Алгоритмы построения дискретных реализаций стационарного гауссовского
псевдослучайного процесса с заданными типами корреляционных функций
заимствованы из книги Быкова В.В. Цифровое моделирование в статистической
радиотехнике, 1971. 328 с.

Сокращения
----------
КФ: корреляционная функция
ПС: псевдослучайный
ПСП: пседослучайный процесс
ПСЧ: псевдослучайное число
МО: математическое ожидание
"""

from pprint import pprint
import math
import pandas as pd
import numpy as np
import numpy.random as rnd
from helper_funcs_and_class_schema import read_yaml_file, cmd_line_parser


def gauss_with_exp_acf_gen(
    *,
    sigma: float = 2,
    w_star: float = 1.25,
    delta_t: float = 0.05,
    N: int = 1000,
) -> np.array:
    """
    Описание
    --------
    Генерирует дискретную реализацию
    стационарного гауссовского ПСП
    с КФ экспоненциального типа

    Параметры
    ---------
    sigma : стандартное отклонение ординат ПСП.
    w_star : параметр модели ПСП.
    delta_t : шаг по времени.
    N : число отсчетов ПСП.

    Возвращает
    ----------
    xi : массив элементов ПСП с заданной КФ
    """
    gamma_star = w_star * delta_t
    rho = math.exp(-gamma_star)
    b1 = rho
    a0 = sigma * math.sqrt(1 - rho ** 2)

    xi = np.zeros(N)
    xi[0] = rnd.rand()
    x = rnd.randn(N)

    for n in range(1, N):
        xi[n] = a0 * x[n] + b1 * xi[n - 1]

    return xi


def gauss_with_expcos_family_acf_base(
    *,
    a0: float,
    a1: float,
    b1: float,
    b2: float,
    N: int,
) -> np.array:
    """
    Описание
    --------
    Опорный алгоритм построения дискретной реализации ПСП
    с КФ экспоненциально-косинусного семейства

    Параметры
    ----------
    a0 : параметр модели ПСП.
    a1 : параметр модели ПСП.
    b1 : параметр модели ПСП.
    b2 : параметр модели ПСП.
    N : число отсчетов ПСП.

    Возвращает
    -------
    xi : массив элементов ПСП с заданной КФ.
    """
    xi = np.zeros(N)
    # инициализация первых двух элементов массива ПСП
    # ПС числами с равномерным распределеннием
    for i in range(2):
        xi[i] = rnd.rand()

    # массив гауссовских ПСЧ с нулевым МО и единичной дисперсией.
    x = rnd.randn(N)

    for n in range(1, N):
        xi[n] = a0 * x[n] + a1 * x[n - 1] + b1 * xi[n - 1] + b2 * xi[n - 2]

    return xi


def gauss_with_expcos_acf_gen(
    *,
    sigma: float = 2,
    w_star: float = 1.25,
    w0: float = 3,
    delta_t: float = 0.05,
    N: int = 10000,
) -> np.array:
    """
    Описание
    --------
    Генерирует дискретную реализацию
    стационарного гауссовского ПСП
    с КФ экспоненциально-косинусного типа

    Параметры
    ---------
    sigma : стандартное отклонение ПСП.
    w_star : параметр модели ПСП.
    w0 : параметр модели ПСП.
    delta_t : шаг по времени.
    N : число отсчетов ПСП.

    Возвращает
    ----------
    xi : массив элементов ПСП с заданной КФ.
    """
    gamma_star = w_star * delta_t
    gamma0 = w0 * delta_t
    rho = math.exp(-gamma_star)
    alpha0 = rho * (rho ** 2 - 1) * math.cos(gamma0)
    alpha1 = 1 - rho ** 4
    alpha = math.sqrt((alpha1 + math.sqrt(alpha1 ** 2 - 4 * alpha0 ** 2)) / 2)
    a0 = sigma * alpha
    a1 = sigma * alpha0 / alpha
    b1 = 2 * rho * math.cos(gamma0)
    b2 = -(rho ** 2)
    
    params = dict(a0=a0, a1=a1, b1=b1, b2=b2, N=N,)
    xi = gauss_with_expcos_family_acf_base(**params)

    return xi


def gauss_with_expcossin_acf_base(
    *,
    sign: str,
    sigma: float = 2,
    w_star: float = 1.25,
    w0: float = 3,
    delta_t: float = 0.05,
    N: int = 10000,
) -> np.array:
    """
    Описание
    --------
    Опорный алгоритм построения дискретной реализации ПСП
    с КФ экспоненциально-косинусно-синусного типа для
    с учетом знака в сумме гармонических функций КФ

    Параметры
    ----------
    sign: знак в сумме гармонических функций КФ.
    a0 : параметр модели ПСП.
    a1 : параметр модели ПСП.
    b1 : параметр модели ПСП.
    b2 : параметр модели ПСП.
    N : число отсчетов ПСП.

    Возвращает
    -------
    xi : массив элементов ПСП с заданной КФ.
    """
    if sign == "+":
        k1, k2 = 1, 1
    else:
        k1, k2 = -1, -1

    gamma_star = w_star * delta_t
    gamma0 = w0 * delta_t
    rho = math.exp(-gamma_star)
    alpha0 = rho * (rho ** 2 - 1) * math.cos(gamma0) + k1 * w_star / w0 * (
        1 + rho ** 2
    ) * rho * math.sin(gamma0)
    alpha1 = (
        1
        - rho ** 4
        - k2 * 4 * rho ** 2 * w_star / w0 * math.sin(gamma0) * math.cos(gamma0)
    )
    alpha = math.sqrt((alpha1 + math.sqrt(alpha1 ** 2 - 4 * alpha0 ** 2)) / 2)
    a0 = sigma * alpha
    a1 = sigma * alpha0 / alpha
    b1 = 2 * rho * math.cos(gamma_star)
    b2 = -(rho ** 2)

    params = dict(a0=a0, a1=a1, b1=b1, b2=b2, N=N,)
    xi = gauss_with_expcos_family_acf_base(**params)

    return xi


def gauss_with_expcossin_plus_acf_gen(
    *,
    sigma: float = 2,
    w_star: float = 1.25,
    w0: float = 3,
    delta_t: float = 0.05,
    N: int = 1000,
) -> np.array:
    """
    Описание
    --------
    Генерирует дискретную реализацию
    стационарного гауссовского ПСП
    с КФ экспоненциально-косинусно-синусного типа (плюс)

    Параметры
    ---------
    sigma : стандартное отклонение ПСП.
    w_star : параметр модели ПСП.
    w0 : параметр модели ПСП.
    delta_t : шаг по времени.
    N : число отсчетов ПСП.

    Возвращает
    ----------
    xi : массив элементов ПСП с заданной КФ.
    """
    params = dict(sigma=sigma, w_star=w_star, w0=w0, delta_t=delta_t, N=N,)
    xi = gauss_with_expcossin_acf_base(sign="+", **params)

    return xi


def gauss_with_expcossin_minus_acf_gen(
    *,
    sigma: float = 2,
    w_star: float = 1.25,
    w0: float = 3,
    delta_t: float = 0.05,
    N: int = 1000,
) -> np.array:
    """
    Описание
    --------
    Генерирует дискретную реализацию
    стационарного гауссовского ПСП
    с КФ экспоненциально-косинусно-синусного типа (минус)

    Параметры
    ---------
    sigma : стандартное отклонение ПСП.
    w_star : параметр модели ПСП.
    w0 : параметр модели ПСП.
    delta_t : шаг по времени.
    N : число отсчетов ПСП.

    Возвращает
    ----------
    xi : массив элементов ПСП с заданной КФ.
    """
    params = dict(sigma=sigma, w_star=w_star, w0=w0, delta_t=delta_t, N=N,)
    xi = gauss_with_expcossin_acf_base(sign="-", **params)

    return xi



if __name__ == "__main__":
    path_to_config, path_to_figure = cmd_line_parser()
    configs = read_yaml_file(path_to_config)
    pprint(configs)