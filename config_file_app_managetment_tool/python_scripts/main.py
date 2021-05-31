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

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import numpy.random as rnd
from typing import Tuple, NoReturn
from pathlib2 import Path
from helper_funcs_and_class_schema import (
    Params,
    read_yaml_file,
    cmd_line_parser,
)


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

    params = dict(
        a0=a0,
        a1=a1,
        b1=b1,
        b2=b2,
        N=N,
    )
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
    b1 = 2 * rho * math.cos(gamma0)
    b2 = -(rho ** 2)

    params = dict(
        a0=a0,
        a1=a1,
        b1=b1,
        b2=b2,
        N=N,
    )
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
    params = dict(
        sigma=sigma,
        w_star=w_star,
        w0=w0,
        delta_t=delta_t,
        N=N,
    )
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
    params = dict(
        sigma=sigma,
        w_star=w_star,
        w0=w0,
        delta_t=delta_t,
        N=N,
    )
    xi = gauss_with_expcossin_acf_base(sign="-", **params)

    return xi


def exp_acf(
    *,
    sigma: float,
    w_star: float,
    N: int = 150,
) -> Tuple[np.array, np.array]:
    """
    Возвращает массив ординат
    КФ экспоненциального типа
    """
    tau = np.linspace(
        configs.figure_settings.left_xlim_acf, configs.figure_settings.right_xlim_acf, N
    )

    return np.array([sigma ** 2 * math.exp(-w_star * abs(t)) for t in tau]), tau


def exp_cos_acf(
    *,
    sigma: float,
    w_star: float,
    w0: float,
    N: int = 150,
) -> Tuple[np.array, np.array]:
    """
    Возвращает массив ординат
    КФ экспоненциально-косинусного типа
    """
    tau = np.linspace(
        configs.figure_settings.left_xlim_acf, configs.figure_settings.right_xlim_acf, N
    )

    return (
        np.array(
            [sigma ** 2 * math.exp(-w_star * abs(t)) * math.cos(w0 * t) for t in tau]
        ),
        tau,
    )


def exp_cos_sin_acf_base(
    *,
    sign: str,
    sigma: float,
    w_star: float,
    w0: float,
    N: int = 150,
) -> Tuple[np.array, np.array]:
    """
    Возвращает массив ординат
    КФ экспоненциально-косинусно-синусного семейства
    """
    if sign == "+":
        k = 1
    else:
        k = -1

    tau = np.linspace(
        configs.figure_settings.left_xlim_acf, configs.figure_settings.right_xlim_acf, N
    )

    return (
        np.array(
            [
                sigma ** 2
                * math.exp(-w_star * abs(t))
                * (math.cos(w0 * t) + k * w_star / w0 * math.sin(w0 * abs(t)))
                for t in tau
            ]
        ),
        tau,
    )


def gauss01(*, sigma: float, N: int = 150,) -> Tuple[np.array, np.array]:
    """
    Возвращает массив ординат
    гауссовой плотности распределения
    с нулевым МО и единичной дисперсией
    """
    xi = np.linspace(
        configs.figure_settings.left_xlim_pdf,
        configs.figure_settings.right_xlim_pdf,
        N,
    )
    return np.array([
        1 / ( sigma * math.sqrt(2 * math.pi) ) * math.exp( - x**2 / ( 2 * sigma**2 ) )
        for x in xi
    ]), xi


def draw_graph(
    configs: Params,
    process: np.array,
    acf: np.array,
    tau: np.array,
    output_fig_path: str,
) -> NoReturn:
    """
    Отображает КФ, реализацию ПСП с заданным типом КФ,
    а также точки выбороса по различным критериям
    """
    abspath_to_output_fig = Path(output_fig_path).absolute()
    abspath_to_output_fig_dir = abspath_to_output_fig.parents[0]

    if not abspath_to_output_fig_dir.exists():
        Path.mkdir(abspath_to_output_fig_dir)

    fig = plt.figure(
        figsize=(
            configs.figure_settings.width_main_fig, 
            configs.figure_settings.height_main_fig
            )
        )
    grid = plt.GridSpec(3, 4, wspace=0.35, hspace=0.45)

    sns.set_context(
        "paper",
        rc={
            "font.size": 11,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        },
    )

    ax1 = plt.subplot(grid[:2, :])
    ax2 = plt.subplot(grid[2, :1])
    ax3 = plt.subplot(grid[2, 1:2])
    ax4 = plt.subplot(grid[2, 2:])

    process = pd.Series(process)
    process_ma = process.rolling(window=configs.window_width).mean()
    process_std = process.rolling(window=configs.window_width).std()
    acf = pd.Series(acf)
    gauss01_pdf, gauss01_xi = gauss01(sigma=configs.sigma)
    
    hline_params = dict(y=0, xmin=0, xmax=configs.N, lw=1., alpha=0.7, color=configs.colors.grey)

    ax1.plot(
        process,
        label=(
            "Реализация гауссовского процесса \n"
            f"с КФ {acf_types[configs.kind_acf]} типа"
        ),
        color=configs.colors.pearl_night,
    )

    if configs.visibility.ma_show == True:
        ax1.plot(
            process_ma, label=("Cкользящее среднее"), color=configs.colors.terakotta
        )
    ax1.plot(
        process_ma + 3 * process_std,
        label=("Скользящее станадартное \nотклонение с коэффициентом 3"),
        color=configs.colors.krayola_green,
    )
    ax1.plot(
        process_ma - 3 * process_std, label=None, color=configs.colors.krayola_green
    )
    
    ax1.set_title(
        ("Сводка по анализу выбросов в "
        "стационарном гауссовском процессе")
        , fontsize=12
    )
    ax1.axhline(**hline_params)
    ax1.legend(loc=1)
    ax1.set_xlabel("временная координата " + r"$ t $")
    ax1.set_ylabel("ординаты процесса " + r"$ \xi $")
    ax1.set_xticks(range(0, configs.N + 1, 100))
    
    ax2.set_title("Плотность распределения \nординат процесса")
    process.plot.kde(ax=ax2, label = "Эмпир-ская", color=configs.colors.pearl_night)  
    ax2.plot(gauss01_xi, gauss01_pdf, label = "Теор-ская", color=configs.colors.terakotta)
    ax2.set_xlabel("ординаты " + r"$ \xi $")
    ax2.set_ylabel(r"$ f(\xi) $")
    ax2.set_xlim(
        configs.figure_settings.left_xlim_pdf,
        configs.figure_settings.right_xlim_pdf,
    )
    ax2.legend(loc=4)

    ax3.set_title(f"КФ {acf_types[configs.kind_acf]} типа")
    ax3.plot(tau, acf, color=configs.colors.pearl_night)
    ax3.axhline(**hline_params)
    ax3.set_xlabel("сдвиг " + r"$ \tau $")
    ax3.set_ylabel(r"$ K_{\xi}(\tau) $")
    ax3.set_xlim(
        configs.figure_settings.left_xlim_acf,
        configs.figure_settings.right_xlim_acf
    )

    
    fig.savefig(abspath_to_output_fig, dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    path_to_config, path_to_output_figure = cmd_line_parser()
    configs: Params = read_yaml_file(path_to_config)

    acf_types = {
        1: "экспоненциального",
        2: "экспоненциально-\nкосинусного",
        3: "экспоненциально-косинусно\n-синусного (плюс)",
        4: "экспоненциально-косинусно-\nсинусного (минус)",
    }

    if configs.kind_acf == 1:  # КФ экспоненциального типа
        gauss_process = gauss_with_exp_acf_gen(
            sigma=configs.sigma,
            w_star=configs.w_star,
            delta_t=configs.delta_t,
            N=configs.N,
        )
        acf, tau = exp_acf(sigma=configs.sigma, w_star=configs.w_star)
    elif configs.kind_acf == 2:  # КФ экспоненциально-косинусного типа
        gauss_process = gauss_with_expcos_acf_gen(
            sigma=configs.sigma,
            w_star=configs.w_star,
            w0=configs.w0,
            delta_t=configs.delta_t,
            N=configs.N,
        )
        acf, tau = exp_cos_acf(
            sigma=configs.sigma, w_star=configs.w_star, w0=configs.w0
        )
    elif configs.kind_acf == 3:  # КФ экспоненциально-косинусно-синусного типа (плюс)
        gauss_process = gauss_with_expcossin_plus_acf_gen(
            sigma=configs.sigma,
            w_star=configs.w_star,
            w0=configs.w0,
            delta_t=configs.delta_t,
            N=configs.N,
        )
        acf, tau = exp_cos_sin_acf_base(
            sign="+", sigma=configs.sigma, w_star=configs.w_star, w0=configs.w0
        )
    elif configs.kind_acf == 4:  # КФ экспоненциально-косинусно-синусного типа (минус)
        gauss_process = gauss_with_expcossin_minus_acf_gen(
            sigma=configs.sigma,
            w_star=configs.w_star,
            w0=configs.w0,
            delta_t=configs.delta_t,
            N=configs.N,
        )
        acf, tau = exp_cos_sin_acf_base(
            sign="-", sigma=configs.sigma, w_star=configs.w_star, w0=configs.w0
        )

    draw_graph(configs, gauss_process, acf, tau, path_to_output_figure)
