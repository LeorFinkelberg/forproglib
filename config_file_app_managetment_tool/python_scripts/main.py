"""
Алгоритмы построения дискретных реализаций стационарного гауссовского
псевдослучайного процесса с заданными типами корреляционных функций
заимствованы из книги Быкова В.В. Цифровое моделирование в статистической
радиотехнике, 1971. 328 с.
"""

import math
import sys
from typing import NoReturn, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rnd
import pandas as pd
import seaborn as sns
from helper_funcs_and_class_schema import (
    Params,
    cmd_line_parser,
    read_yaml_file,
)
from pathlib2 import Path


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
    N: int = 250,
) -> Tuple[np.array, np.array]:
    """
    Возвращает массив ординат
    КФ экспоненциального типа
    """
    tau = np.linspace(
        configs.figure_settings.left_xlim_acf,
        configs.figure_settings.right_xlim_acf,
        N,
    )

    return (
        np.array([sigma ** 2 * math.exp(-w_star * abs(t)) for t in tau]),
        tau,
    )


def exp_cos_acf(
    *,
    sigma: float,
    w_star: float,
    w0: float,
    N: int = 250,
) -> Tuple[np.array, np.array]:
    """
    Возвращает массив ординат
    КФ экспоненциально-косинусного типа
    """
    tau = np.linspace(
        configs.figure_settings.left_xlim_acf,
        configs.figure_settings.right_xlim_acf,
        N,
    )

    return (
        np.array(
            [
                sigma ** 2 * math.exp(-w_star * abs(t)) * math.cos(w0 * t)
                for t in tau
            ]
        ),
        tau,
    )


def exp_cos_sin_acf_base(
    *,
    sign: str,
    sigma: float,
    w_star: float,
    w0: float,
    N: int = 250,
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
        configs.figure_settings.left_xlim_acf,
        configs.figure_settings.right_xlim_acf,
        N,
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


def gauss01(
    *,
    sigma: float,
    N: int = 250,
) -> Tuple[np.array, np.array]:
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
    return (
        np.array(
            [
                1
                / (sigma * math.sqrt(2 * math.pi))
                * math.exp(-(x ** 2) / (2 * sigma ** 2))
                for x in xi
            ]
        ),
        xi,
    )


def Zscore(
    data: np.array,
    threshold: float = 3.0,
) -> Tuple[tuple, np.array]:
    """
    Вычисляет классическую Z-оценку
    на выборочных средних.
    NB: На практике следует применять с большой осторожностью
    """
    mean = np.mean(data)
    std = np.std(data)
    Zscore = (data - mean) / std

    indexes = np.where((Zscore > threshold) | (Zscore < -threshold))
    outliers = data[indexes]

    return indexes, outliers


def modified_Zscore_median(
    data: np.array,
    threshold: float = 3.5,
) -> Tuple[tuple, np.array]:
    """
    Вычисляет робастную Z-оценку на медианах
    """
    scale_factor = 0.6745
    median = np.median(data)
    MAD = np.median(np.abs(data - median))
    modif_Zscore = scale_factor * (data - median) / MAD

    indexes = np.where(
        (modif_Zscore > threshold) | (modif_Zscore < -threshold)
    )
    outliers = data[indexes]

    return indexes, outliers


def outlier_label_maker(
    *,
    ax,
    indexes: np.array,
    outliers: np.array,
    delta_x: float = -35,
):
    for idx, (x, y) in enumerate(zip(indexes, outliers)):
        ax.text(
            x + delta_x,
            y,
            s=f"{outliers[idx]:.2f}",
            color=configs.colors.outliers_red,
        )


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
            configs.figure_settings.height_main_fig,
        )
    )
    grid = plt.GridSpec(2, 5, wspace=0.35, hspace=0.42)

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

    ax1 = plt.subplot(grid[:, :-1])
    ax2 = plt.subplot(grid[0, -1:])
    ax3 = plt.subplot(grid[1, -1:])

    process = pd.Series(process)
    process_ma = process.rolling(window=configs.window_width).mean()
    process_std = process.rolling(window=configs.window_width).std()
    acf = pd.Series(acf)
    gauss01_pdf, gauss01_xi = gauss01(sigma=configs.sigma)
    idx_Zscore, out_Zscore = Zscore(process.to_numpy())
    idx_modif_Zscore, out_modif_Zscore = modified_Zscore_median(
        process.to_numpy()
    )

    ax1.plot(
        process,
        label=("Реализация гауссовского процесса"),
        color=configs.colors.pearl_night,
    )

    ax1.scatter(
        idx_Zscore[0],
        out_Zscore,
        label="Выбросы по классической Z-оценке",
        s=50,
        marker="s",
        color=configs.colors.outliers_red,
    )
    outlier_label_maker(ax=ax1, indexes=idx_Zscore[0], outliers=out_Zscore)

    ax1.scatter(
        idx_modif_Zscore[0],
        out_modif_Zscore,
        label="Выбросы по модифицированной \nустойчивой Z-оценке",
        s=50,
        marker="o",
        color=configs.colors.outliers_red,
    )
    outlier_label_maker(
        ax=ax1, indexes=idx_modif_Zscore[0], outliers=out_modif_Zscore
    )

    if configs.visibility.ma_show == True:
        ax1.plot(
            process_ma,
            label=("Cкользящее среднее"),
            color=configs.colors.terakotta,
        )
    ax1.plot(
        process_ma + 2.5 * process_std,
        label=("Скользящее станадартное \nотклонение с коэффициентом 2.5"),
        color=configs.colors.krayola_green,
    )
    ax1.plot(
        process_ma - 2.5 * process_std,
        label=None,
        color=configs.colors.krayola_green,
    )

    ax1.set_title(
        (
            "Сводка по анализу выбросов в стационарном гауссовском \n"
            f"псевдослучайном процессе с КФ {acf_types[configs.kind_acf]} типа"
        ),
        fontsize=12,
    )
    ax1.set_facecolor(configs.colors.white)
    ax1.legend(loc="best", frameon=False)
    ax1.set_xlabel("временная координата " + r"$ t $")
    ax1.set_ylabel("ординаты процесса " + r"$ \xi $")
    ax1.set_xticks(range(0, configs.N + 1, 100))

    ax2.set_title("Плотность распределения \nординат процесса")
    process.plot.kde(
        ax=ax2, label="Эмпирическая", color=configs.colors.pearl_night
    )
    ax2.plot(
        gauss01_xi,
        gauss01_pdf,
        label="Теоретическая",
        color=configs.colors.terakotta,
    )
    ax2.set_xlabel("ординаты " + r"$ \xi $")
    ax2.set_ylabel(r"$ f(\xi) $")
    ax2.set_xlim(
        configs.figure_settings.left_xlim_pdf,
        configs.figure_settings.right_xlim_pdf,
    )
    ax2.set_facecolor(configs.colors.white)
    ax2.legend(loc=3, frameon=False)

    ax3.set_title(f"КФ {acf_types[configs.kind_acf]} типа")
    ax3.plot(tau, acf, color=configs.colors.terakotta)
    ax3.set_xlabel("сдвиг " + r"$ \tau $")
    ax3.set_ylabel(r"$ K_{\xi}(\tau) $")
    ax3.set_xlim(
        configs.figure_settings.left_xlim_acf,
        configs.figure_settings.right_xlim_acf,
    )
    ax3.set_facecolor(configs.colors.white)

    fig.savefig(
        abspath_to_output_fig, dpi=350, bbox_inches="tight", pad_inches=0.25
    )


if __name__ == "__main__":
    path_to_config, path_to_output_figure = cmd_line_parser()
    configs: Params = read_yaml_file(path_to_config)

    acf_types = {
        1: "экспоненциального",
        2: "экспоненциально-косинусного",
        3: "экспоненциально-косинусно-\nсинусного (плюс)",
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
    elif (
        configs.kind_acf == 3
    ):  # КФ экспоненциально-косинусно-синусного типа (плюс)
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
    elif (
        configs.kind_acf == 4
    ):  # КФ экспоненциально-косинусно-синусного типа (минус)
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
