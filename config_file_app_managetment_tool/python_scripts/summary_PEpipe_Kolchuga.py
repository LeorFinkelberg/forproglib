import argparse
import logging
import math
import sys
from collections import namedtuple
from typing import Dict, NoReturn, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from pathlib2 import Path, PosixPath, WindowsPath


class DataFilepathNotExists(Exception):
    """
    Возбуждает исключение, если файл с данными не найден
    """


class ConfigPathNotExists(Exception):
    """
    Возбуждает исключение, если конфигурационный файл не найден
    """


class UnknownSeparatorInPath(Exception):
    """
    Возбуждает исключение, если в пути встречается
    неизвестный разделитель
    """


PathWinOrLinux = Union[WindowsPath, PosixPath]

file_log = logging.FileHandler("logs.log")
console_out = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    handlers=(file_log, console_out),
    format=("[%(asctime)s | %(levelname)s]: %(message)s"),
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


def preprocessing_paths(
    params: namedtuple,
) -> Tuple[PathWinOrLinux, PathWinOrLinux]:
    """
    Формирует пути
    """
    from_project_dir_figures_dir_path = PROJECT_DIR.joinpath(
        Path(params.FIGURES_DIRNAME)
    )
    data_filepath = PROJECT_DIR.joinpath(Path(params.DATA_FILENAME))
    try:
        if not data_filepath.exists():
            raise DataFilepathNotExists(
                f"Ошибка! Файл `{data_filepath.name}` не найден!"
            )
    except DataFilepathNotExists as err:
        logging.error(f"{err}")
        sys.exit()
    else:
        if not from_project_dir_figures_dir_path.exists():
            Path.mkdir(from_project_dir_figures_dir_path)
        output_fig_filepath = PROJECT_DIR.joinpath(
            from_project_dir_figures_dir_path,
            Path(params.OUTPUT_FIG_FILENAME),
        )
        return data_filepath, output_fig_filepath


def read_config(config_filename: str) -> Dict:
    config_path = PROJECT_DIR.joinpath(Path(config_filename))
    try:
        if not config_path.exists():
            raise ConfigPathNotExists(
                f"Ошибка! Конфигурационный файл `{config_path.name}` не найден!"
            )
    except ConfigPathNotExists as err:
        logging.error(f"{err}")
        sys.exit()
    else:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            logging.info(f"Конфигурационный файл `{config_filename}` успешно прочитан!")
        return config


def read_data(data_filepath: Union[WindowsPath, PosixPath]) -> pd.DataFrame:
    """
    Читает csv-файл. Эти данные используется для вычисления стоимости
    ЗСП Кольчуга
    """
    with open(data_filepath) as f:
        data = pd.read_csv(f, delimiter=";")
    return data


def MOP(*, dn: float, c0: float, e: float) -> float:
    return 2 * params.MRS / (params.c0 * (dn / e - 1))


def MOP_Kolchuga(*, dn: float, k0: float, c0: float, e: float) -> float:
    return 2 * params.MRS * k0 / (c0 * (dn / e - 1))


def m_PE(*, dn: float, e: float) -> float:
    return params.ro_PE * math.pi / 4 * (dn ** 2 - (dn - 2 * e) ** 2) * params.l_pipe


def cost_PE(*, dn: float, e: float) -> float:
    return params.price_PE * m_PE(dn=dn, e=e)


def cost_Kolchuga(data: pd.DataFrame, *, dn: float, n: int, label: str) -> float:
    query_dn = data.query(f"`D,mm` == {int(dn*1000)}")
    total_price_ribbon = query_dn[label].values[0]
    return n * math.pi * dn * total_price_ribbon


def cost_PE_Kolchuga(
    data: pd.DataFrame, *, cost_kolchuga: float, dn: float, e: float
) -> float:
    return cost_PE(dn=dn, e=e) + cost_kolchuga


def plot_MOP_emin(ax) -> NoReturn:
    """
    Строит зависимость МОР от минимальной толщины стенки трубы
    """
    e_range = np.linspace(params.e0, params.e1)

    MOP_down_range = [MOP(dn=params.dn, c0=params.c0, e=e) for e in e_range]

    MOP_k_2mm_down_range = [
        MOP_Kolchuga(dn=params.dn, k0=params.k_2mm, c0=params.c0, e=e) for e in e_range
    ]

    MOP_k_4mm_down_range = [
        MOP_Kolchuga(dn=params.dn, k0=params.k_4mm, c0=params.c0, e=e) for e in e_range
    ]

    ax.plot(e_range, MOP_down_range, label="ПЭ труба", color=params.PE_COLOR)
    ax.plot(
        e_range,
        MOP_k_2mm_down_range,
        label="ПЭ труба + ЗСП 'Кольчуга' (2 слоя)",
        color=params.KOLCHUGA_2MM_COLOR,
    )
    ax.plot(
        e_range,
        MOP_k_4mm_down_range,
        label="ПЭ труба + ЗСП 'Кольчуга' (4 слоя)",
        color=params.KOLCHUGA_4MM_COLOR,
    )


def plot_cost_emin(ax, data: pd.DataFrame) -> NoReturn:
    """
    Строит зависимость стоимости от минимальной толщины стенки трубы
    """
    e_range = np.linspace(params.e0, params.e1)
    cost_PE_range = [cost_PE(dn=params.dn, e=e) for e in e_range]

    cost_Kolchuga_2_mm_zsp = cost_Kolchuga(data, dn=params.dn, n=2, label="cost_zsp")
    cost_PE_Kolchuga_2_mm_range_material = [
        cost_PE_Kolchuga(data, cost_kolchuga=cost_Kolchuga_2_mm_zsp, dn=params.dn, e=e)
        for e in e_range
    ]

    cost_Kolchuga_4_mm_zsp = cost_Kolchuga(data, dn=params.dn, n=4, label="cost_zsp")
    cost_PE_Kolchuga_4_mm_range_material = [
        cost_PE_Kolchuga(data, cost_kolchuga=cost_Kolchuga_4_mm_zsp, dn=params.dn, e=e)
        for e in e_range
    ]

    cost_Kolchuga_2_mm_total = cost_Kolchuga(data, dn=params.dn, n=2, label="total")
    cost_PE_Kolchuga_2_mm_range_total = [
        cost_PE_Kolchuga(
            data, cost_kolchuga=cost_Kolchuga_2_mm_total, dn=params.dn, e=e
        )
        for e in e_range
    ]

    cost_Kolchuga_4_mm_total = cost_Kolchuga(data, dn=params.dn, n=4, label="total")
    cost_PE_Kolchuga_4_mm_range_total = [
        cost_PE_Kolchuga(
            data, cost_kolchuga=cost_Kolchuga_4_mm_total, dn=params.dn, e=e
        )
        for e in e_range
    ]

    dashes = (10, 6)
    ax.plot(
        e_range,
        cost_PE_range,
        dashes=dashes,
        ls="--",
        label="Стоимость ПЭ",
        color=params.PE_COLOR,
    )

    ax.plot(
        e_range,
        cost_PE_Kolchuga_2_mm_range_material,
        dashes=dashes,
        ls="--",
        label="Стоимость ПЭ + сырье ЗСП 'Кольчуга' (2 слоя)",
        color=params.KOLCHUGA_2MM_COLOR,
    )

    ax.plot(
        e_range,
        cost_PE_Kolchuga_4_mm_range_material,
        dashes=dashes,
        ls="--",
        label="Стоимость ПЭ + сырье ЗСП 'Кольчуга' (4 слоя)",
        color=params.KOLCHUGA_4MM_COLOR,
    )

    ax.plot(
        e_range,
        cost_PE_Kolchuga_2_mm_range_total,
        ls="-.",
        label="Стоимость ПЭ + продукт ЗСП 'Кольчуга' (2 слоя)",
        color=params.KOLCHUGA_2MM_COLOR,
    )

    ax.plot(
        e_range,
        cost_PE_Kolchuga_4_mm_range_total,
        ls="-.",
        label="Стоимость ПЭ + продукт ЗСП 'Кольчуга' (4 слоя)",
        color=params.KOLCHUGA_4MM_COLOR,
    )


def plot_added_graph_elements(ax) -> NoReturn:
    """
    Строит дополнительные элементы графики
    """
    ax.axhline(params.MOP_normative, lw=1.0, color=params.VERTIC_LINES_POINT_COLOR)
    ax.scatter(
        params.e_SDR,
        params.MOP_normative,
        s=45,
        c=params.VERTIC_LINES_POINT_COLOR,
        marker="o",
    )
    ax.text(
        params.TEXT_OFFSET_X * params.e_SDR,
        params.TEXT_OFFSET_Y * params.MOP_normative,
        s=f"MOP={params.MOP_normative}, [МПа]",
    )
    ax.vlines(
        params.xvalue_SDR,
        ymin=0,
        ymax=5,
        color=params.VERTIC_LINES_POINT_COLOR,
        alpha=0.8,
        lw=1,
        ls="-",
    )
    ax.set_xlim(
        params.xvalue_SDR[0] - params.MARGIN_LEFT,
        params.xvalue_SDR[-1] + params.MARGIN_RIGHT,
    )
    ax.set_ylim(0, params.Y_UPLIM_LEFT)


def plot_top_axis(ax) -> NoReturn:
    """
    Строит метки на верхней оси абсцисс
    """
    ax.set_xlabel("SDR: стандартное размерное соотношение")
    ax.set_xticks(params.xvalue_SDR)
    ax.set_xticklabels(params.xlabel_SDR, rotation=0, ha="center")


def plot_right_axis(ax) -> NoReturn:
    """
    Строит метки на правой оси ординат
    """
    ax.tick_params(axis="y", labelcolor=params.BLACK_COLOR)
    ax.set_ylabel("C: стоимость, [руб.]", color=params.BLACK_COLOR)
    ax.set_yticks(
        np.arange(0, params.Y_UPLIM_RIGTH + params.Y_STEP_RIGTH, params.Y_STEP_RIGTH)
    )
    ax.set_ylim(0, params.Y_UPLIM_RIGTH)
    ax.legend(title="C - $e_{min}$", loc=4)


def plot_main_graph(ax) -> NoReturn:
    """
    Строит основые графические элементы
    """
    ax.tick_params(axis="y", labelcolor=params.BLACK_COLOR)
    ax.set_title(  # главный заголовок рисунка
        "Для наружного номинального диаметра" f" ПЭ трубы $d_n$ = {params.dn:.3f}, [м]"
    )
    ax.set_xlabel("$e_{min}$: толщина стенки ПЭ трубы, [м]")
    ax.set_xticks(params.xvalue_SDR)
    ax.set_xticklabels(
        [
            f"{value:.4f}" if label else ""
            for value, label in zip(params.xvalue_SDR, params.xlabel_SDR)
        ]
    )
    ax.set_ylabel(
        "MOP: максимальное рабочее\nдавление, [МПа]", color=params.BLACK_COLOR
    )
    ax.set_yticks(
        np.arange(0.0, params.Y_UPLIM_LEFT + params.Y_STEP_LEFT, params.Y_STEP_LEFT)
    )
    ax.legend(title="MOP - $e_{min}$", loc=2)


def params_count(config: Dict) -> int:
    """
    Подсчитывает количество параметров, описанных в конфигурационном файле
    """
    return sum(len(config[key]) for key in config.keys())


def init_param(config: Dict) -> namedtuple:
    """
    Задает значения параметров на основе конфигурационного файла
    """
    Params = namedtuple(  # именованный кортеж параметров
        "Params",
        (
            "DATA_FILENAME",
            "OUTPUT_FIG_FILENAME",
            "FIGURES_DIRNAME",
            "BACKGROUND_COLOR_FIG",
            "PE_COLOR",
            "KOLCHUGA_2MM_COLOR",
            "KOLCHUGA_4MM_COLOR",
            "VERTIC_LINES_POINT_COLOR",
            "BLACK_COLOR",
            "MARGIN_LEFT",
            "MARGIN_RIGHT",
            "TEXT_OFFSET_X",
            "TEXT_OFFSET_Y",
            "Y_UPLIM_LEFT",
            "Y_STEP_LEFT",
            "Y_UPLIM_RIGTH",
            "Y_STEP_RIGTH",
            "MOP_normative",
            "e_SDR",
            "dn",
            "ro_PE",
            "price_PE",
            "l_pipe",
            "MRS",
            "c0",
            "c1",
            "k_2mm",
            "k_4mm",
            "xvalue_SDR",
            "xlabel_SDR",
            "e0",
            "e1",
        ),
    )
    # имена файлов и директорий
    Params.DATA_FILENAME = config["file_dir_names"]["data_filename"]
    Params.OUTPUT_FIG_FILENAME = config["file_dir_names"]["output_fig_filename"]
    Params.FIGURES_DIRNAME = config["file_dir_names"]["figures_dirname"]

    # настройки отображения графиков
    Params.BACKGROUND_COLOR_FIG = config["settings_for_figures"]["background_color_fig"]
    Params.PE_COLOR = config["settings_for_figures"]["PE_color"]
    Params.KOLCHUGA_2MM_COLOR = config["settings_for_figures"]["Kolchuga_2mm_color"]
    Params.KOLCHUGA_4MM_COLOR = config["settings_for_figures"]["Kolchuga_4mm_color"]
    Params.VERTIC_LINES_POINT_COLOR = config["settings_for_figures"][
        "vertic_lines_point_color"
    ]
    Params.BLACK_COLOR = config["settings_for_figures"]["black_color"]
    Params.MARGIN_LEFT = config["settings_for_figures"]["margin_left"]
    Params.MARGIN_RIGHT = config["settings_for_figures"]["margin_right"]
    Params.TEXT_OFFSET_X = config["settings_for_figures"]["text_offset_x"]
    Params.TEXT_OFFSET_Y = config["settings_for_figures"]["text_offset_y"]
    Params.Y_UPLIM_LEFT = config["settings_for_figures"]["y_uplim_left"]
    Params.Y_STEP_LEFT = config["settings_for_figures"]["y_step_left"]
    Params.Y_UPLIM_RIGTH = config["settings_for_figures"]["y_uplim_right"]
    Params.Y_STEP_RIGTH = config["settings_for_figures"]["y_step_right"]

    # входные данные для расчетов
    Params.MOP_normative = config["input_data_for_compute"][
        "MOP_normative"
    ]  # регламентированное значение МОР для SDR 11, МПа
    Params.e_SDR = config["input_data_for_compute"][
        "e_SDR"
    ]  # толщина стенки для SDR 11, м
    Params.dn = config["input_data_for_compute"]["dn"]  # наружный диаметр трубы, м
    Params.ro_PE = config["input_data_for_compute"][
        "ro_PE"
    ]  # плотность полиэтилена, кг/м3
    Params.price_PE = config["input_data_for_compute"][
        "price_PE"
    ]  # стоимость полиэтилена, руб/кг
    Params.l_pipe = config["input_data_for_compute"][
        "l_pipe"
    ]  # длина полиэтиленовой трубы, м
    Params.MRS = config["input_data_for_compute"][
        "MRS"
    ]  # минимальная длительная прочность ПЭ трубы, МПа
    Params.c0 = config["input_data_for_compute"][
        "c0"
    ]  # нижняя граница коэффициента запаса прочности
    Params.c1 = config["input_data_for_compute"][
        "c1"
    ]  # верхняя граница коэффициента запаса прочности
    Params.k_2mm = config["input_data_for_compute"][
        "k_2mm"
    ]  # коэффициент усилиния для ПЭ трубы с ЗСП Кольчуга (2 слоя)
    Params.k_4mm = config["input_data_for_compute"][
        "k_4mm"
    ]  # коэффициент усилиния для ПЭ трубы с ЗСП Кольчуга (4 слоя)
    Params.xvalue_SDR = config["input_data_for_compute"]["xvalue_SDR"]
    Params.xlabel_SDR = config["input_data_for_compute"]["xlabel_SDR"]

    Params.e0 = Params.xvalue_SDR[
        0
    ]  # левая граница диапазона изменения минимальной толщины стенки
    Params.e1 = Params.xvalue_SDR[
        -1
    ]  # правая граница диапазона изменеия минимальной толщины стенки

    logging.info(f"Подготовлен именованный кортеж из {params_count(config)} параметров")
    return Params


def plot_all(
    data: pd.DataFrame,
    params: namedtuple,
    output_fig_filepath: PathWinOrLinux,
) -> NoReturn:
    """
    Строит все графические элементы
    """
    fig, ax_left = plt.subplots(figsize=(11, 8))
    ax_right = ax_left.twinx()
    ax_top = ax_left.secondary_xaxis("top")
    ax_left.patch.set_facecolor(params.BACKGROUND_COLOR_FIG)

    plot_MOP_emin(ax_left)
    plot_cost_emin(ax_right, data)
    plot_added_graph_elements(ax_left)
    plot_top_axis(ax_top)
    plot_main_graph(ax_left)
    plot_right_axis(ax_right)

    logging.info(f"График будет сохранен по пути `{output_fig_filepath}`")
    plt.savefig(output_fig_filepath, dpi=350, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    PROJECT_DIR = Path().cwd()
    CONFIG_FILENAME = args.config_path

    config = read_config(CONFIG_FILENAME)
    params = init_param(config)

    data_filepath, output_fig_filepath = preprocessing_paths(params)
    data = read_data(data_filepath)
    plot_all(data, params, output_fig_filepath)
