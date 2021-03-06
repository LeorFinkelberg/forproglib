import argparse
import logging
import sys
from dataclasses import dataclass, field
from typing import NoReturn, Tuple

import marshmallow
import yaml
from marshmallow_dataclass import class_schema
from pathlib2 import Path

_log_format = f"[%(asctime)s | %(levelname)s]: %(message)s"


def make_file_handler() -> logging.FileHandler:
    """
    Настраивает файловый хендлер
    """
    file_handler = logging.FileHandler("app_logs.log")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(_log_format))
    return file_handler


def make_stream_handler() -> logging.FileHandler:
    """
    Настравивает потоковый хендлер
    """
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(_log_format))
    return stream_handler


def make_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.addHandler(make_file_handler())
    logger.addHandler(make_stream_handler())
    return logger


logger = make_logger(__name__)

valid_min = lambda min_value: marshmallow.validate.Range(min=min_value)
valid_min_max = lambda min_value, max_value: marshmallow.validate.Range(
    min=min_value, max=max_value
)


@dataclass
class Colors:
    """
    Вспомогательный параметрический
    класс цветовых решений
    """

    outliers_red: str
    blue_purple: str
    terakotta: str
    pearl_night: str
    krayola_green: str
    grey: str
    white: str


@dataclass
class FigureSettings:
    """
    Вспомогательный параметрический
    класс настроек изображения
    """

    left_xlim_acf: float
    right_xlim_acf: float
    height_main_fig: int
    width_main_fig: int
    left_xlim_pdf: float
    right_xlim_pdf: float


@dataclass
class Visibility:
    """
    Вспомогательный параметрический
    класс для управления видимостью
    элементов графика
    """

    ma_show: bool


@dataclass
class Params:
    """
    Главный параметрический класс
    """

    colors: Colors
    figure_settings: FigureSettings
    visibility: Visibility
    sigma: float = field(metadata={"validate": valid_min(0.0)}, default=2)
    w_star: float = field(metadata={"validate": valid_min(1.0)}, default=1.25)
    w0: float = field(metadata={"validate": valid_min(1.0)}, default=3.0)
    alpha: float = field(metadata={"validate": valid_min(0.01)}, default=0.15)
    window_width: int = field(metadata={"validate": valid_min(3)}, default=10)
    delta_t: float = field(
        metadata={"validate": valid_min(0.01)}, default=0.05
    )
    N: int = field(metadata={"validate": valid_min(10)}, default=1000)
    kind_acf: int = field(
        metadata={"validate": valid_min_max(1, 4)}, default=1
    )


ParamsSchema = class_schema(Params)


def print_err_and_exit(err: str) -> NoReturn:
    logger.error(f"Ошибка: {err}")
    sys.exit()


def read_yaml_file(config_path: str) -> Params:
    """
    Описание
    --------
    Принимает путь до конфигурационного файла
    в виде строки. В качестве разделителя узлов пути может
    использоваться как символ "\", так и символ "/"

    Возвращает
    ----------
    Объект, к полям которого можно
    обращаться с помощью точечной нотации
    """
    schema = ParamsSchema()  # экземпляр схемы

    abspath_to_config_file = Path(config_path).absolute()

    try:
        with open(abspath_to_config_file) as fo:
            loaded_schema = schema.load(
                yaml.safe_load(fo)
            )  # вернет обычный словарь Python
            logger.info(
                f"Конфигурационный файл {config_path} успешно прочитан."
            )
            return loaded_schema
    except ValueError as err:
        print_err_and_exit(err)
    except marshmallow.ValidationError as err:
        print_err_and_exit(err)


def cmd_line_parser() -> Tuple[str, str]:
    """
    Описание
    --------
    Разбирает командную строку

    Возвращает
    ----------
    Путь до конфигурационного файла,
    переданный через флаг --config-path и
    путь до файла-сводки
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--output-fig-path", type=str)
    args = parser.parse_args()

    return args.config_path, args.output_fig_path
