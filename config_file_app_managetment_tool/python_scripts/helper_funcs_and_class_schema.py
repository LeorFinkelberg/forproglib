from typing import Tuple, NoReturn
import yaml
import sys
import argparse
from pathlib2 import Path, PosixPath, WindowsPath
from dataclasses import dataclass, field
import marshmallow
from marshmallow_dataclass import class_schema
import matplotlib.pyplot as plt
    

valid_min = lambda value: marshmallow.validate.Range(min=value)

@dataclass
class Colors:
    """
    Вспомогательный параметрический
    класс цветовых решений
    """
    blue_purple: str
    fire_sienna: str
    # blue: str
    
    
@dataclass
class FigureSettings:
    """
    Вспомогательный параметрический
    класс настроек изображения
    """
    bg_color: str
    # convas_color: str
    # line_color: str


@dataclass
class Params:
    """
    Главный параметрический класс
    """
    colors: Colors
    figure_settings: FigureSettings 
    sigma: float = field(
        metadata = {"validate" : valid_min(0.0)},
        default = 2
    )
    w_star: float = field(
        metadata = {"validate" : valid_min(1.0)},
        default = 1.25
    )
    w0: float = field(
        metadata = {"validate" : valid_min(1.0)},
        default = 3.0
    )
    delta_t: float = field(
        metadata = {"validate" : valid_min(0.01)},
        default = 0.05
    )
    N: int = field(
        metadata = {"validate" : valid_min(10)},
        default = 1000
    )
    kind_acf: int = field(default = 1)
    
ParamsSchema = class_schema(Params)


def print_err_and_exit(err: str) -> NoReturn:
    print(f"Ошибка: {err}")
    sys.exit()


def read_yaml_file(filepath: str) -> Params:
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
    schema = ParamsSchema() # экземпляр схемы
    
    abspath_to_config_file = Path(filepath).absolute()
    
    try:
        with open(abspath_to_config_file, "r") as fo:
            return schema.load(
                yaml.safe_load(fo) # вернет обычный словарь Python
            )
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
    переданный через флаг --config-path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    parser.add_argument("--output-fig-path", type=str)
    args = parser.parse_args()
    
    return args.config_path, args.output_fig_path


def draw_graph():
    fig, ax = ...
    
    fig.savefig(filepath, dpi=350, bbox_inches='tight')