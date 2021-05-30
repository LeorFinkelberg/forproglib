from typing import Union, NoReturn
import yaml
import sys
import argparse
from pathlib2 import Path, PosixPath, WindowsPath
from dataclasses import dataclass
import marshmallow
from marshmallow_dataclass import class_schema


@dataclass
class Colors:
    blue_purple: str
    fire_sienna: str
    blue: str
    
    
@dataclass
class FigureSettings:
    bg_color: str
    convas_color: str
    line_color: str
    

@dataclass
class Params:
    """
    """
    rnd_seed: int
    threshold: float
    colors: Colors
    figure_settings: FigureSettings
    

def print_err_and_exit(err: str) -> NoReturn:
    print(f"Ошибка: {err}")
    sys.exit()
    
    
ParamsSchema = class_schema(Params)

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
    schema = ParamsSchema()
    
    try:
        with open(Path(filepath), "r") as fo:
            return schema.load(
                yaml.safe_load(fo) # вернет обычный словарь Python
            )
    except ValueError as err:
        print_err_and_exit(err)
    except marshmallow.ValidationError as err:
        print_err_and_exit(err)
        
        
def cmd_line_parser() -> str:
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
    args = parser.parse_args()
    
    return args.config_path