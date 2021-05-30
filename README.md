## Сносный пример python-сценария, использующего концепцию переиспользования кода

### Описание: один и тот же сценарий summary_PEpipe_Kolchuga.py используется для создания различных графиков (в рамках одного шаблона) и управления их внешним видом "на лету" (с помощью конфигурационных файлов \*.yaml), без внесения изменений в код сценария.

#### Структура проекта

```sh
project/
  -- python_scripts/
    -- summary_PEpipe_Kolchuga.py
  README.md
  -- configs/
    -- config_SDR9.yaml
    -- config_SDR11.yaml
    -- config_SDR17.yaml
```

#### Пример использования
```sh
# в корне проекта
$ python python_scripts\summary_PEpipe_Kolchuga.py /
    --config-path configs\config_SDR9.yaml
```
