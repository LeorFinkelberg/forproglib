## Сокращения по проекту
- КФ: _корреляционная функция_
- ПСП: _псевдослучайный процесс_
- ПСЧ: _псевдослучайное число_


### Пример вызова сценария с конфигурационным файлом
```sh
$ python python_scripts/main.py \
    --config-path configs/gauss_processes_acf.yaml \
    --output-fig-path figure/gauss_process_exp_acf.pdf
```

### Приемы работы с потоковым редактором sed
```sh
# изменяем значение параметра w0 на 3.15 и сохраняем 
# новый конфигурационный файл под именем gauss_exp_acf_w0=3.15.yaml
$ sed 's/w0: !!float 3.0/w0: !!float 3.15/' configs/gauss_processes_acf.yaml \
    > configs/gauss_exp_acf_w0=3.15.yaml
```
