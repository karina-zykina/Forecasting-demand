# Forecasting Demand

Модель прогнозирования спроса как B2B-продукт для клиентов компании.

Клиент передает `pandas.DataFrame`, указывает названия обязательных колонок:
- колонка с датой;
- колонка с целевой переменной спроса или продаж.

Также для повышения качества модели клиент может передать дополнительные признаки:
- цену;
- промо;
- канал продаж;
- регион;
- бренд;
- и любые другие факторы, которые могут влиять на спрос.


## Pipeline работы проекта

### 1. Пользователь передает данные в формате `pandas.DataFrame`

На практике это означает, что до обучения модели данные можно получить из базы данных, а затем преобразовать в `pandas.DataFrame`.

Пример загрузки данных из базы:

```python
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine("postgresql://...")

query = """
select *
from sales_table
"""

df = pd.read_sql(query, engine)
```

После этого `df` можно сразу подавать в модель.

Пример инициализации модели:

```python
model = DemandForecastModel(
    date_column="date",                  # название колонки с датой
    target_column="quantity",            # название колонки с целевой переменной
    feature_columns=["temperature", "promo_type", "price"],  # дополнительные признаки
    group_columns=["sku"],               # группировка независимых временных рядов
)
```

### Что такое `group_columns`

`group_columns` нужны, когда в одном датафрейме лежит не один временной ряд, а много разных.

Пример:
- `SKU_A` имеет одну историю продаж;
- `SKU_B` имеет другую историю продаж;
- `SKU_C` имеет третью историю продаж.

Если передать:

```python
group_columns=["sku"]
```

модель понимает:
- лаги для `SKU_A` нужно считать только по истории `SKU_A`;
- лаги для `SKU_B` только по истории `SKU_B`;
- rolling-статистики тоже отдельно по каждому SKU;
- прогноз строится отдельно для каждого SKU.

Если группировка сложнее, можно передавать несколько колонок:

```python
group_columns=["sku", "region"]
```

Тогда отдельным временным рядом считается каждая комбинация `sku + region`.

Иными словами, `group_columns` говорят модели, какие объекты нужно прогнозировать независимо друг от друга.
Таким образом, модель не смешивает историю разных SKU, регионов или других объектов прогноза.

### 2. Проект проверяет входные данные

На этапе `prepare_dataframe(...)` происходит следующее:

- проверяется наличие `date_column`;
- на обучении проверяется наличие `target_column`;
- `date` приводится к `datetime`;
- target приводится к числу;
- строки с невалидной датой или невалидным target удаляются.


##### Как агрегируются дубликаты

Если во входной таблице есть несколько строк на одну и ту же дату и одну и ту же группу:

- target суммируется;
- числовые дополнительные признаки усредняются;
- категориальные признаки берут первое непустое значение.


### 3. Проект строит признаки из временного ряда

Из истории автоматически создаются следующие признаки.

### Календарные признаки

- `day_of_week`
- `day_of_month`
- `month`
- `quarter`
- `week_of_year`
- `is_weekend`
- `day_of_year_sin`
- `day_of_year_cos`


### Лаговые признаки

- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

Это означает:
- сколько продали 1 день назад;
- 7 дней назад;
- 14 дней назад;
- 28 дней назад.

Лаги позволяют модели использовать прошлую историю спроса как главный сигнал для прогноза.

### Rolling-признаки

- `rolling_mean_7`
- `rolling_std_7`
- `rolling_mean_14`
- `rolling_std_14`

Это:
- среднее за окно;
- стандартное отклонение за окно.

Rolling-признаки помогают модели видеть локальный уровень и волатильность спроса.

##### Важно

Лаги и rolling-признаки можно настраивать через конфиг:
- добавлять;
- убирать;
- менять окна.

### 4. Проект обрабатывает внешние признаки пользователя

Все признаки из `feature_columns` добавляются в модель как дополнительные факторы.

Если признак числовой:
- он приводится к numeric;
- пропуски заполняются медианой по train.

Если признак категориальный:
- он приводится к строке;
- затем кодируется через one-hot.


##### Важная бизнес-оговорка

Технически проект умеет обработать любые переданные признаки, но бизнесово важно учитывать, известны ли эти признаки в будущем.

Например:
- `price_unit` и `promotion_flag` часто можно знать заранее;
- `stock_available` и `delivered_qty` могут быть недоступны на момент прогноза.

Поэтому для production-прогноза желательно использовать те признаки, которые:
- реально известны на момент расчета прогноза;
- либо могут быть заранее подготовлены для будущих дат.

### 5. Проект делит данные на train и validation

Разбиение делается по времени:

- ранняя часть ряда идет в train;
- хвост ряда идет в validation.

Случайное перемешивание не используется, потому что оно нарушает временную структуру.

### 6. Обучается ансамбль `LightGBM + CatBoost`

После подготовки матрицы признаков обучаются две базовые модели:

1. `LightGBM`
2. `CatBoost`

Затем их прогнозы усредняются:

```python
final_pred = (
    w_lightgbm * pred_lightgbm
    + w_catboost * pred_catboost
)
```

Идея ансамбля:
- `LightGBM` хорошо улавливает нелинейности;
- `CatBoost` хорошо работает на табличных данных и устойчиво обрабатывает сложные зависимости;
- их усреднение часто дает более устойчивый результат, чем одна отдельная модель.

### 7. Считаются метрики качества

После обучения проект считает:

- `MAE`
- `MAPE`
- `SMAPE`
- `R2`

Также дополнительно считается `quality_score`:

```python
quality_score = 1 - smape / 100
```

### Интерпретация метрик

- `MAE` — средняя абсолютная ошибка прогноза в тех же единицах, что и target.
  Например, если target — это количество проданных единиц товара, то `MAE = 5` означает, что модель в среднем ошибается на 5 единиц товара.

- `MAPE` — средняя абсолютная ошибка в процентах.
  Удобна для бизнес-интерпретации, но чувствительна к строкам, где фактический спрос близок к нулю.

- `SMAPE` — симметричная процентная ошибка.
  Более устойчива, чем `MAPE`, когда значения ряда маленькие или сильно отличаются по масштабу.

- `R2` — коэффициент детерминации.
  Показывает, какую долю вариации target объясняет модель. Чем ближе к 1, тем лучше.

- `quality_score` — упрощенная агрегированная оценка качества.
  Это удобная summary-метрика, но для анализа качества модели лучше смотреть на полный набор метрик, особенно `MAE` и `SMAPE`.


### 8. Сохраняется история для будущего прогноза

После `fit()` объект модели хранит:

- историю исходных данных после подготовки;
- схему кодирования признаков;
- параметры и веса моделей;
- дату последней наблюдаемой точки;
- метрики качества.

Это нужно для того, чтобы потом можно было:

- делать `predict()` на future-таблице;
- делать `forecast()` без future-таблицы;
- сохранять и загружать модель без повторного обучения.

## Структура проекта

```text
Forecasting-demand/
|-- demand_forecasting/
|   |-- __init__.py
|   |-- config.py
|   `-- model.py
|-- src/
|   |-- __init__.py
|   |-- runtime.py
|   |-- model/
|   |   |-- __init__.py
|   |   |-- config.py
|   |   |-- forecast_model.py
|   |   `-- preprocessing.py
|   `-- pipeline/
|       |-- __init__.py
|       |-- train.py
|       |-- evaluate.py
|       `-- predict.py
|-- pyproject.toml
|-- requirements.txt
`-- README.md
```

## Полный процесс предсказания

Есть два основных режима прогноза.

### Режим 1. Прогноз по `future_df`

Пользователь передает будущие даты и, при необходимости, будущие внешние признаки:

```python
future_df = pd.DataFrame(
    {
        "date": [...],
        "temperature": [...],
        "promo_type": [...],
    }
)

forecast = model.predict(future_df)
```

### Что такое `future_df`

`future_df` — это таблица будущих дат, для которых нужно построить прогноз.

Она может содержать:
- только дату и группирующие колонки, например `sku`;
- дату, группирующие колонки и будущие значения дополнительных признаков, если они известны заранее.

Пример:
- будущая цена;
- будущий promo flag;
- будущий канал продаж;
- будущий региональный сценарий.

Если `future_df` передан, пользователь сам задает стартовую дату прогноза через колонку даты.


### Режим 2. Прогноз только по горизонту

Если отдельного `future_df` нет:

```python
forecast = model.forecast(14)
```

Тогда проект:

- сам строит будущие даты;
- берет последние известные значения внешних признаков;
- выполняет рекурсивный прогноз шаг за шагом.

### Как задается стартовая дата прогноза

Если `future_df` не передан, стартовая дата определяется автоматически:

- берется последняя дата из исторических данных после `fit`;
- к ней добавляется 1 день;
- это и будет первая дата прогноза.



## Пример использования в ноутбуке для экспериментов


Если вы хотите протестировать проект локально или в ноутбуке, можно клонировать репозиторий из GitHub:

```bash
!git clone https://github.com/karina-zykina/Forecasting-demand.git
!cd Forecasting-demand
!pip install -r requirements.txt
```

После клонирования репозитория можно добавить его в `PYTHONPATH`

```python
import sys

sys.path.append("/path/to/Forecasting-demand")

from demand_forecasting import DemandForecastModel, ForecastModelConfig
```


Пример использования:

```python
import pandas as pd

from demand_forecasting import DemandForecastModel, ForecastModelConfig


df = pd.read_csv("my_sales.csv")

config = ForecastModelConfig(
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14, 28),
    default_horizon=14,
)

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type", "price"],
    group_columns=["region", "brand"],
    prediction_column="predicted_quantity",
    config=config,
)

tuning = model.tune_hyperparameters(df, max_trials=4)
print(tuning["best_metrics"])

print(model.quality_metrics_)
print(model.quality_score_)

future_df = pd.read_csv("future_features.csv")
forecast = model.predict(future_df)

horizon_forecast = model.forecast(14)

model.save("artifacts/my_model.zip")
```



## Пример использования через командную строку

### Как передаются данные в CLI

В Python API клиент сам передает `pandas.DataFrame`.

При использовании через командную строку клиент не загружает `DataFrame` вручную, а передает путь к CSV-файлу, который скрипт читает самостоятельно.


### Обучение

```powershell
python .\src\pipeline\train.py --data .\data\train.csv --model-path .\artifacts\model.zip --date-column date --target-column quantity --feature-columns temperature,promo_type,price --group-columns region,brand
```


### Оценка

```powershell
python .\src\pipeline\evaluate.py --model-path .\artifacts\model.zip --data .\data\test.csv --date-column date --target-column quantity
```


### Прогноз

Если есть future-файл:

```powershell
python .\src\pipeline\predict.py --model-path .\artifacts\model.zip --future-data .\data\future.csv --horizon 14 --output-path .\artifacts\forecast.csv
```

Если future-файла нет:

```powershell
python .\src\pipeline\predict.py --model-path .\artifacts\model.zip --horizon 14
```

## Python API

Импорт:

```python
from demand_forecasting import DemandForecastModel, ForecastModelConfig
```

Основные методы:

- `fit(df)` обучает модель;
- `tune_hyperparameters(df, max_trials=10)` подбирает гиперпараметры через Optuna и переобучает модель на лучших настройках;
- `predict(future_df)` строит прогноз по будущей таблице;
- `forecast(horizon)` строит прогноз только по горизонту;
- `predict_by_sku(sku_column, future_df)` возвращает прогноз в формате `SKU: количество`;
- `forecast_by_sku(horizon, sku_column)` строит прогноз на горизонт в формате `SKU: количество`;
- `save(path)` сохраняет модель;
- `load(path)` загружает сохраненную модель;
- `summary()` возвращает краткую сводку;
- `get_feature_importance()` возвращает важности признаков.

## Конфиг модели

Конфиг проекта хранится в отдельном файле проекта и описывается классом `ForecastModelConfig`.

Можно:
- явно передать свой конфиг;
- не передавать конфиг вообще, тогда будет использован дефолтный конфиг проекта.

Пример:

```python
config = ForecastModelConfig(
    lags=(1, 7, 14, 28),
    rolling_windows=(7, 14),
    default_horizon=7,
    min_history=30,
)
```
