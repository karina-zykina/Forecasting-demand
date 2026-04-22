# Forecasting Demand

Проект для прогнозирования спроса на базе ансамбля `LightGBM + XGBoost`.

Главная идея: пользователь передает `pandas.DataFrame`, указывает, где лежат дата, целевая колонка и дополнительные признаки, а проект сам:

- подготавливает данные;
- агрегирует дубли;
- строит временные признаки;
- кодирует категориальные поля;
- обучает ансамбль;
- считает метрики качества;
- сохраняет модель;
- строит будущий прогноз.

## Что использует проект

Ансамбль состоит из трех моделей:

1. `LightGBM`
2. `XGBoost`

Итоговый прогноз считается как взвешенное среднее трех предсказаний.

По умолчанию веса такие:

```python
{
    "lightgbm": 0.50,
    "xgboost": 0.50,
}
```

Почему именно такой стек:

- `LightGBM` хорошо улавливает нелинейности и взаимодействия признаков;
- `XGBoost` часто устойчиво работает на табличных данных;
- вместе они обычно надежнее одной отдельной модели.

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
|-- examples/
|   `-- client_usage.py
|-- tests/
|   `-- test_demand_forecasting.py
|-- pyproject.toml
`-- README.md
```

## Как файлы взаимодействуют между собой

`demand_forecasting/__init__.py`

- публичная точка входа;
- отсюда пользователь импортирует `DemandForecastModel` и `ForecastModelConfig`.

`demand_forecasting/config.py`

- тонкий re-export конфига наружу.

`demand_forecasting/model.py`

- тонкий re-export публичной модели наружу.

`src/runtime.py`

- добавляет локальные зависимости в `sys.path`, если проект работает через локальный vendor-runtime.

`src/model/config.py`

- хранит все основные настройки обучения;
- задает лаги, rolling-окна, минимальную длину истории, параметры `LightGBM`, `XGBoost`, веса ансамбля.

`src/model/preprocessing.py`

- подготавливает входной `DataFrame`;
- проверяет обязательные колонки;
- приводит типы;
- агрегирует дубли;
- строит training frame;
- кодирует признаки в числовую матрицу.

`src/model/forecast_model.py`

- основной класс проекта;
- управляет полным циклом `fit -> predict -> save/load`;
- обучает ансамбль;
- считает метрики;
- делает рекурсивный прогноз.

`src/pipeline/train.py`

- CLI-скрипт для обучения модели из CSV.

`src/pipeline/evaluate.py`

- CLI-скрипт для оценки сохраненной модели на отложенной выборке.

`src/pipeline/predict.py`

- CLI-скрипт для построения прогноза из сохраненной модели.

`examples/client_usage.py`

- живой пример использования API из Python.

`tests/test_demand_forecasting.py`

- smoke-тест основного сценария использования.

## Какие данные принимает проект

Минимально нужны две колонки:

- колонка с датой;
- колонка с целевой переменной спроса или продаж.

Пример минимального датасета:

```python
import pandas as pd

df = pd.DataFrame(
    {
        "date": pd.date_range("2025-01-01", periods=120, freq="D"),
        "quantity": [10, 12, 15, 13, 17, 19, 18] * 17 + [20],
    }
)
```

Можно передавать и дополнительные признаки:

- `temperature`
- `promo_type`
- `price`
- `region`
- `brand`
- `store_type`
- любые другие внешние факторы

Пример расширенного датасета:

```python
df = pd.DataFrame(
    {
        "date": [...],
        "quantity": [...],
        "temperature": [...],
        "promo_type": [...],
        "price": [...],
        "region": [...],
        "brand": [...],
    }
)
```

## Полный процесс обучения

Ниже весь pipeline подробно, по шагам.

### 1. Пользователь передает `pandas.DataFrame`

Обычно это выглядит так:

```python
import pandas as pd

from demand_forecasting import DemandForecastModel


df = pd.read_csv("my_sales.csv")

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type", "price"],
)

model.fit(df)
```

### 2. Проект проверяет входные данные

На этапе `prepare_dataframe(...)` происходит следующее:

- проверяется наличие `date_column`;
- на обучении проверяется наличие `target_column`;
- если часть `feature_columns` или `group_columns` отсутствует, они создаются пустыми;
- дата приводится к `datetime`;
- target приводится к числу;
- строки с невалидной датой или невалидным target удаляются.

### 3. Проект агрегирует дубликаты

Если во входной таблице есть несколько строк на одну и ту же дату и одну и ту же группу:

- target суммируется;
- числовые дополнительные признаки усредняются;
- категориальные признаки берут первое непустое значение.

Это полезно, если исходный датасет еще не приведен к финальному grain-level.

### 4. Проект строит признаки из временного ряда

Из истории создаются:

- календарные признаки:
- `day_of_week`
- `day_of_month`
- `month`
- `quarter`
- `week_of_year`
- `is_weekend`
- `day_of_year_sin`
- `day_of_year_cos`

- лаговые признаки:
- `lag_1`
- `lag_7`
- `lag_14`
- `lag_28`

- rolling-признаки:
- `rolling_mean_7`
- `rolling_std_7`
- `rolling_mean_14`
- `rolling_std_14`
- и так далее, в зависимости от конфига

Очень важно: для каждой строки используются только прошлые данные. Будущее в обучение не подмешивается.

### 5. Проект обрабатывает внешние признаки пользователя

Все признаки из `feature_columns` добавляются в feature-row.

Если признак числовой:

- он приводится к numeric;
- пропуски заполняются медианой по train.

Если признак категориальный:

- он приводится к строке;
- затем кодируется через one-hot.

Если на прогнозе:

- какой-то столбец отсутствует, он будет создан пустым;
- встретилась новая категория, код не упадет, просто будет использована train-схема dummy-колонок.

### 6. Проект делит данные на train и validation

Разбиение делается по времени:

- ранняя часть идет в train;
- хвост ряда идет в validation.

Это сделано специально для временных рядов.

Случайное перемешивание тут не используется, потому что оно нарушает временную структуру и дает слишком оптимистичную оценку качества.

Если данных мало, проект может отказаться от отдельной validation и посчитать метрики на train как fallback.

### 7. Обучается ансамбль `LightGBM + XGBoost`

После подготовки матрицы признаков обучаются три базовые модели:

1. `LightGBM`
2. `XGBoost`

Затем их прогнозы усредняются:

```python
final_pred = (
    w_lightgbm * pred_lightgbm
    + w_xgboost * pred_xgboost
)
```

### 8. Считаются метрики качества

После обучения проект считает:

- `MAE`
- `MAPE`
- `SMAPE`
- `R2`

Дополнительно считается `quality_score`:

```python
quality_score = 1 - smape / 100
```

Это внутренняя удобная метрика в диапазоне примерно от `0` до `1`.

### 9. Сохраняется история для будущего прогноза

После `fit()` объект хранит:

- историю исходных данных после подготовки;
- схему кодирования признаков;
- параметры и веса моделей;
- дату последней наблюдаемой точки;
- метрики качества.

Это нужно для того, чтобы потом можно было:

- делать `predict()` на future-таблице;
- делать `forecast(horizon)` без future-таблицы;
- сохранять и загружать модель без повторного обучения.

## Полный процесс предсказания

Есть два основных режима.

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

Что делает проект:

- подготавливает `future_df` так же, как train-данные;
- для каждой будущей строки строит lag/rolling/calendar-признаки;
- кодирует признаки по схеме, выученной на train;
- получает прогноз от `LightGBM` и `XGBoost`;
- усредняет их;
- возвращает `DataFrame` с колонкой прогноза.

### Режим 2. Прогноз только по горизонту

Если отдельного `future_df` нет:

```python
forecast = model.forecast(14)
```

Тогда проект:

- сам строит будущие даты;
- берет последние известные значения внешних признаков;
- выполняет рекурсивный прогноз шаг за шагом.

## Что такое рекурсивный прогноз

Если нужно предсказать несколько будущих точек подряд, модель работает так:

1. предсказывает `t+1`;
2. добавляет этот прогноз в историю;
3. строит признаки для `t+2` уже с учетом прогноза на `t+1`;
4. предсказывает `t+2`;
5. повторяет процесс дальше.

Это нужно потому, что лаговые признаки для будущих дат зависят от предыдущих предсказаний.

## Как использовать проект для своего `pandas`-датасета

Ниже самый прямой пользовательский сценарий.

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

model.fit(df)

print(model.quality_metrics_)
print(model.quality_score_)

future_df = pd.read_csv("future_features.csv")
forecast = model.predict(future_df)

horizon_forecast = model.forecast(14)

model.save("artifacts/my_model.zip")
```

## Как подготовить свой датасет из pandas

Практически всегда достаточно следующего:

1. Загрузить данные в `DataFrame`.
2. Убедиться, что есть колонка даты.
3. Убедиться, что есть числовой target.
4. Передать список внешних признаков через `feature_columns`.
5. Если у вас несколько независимых рядов, передать `group_columns`.
6. Вызвать `fit`.
7. Для будущего прогноза передать `future_df` или использовать `forecast(horizon)`.

### Пример для одного ряда

```python
import pandas as pd

from demand_forecasting import DemandForecastModel


df = pd.read_csv("sales.csv")

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type", "price"],
)

model.fit(df)
forecast = model.forecast(14)
print(forecast.head())
```

### Пример для нескольких рядов

Если у вас данные по нескольким комбинациям `region + brand`, то можно обучать один объект модели на всем датасете:

```python
model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type", "price"],
    group_columns=["region", "brand"],
)

model.fit(df)
forecast = model.forecast(14)
```

В этом случае проект:

- различает ряды по `group_columns`;
- строит лаги только внутри каждой группы;
- при `forecast(horizon)` строит будущие даты отдельно для каждой группы.

## Как попробовать проект на своем датасете из pandas: пошагово

Если у вас уже есть `DataFrame`, можно начать так:

```python
import pandas as pd

from demand_forecasting import DemandForecastModel


df = pd.read_csv("internet_dataset.csv")

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type"],
)

model.fit(df)
print(model.summary())
```

Если вы хотите проверить модель на отложенной выборке вручную:

```python
import pandas as pd

from demand_forecasting import DemandForecastModel


df = pd.read_csv("internet_dataset.csv")
df["date"] = pd.to_datetime(df["date"])

train_df = df.iloc[:-30].copy()
test_df = df.iloc[-30:].copy()

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type"],
)

model.fit(train_df)

forecast = model.predict(test_df.drop(columns=["quantity"]))

y_true = test_df["quantity"].to_numpy()
y_pred = forecast["predicted_quantity"].to_numpy()

print(forecast.head())
print(model.quality_metrics_)
```

## Какой формат данных лучше всего подходит

Лучше всего подходят датасеты, где:

- есть явная колонка даты;
- есть числовой target спроса, продаж, заказов или отгрузок;
- есть регулярная временная частота;
- есть внешние признаки, которые известны в будущем или хотя бы частично предсказуемы.

Хорошие примеры внешних признаков:

- промо;
- цена;
- температура;
- день зарплаты;
- регион;
- магазин;
- бренд;
- категория;
- праздники;
- маркетинговая активность.

## Пример минимального живого сценария

```python
import pandas as pd

from demand_forecasting import DemandForecastModel


df = pd.DataFrame(
    {
        "date": pd.date_range("2025-01-01", periods=150, freq="D"),
        "quantity": [10, 11, 12, 13, 15, 18, 16] * 21 + [20, 21, 22],
    }
)

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
)

model.fit(df)
forecast = model.forecast(14)
print(forecast.head())
```

## Командная строка

### Обучение

```powershell
python .\src\pipeline\train.py --data .\data\train.csv --model-path .\artifacts\model.zip --date-column date --target-column quantity --feature-columns temperature,promo_type,price --group-columns region,brand
```

Что делает скрипт:

- читает CSV;
- создает объект модели;
- обучает ансамбль `LightGBM + XGBoost`;
- сохраняет zip-артефакт.

### Оценка

```powershell
python .\src\pipeline\evaluate.py --model-path .\artifacts\model.zip --data .\data\test.csv --date-column date --target-column quantity
```

Что делает скрипт:

- загружает сохраненную модель;
- строит прогноз на тестовом CSV;
- сравнивает прогноз с фактом;
- печатает метрики.

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
- `predict(future_df)` строит прогноз по будущей таблице;
- `forecast(horizon)` строит прогноз только по горизонту;
- `save(path)` сохраняет модель;
- `load(path)` загружает сохраненную модель;
- `summary()` возвращает краткую сводку;
- `get_feature_importance()` возвращает важности признаков.

## Пример полного цикла

```python
import pandas as pd

from demand_forecasting import DemandForecastModel


train_df = pd.read_csv("train.csv")
future_df = pd.read_csv("future.csv")

model = DemandForecastModel(
    date_column="date",
    target_column="quantity",
    feature_columns=["temperature", "promo_type", "price"],
)

model.fit(train_df)

print(model.quality_metrics_)
print(model.get_feature_importance())

forecast = model.predict(future_df)
model.save("artifacts/model.zip")

restored = DemandForecastModel.load("artifacts/model.zip")
same_forecast = restored.predict(future_df)
```

## Локальный пример

```powershell
python .\examples\client_usage.py
```

## Тесты

```powershell
python -m unittest .\tests\test_demand_forecasting.py
```

## Коротко

Весь проект работает так:

1. Получаем `DataFrame`.
2. Проверяем схему колонок.
3. Чистим и агрегируем данные.
4. Строим лаговые, rolling и календарные признаки.
5. Кодируем внешние признаки.
6. Обучаем `LightGBM + XGBoost`.
7. Усредняем их прогнозы.
8. Считаем метрики.
9. Сохраняем историю и модель.
10. Строим будущий прогноз рекурсивно.
