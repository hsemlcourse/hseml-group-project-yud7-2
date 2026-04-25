[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — классификация сельскохозяйственных культур

**Студент:Юдинцев Владимир Алексеевич**

**Группа: БИВ232**


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуск](#запуск)
4. [Данные](#данные)
5. [Результаты](#результаты)
6. [Отчёт](#отчёт)


## Описание задачи

**Задача:** многоклассовая классификация сельскохозяйственной культуры поля по спутниковым временным рядам Sentinel-2.

**Датасет:** EuroCropsML, Zenodo: https://zenodo.org/records/15095445

**Целевая метрика:** macro F1. Дополнительно считаются accuracy и balanced accuracy.


## Структура репозитория
```
.
├── Dockerfile
├── docker-compose.yml
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda_portugal.ipynb   # EDA по Portugal subset
│   └── 02_baseline_portugal.ipynb # Baseline-модель
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   ├── baseline_metrics.csv    # Метрики baseline
│   └── report.md               # Финальный отчёт
├── src
│   ├── config.py               # Общие настройки и метаданные Zenodo
│   ├── data.py                 # Загрузка и подготовка preprocess-версии
│   ├── raw_portugal.py         # Извлечение и подготовка Portugal raw data
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   ├── test.py
│   └── test_preprocessing.py   # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
pip install -r requirements.txt
```

Посмотреть доступные файлы Zenodo и размеры:
```bash
python -m src.data info
```

Для CP1 используется только Португалия из `raw_data.zip`, потому что это самая компактная и понятная постановка: одна страна, свой train/validation/test split, baseline без feature engineering.

Скачать общий raw-архив с докачкой можно так:
```bash
curl -L -C - -o data/raw/raw_data.zip \
  "https://zenodo.org/api/records/15095445/files/raw_data.zip/content"
```

Размеры файлов в актуальной версии Zenodo v11:
- `split.zip` — 20.7 MB
- `preprocess.zip` — 1.47 GB
- `raw_data.zip` — 3.28 GB

На Zenodo нет отдельного архива только для Portugal, поэтому нужные файлы извлекаются из общего `raw_data.zip`:
```bash
python -m src.raw_portugal extract
python -m src.raw_portugal prepare
python -m src.modeling --dataset data/processed/portugal_baseline_dataset.npz \
  --model models/portugal_sgd_logistic_baseline.joblib
```

Docker-проверка окружения:
```bash
docker compose up --build
```

Основные ноутбуки:
- `notebooks/01_eda_portugal.ipynb` — описание датасета, пропуски, дубли, дисбаланс классов, региональные и временные графики.
- `notebooks/02_baseline_portugal.ipynb` — подготовка `date x band` признаков, train/validation/test split 60/20/20, baseline без feature engineering.

Если нужен готовый preprocess-пайплайн по `.npz`, можно скачать `split.zip` и `preprocess.zip`:
```bash
python -m src.data download --files split.zip preprocess.zip
python -m src.data prepare
python -m src.modeling
```

## Данные
- `data/raw/` — исходные файлы
- `data/processed/` — предобработанные данные

Исходный датасет EuroCropsML объединяет EuroCrops reference data с Sentinel-2 reflectance за 2021 год для Latvia, Portugal и Estonia. В CP1 берётся Portugal из raw stage:
- `Portugal.parquet` — временные ряды наблюдений по полям
- `Portugal_labels.parquet` — классы культур; в `Portugal.parquet` целевая переменная `EC_hcat_c` уже продублирована
- `Portugal.geojson` — геометрии полей, для baseline не используются

Сырые и обработанные данные не коммитятся, это настроено в `.gitignore`.


## Результаты
| Модель | Test macro F1 | Test accuracy | Примечание |
|--------|---------------|---------------|------------|
| Most frequent dummy | 0.007 | 0.296 | Test, наивная нижняя планка |
| SGD Logistic baseline | 0.126 | 0.288 | Test, baseline без feature engineering |


## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)
