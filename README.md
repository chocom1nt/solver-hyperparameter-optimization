# Оптимизация гиперпараметров решателя SCIP для задач MILP

[![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![OR-Tools](https://img.shields.io/badge/OR--Tools-9.8-green)](https://developers.google.com/optimization)
[![SCIP](https://img.shields.io/badge/SCIP-8.0-red)](https://scipopt.org)
[![Optuna](https://img.shields.io/badge/Optuna-3.5-orange)](https://optuna.org)
![GitHub repo size](https://img.shields.io/github/repo-size/chocom1nt/solver-hyperparameter-optimization)

Этот проект - исследовательская работа, посвящённая автоматическому подбору параметров решателя SCIP для задач смешанного целочисленного линейного программирования (MILP). В рамках работы разработан программный комплекс для проведения вычислительных экспериментов, сбора статистики и анализа влияния гиперпараметров на производительность решателя.

## 📋 О проекте

**Цель работы** - выявить оптимальные значения параметров SCIP для заданного класса задач MILP (например, планирование, упаковка, маршрутизация) и предложить рекомендации по настройке решателя.

**Основные задачи:**
- Сформировать тестовую выборку задач из коллекции MIPLIB.
- Реализовать обёртку для запуска SCIP с произвольными параметрами через OR-Tools.
- Провести серию экспериментов с различными комбинациями параметров.
- Проанализировать полученные данные с помощью профилей производительности и визуализации.
- Применить методы оптимизации гиперпараметров (Optuna) для автоматического поиска лучших настроек.

**Ключевые особенности:**
- Модульная архитектура, позволяющая легко добавлять новые решатели и метрики.
- Поддержка формата `.mps` (в том числе сжатых `.mps.gz`).
- Конфигурация экспериментов через YAML-файлы.
- Воспроизводимость результатов с помощью Docker.

## 🏗️ Архитектура и технологии

*   **Язык программирования:** Python 3.10+
*   **Решатели:** SCIP (через OR-Tools), HiGHS, (опционально Gurobi)
*   **Оптимизация гиперпараметров:** Optuna
*   **Анализ данных:** Pandas, NumPy, Matplotlib, Seaborn
*   **Управление экспериментами:** собственный модуль `experiment_runner`
*   **Формат хранения результатов:** CSV / SQLite
*   **Контейнеризация:** Docker, Docker Compose

**Структура репозитория:**
```
coursework/
├── config/                # YAML-конфигурации экспериментов
├── data/                  # Данные: сырые .mps.gz, метаданные, результаты
├── docs/                  # Документация (постановка задачи, протоколы)
├── experiments/           # Скрипты для запуска экспериментов
├── notebooks/             # Jupyter-ноутбуки для анализа и визуализации
├── src/                   # Исходный код
│   ├── solvers/           # Обёртки для решателей (SCIP, HiGHS)
│   ├── experiment/        # Запуск экспериментов, работа с конфигами
│   ├── data/              # Загрузка задач, сбор метаданных
│   ├── analysis/          # Расчёт метрик, построение графиков
│   └── hpo/               # Оптимизация гиперпараметров (Optuna)
├── scripts/               # Вспомогательные скрипты (сбор метаданных и т.д.)
├── tests/                 # Модульные тесты (опционально)
├── .gitignore
├── README.md
├── requirements.txt
└── docker-compose.yml
```

## ⚙️ Основные компоненты

### 1. Загрузка и подготовка данных (`src/data/`)
*   **metadata_extractor.py** - извлекает характеристики задачи (число переменных, ограничений, тип) из `.mps`-файла.
*   **task_loader.py** - сканирует директорию и возвращает список задач, отфильтрованных по метаданным.

### 2. Обёртки для решателей (`src/solvers/`)
*   Базовый класс `BaseSolver` с методом `solve(problem_path, time_limit) -> dict`.
*   Реализации для SCIP и HiGHS (через OR-Tools). Параметры передаются в виде словаря и преобразуются в строку для `SetSolverSpecificParametersAsString`.

### 3. Управление экспериментами (`src/experiment/`)
*   **config.py** - загрузка и валидация YAML-конфига.
*   **runner.py** - перебор комбинаций (задача, решатель, параметры), запуск решения, сбор результатов, сохранение.
*   **result_writer.py** - запись в CSV или SQLite.

### 4. Анализ результатов (`src/analysis/`, `notebooks/`)
*   **performance_profiles.py** - реализация профилей производительности (Долан-Морэ).
*   **plotting.py** - функции для построения boxplot, scatter plots, heatmap.
*   Jupyter-ноутбуки для интерактивного анализа.

### 5. Оптимизация гиперпараметров (`src/hpo/`)
*   **search_space.py** - определение пространства поиска для Optuna.
*   **optimizer.py** - запуск оптимизации с целевой функцией (например, среднее время решения на валидационной выборке).

## 🧪 Пример конфигурации эксперимента

```yaml
# config/exp1.yaml
name: "SCIP parameter tuning"
description: "Grid search over branching rules and time limits"
tasks:
  source: "data/raw"
  filter:
    min_vars: 100
    max_vars: 5000
solvers:
    name: "SCIP"
    class: "ScipSolver"
    parameters:
      limits/time: [10, 30, 60]
      branching/rule: ["mostinf", "leastinf", "pseudocost"]
    repetitions: 1
output:
  format: "csv"
  path: "experiments/results/exp1.csv"
```

## 🚀 Запуск проекта

### 1. Клонирование и установка зависимостей
```bash
git clone https://github.com/yourusername/solver-hyperparameter-optimization.git
cd solver-hyperparameter-optimization
pip install -r requirements.txt
```

### 2. Подготовка данных
Поместите `.mps` или `.mps.gz` файлы в папку `data/raw/`.  
Для сбора метаданных выполните:
```bash
python scripts/collect_metadata.py
```

### 3. Запуск эксперимента
```bash
python -m experiments.run_experiment --config config/exp1.yaml
```

### 4. Анализ в Jupyter
```bash
jupyter notebook notebooks/01_analyze_results.ipynb
```

### 5. Запуск с Docker
```bash
docker-compose up
```

## 📊 Ожидаемые результаты

- Набор данных (CSV) с результатами экспериментов для различных комбинаций параметров.
- Графики профилей производительности, сравнивающие решатели и настройки.
- Оптимальные значения параметров SCIP для выбранного класса задач, найденные с помощью Optuna.
- Рекомендации по настройке решателя (в тексте работы).

## 🔮 Возможные улучшения

*   Добавление поддержки других решателей (Gurobi, COPT) через их Python API.
*   Распараллеливание экспериментов на несколько ядер/машин.
*   Интеграция с системами отслеживания экспериментов (MLflow, Weights & Biases).
*   Автоматическая генерация отчётов в LaTeX.
*   Расширение набора метрик (gap, количество узлов в дереве Branch-and-Bound).

## 📚 Полезные ссылки

- [Документация OR-Tools](https://developers.google.com/optimization)
- [SCIP Optimization Suite](https://scipopt.org)
- [MIPLIB 2017](https://miplib.zib.de)

