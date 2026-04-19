#!/usr/bin/env pwsh
# Пример последовательного запуска нескольких конфигураций.
# Для запуска: откройте PowerShell в папке проекта и выполните:
# .\run_all_experiments.ps1
# При необходимости предварительно разрешите выполнение скриптов:
# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process

# Определяем корневую директорию проекта (папка, содержащая этот скрипт)
$scriptDir = $PSScriptRoot
if (-not $scriptDir) {
    # Если скрипт запущен из консоли без указания пути, берём текущую директорию
    $scriptDir = (Get-Location).Path
}
$rootDir = Split-Path -Parent $scriptDir

# Путь к конфигурационному файлу (можете добавить несколько)
$config1 = Join-Path $rootDir "experiments\configs\default.yaml"

# Запускаем эксперимент
Write-Host "Running experiment with config: $config1"
python (Join-Path $rootDir "experiments\run_grid.py") --config $config1

# Если нужно запустить несколько конфигураций, раскомментируйте:
# $config2 = Join-Path $rootDir "experiments\configs\another.yaml"
# python (Join-Path $rootDir "experiments\run_grid.py") --config $config2