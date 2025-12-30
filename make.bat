@echo off
REM Automation script for common development tasks

if "%1"=="" goto help
if "%1"=="help" goto help
if "%1"=="install" goto install
if "%1"=="install-dev" goto install_dev
if "%1"=="train" goto train
if "%1"=="train-quick" goto train_quick
if "%1"=="evaluate" goto evaluate
if "%1"=="compare" goto compare
if "%1"=="tensorboard" goto tensorboard
if "%1"=="test" goto test
if "%1"=="test-cov" goto test_cov
if "%1"=="lint" goto lint
if "%1"=="format" goto format
if "%1"=="clean" goto clean
goto help

:help
echo Available commands:
echo   make install          - Install package in development mode
echo   make install-dev      - Install package with development dependencies
echo   make train            - Run training with default config
echo   make train-quick      - Run quick test training (10k timesteps)
echo   make evaluate         - Evaluate the most recent model
echo   make compare          - Compare most recent model against baselines
echo   make tensorboard      - Launch TensorBoard for most recent run
echo   make test             - Run tests with pytest
echo   make test-cov         - Run tests with coverage report
echo   make lint             - Run code linting with flake8
echo   make format           - Format code with black
echo   make clean            - Remove temporary files and caches
goto end

:install
echo Installing package in development mode...
pip install -e .
goto end

:install_dev
echo Installing package with development dependencies...
pip install -e ".[dev]"
goto end

:train
echo Training with default config...
.venv\Scripts\python.exe src\train.py --config configs\default.yaml
goto end

:train_quick
echo Training with quick test config...
.venv\Scripts\python.exe src\train.py --config configs\quick_test.yaml
goto end

:evaluate
echo Finding most recent run...
for /f "delims=" %%i in ('dir /b /ad /o-d outputs\runs 2^>nul ^| findstr /r "^20"') do (
    set LATEST_RUN=%%i
    goto evaluate_run
)
echo No runs found in outputs\runs
goto end

:evaluate_run
echo Evaluating model from %LATEST_RUN%...
.venv\Scripts\python.exe src\evaluate.py --model-path outputs\runs\%LATEST_RUN%\model.zip --episodes 20 --num-seeds 3
goto end

:compare
echo Finding most recent run...
for /f "delims=" %%i in ('dir /b /ad /o-d outputs\runs 2^>nul ^| findstr /r "^20"') do (
    set LATEST_RUN=%%i
    goto compare_run
)
echo No runs found in outputs\runs
goto end

:compare_run
echo Comparing model from %LATEST_RUN%...
.venv\Scripts\python.exe src\compare.py --model-path outputs\runs\%LATEST_RUN%\model.zip --episodes 20 --num-seeds 3
goto end

:tensorboard
echo Finding most recent run...
for /f "delims=" %%i in ('dir /b /ad /o-d outputs\runs 2^>nul ^| findstr /r "^20"') do (
    set LATEST_RUN=%%i
    goto tensorboard_run
)
echo No runs found in outputs\runs
goto end

:tensorboard_run
echo Launching TensorBoard for %LATEST_RUN%...
tensorboard --logdir outputs\runs\%LATEST_RUN%\tensorboard
goto end

:test
echo Running tests...
pytest tests/ -v
goto end

:test_cov
echo Running tests with coverage...
pytest tests/ -v --cov=src --cov-report=html --cov-report=term
echo Coverage report saved to htmlcov/index.html
goto end

:lint
echo Running flake8 linter...
flake8 src tests --count --statistics
goto end

:format
echo Formatting code with black...
black src tests
isort src tests
goto end

:clean
echo Cleaning temporary files...
if exist __pycache__ rmdir /s /q __pycache__
if exist src\__pycache__ rmdir /s /q src\__pycache__
if exist tests\__pycache__ rmdir /s /q tests\__pycache__
if exist .pytest_cache rmdir /s /q .pytest_cache
if exist .coverage del /q .coverage
if exist htmlcov rmdir /s /q htmlcov
if exist *.egg-info rmdir /s /q *.egg-info
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
echo Clean complete.
goto end

:end
