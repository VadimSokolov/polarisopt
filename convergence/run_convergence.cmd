@echo off

REM run_convergence.cmd

IF [%1]==[] goto usage

python run_convergence.py convergence_control.json %1

goto end

:usage
@echo Usage: %0 ^<data_directory^>
exit /B 1

:end