setlocal ENABLEDELAYEDEXPANSION
echo off

REM +++++++++++++++++++++++++++++++++++++++
REM Run Polaris model for DTA convergence
REM +++++++++++++++++++++++++++++++++++++++

REM -- SET THE MAIN AND DTA RUN SCENARIO FILES
set python_path=C:\Program Files\Python36\
set sqlite3_path=%CD%\third_party\Sqlite3\
set tail_app=%CD%\third_party\baretail.exe
set cloud_backup_path=\\vms-fs2\VMS_FY19_SMART_Runs\Chicago_ABM_Convergence2018\Chicago_202000406_results\

set SCENARIO_MAIN_INIT=scenario_abm_init_10per.json
set SCENARIO_MAIN=scenario_abm.json

REM -- ENTER THE NUMBER OF MAIN RUNS AND DTA RUNS PER MAIN RUN
set NUM_ABM_RUNS=12

REM -- ENTER THE OUTPUT DIRECTORIES AS SPECIFIED IN THE SCENARIO FILES
set OUT_DIR_MAIN=linux_vs17_chicago2018_abm

REM -- ENTER THE POLARIS RUN INFORMATION
set THREADS=20
set DB=chicago2018
set POLARIS_EXE=bin_new\Integrated_Model.exe

REM -------------------------------------------------------------------------------------------
REM Do not modify below here
REM -------------------------------------------------------------------------------------------
set WORKDIR=%CD%

del artificial_count.csv
del gap_calculations.csv
del gap_breakdown.csv

REM store the original inputs		
xcopy "%WORKDIR%\backup\%DB%-Demand.sqlite" %WORKDIR% /y /i
xcopy "%WORKDIR%\backup\highway_skim_file.bin" %WORKDIR% /y /i
echo "Polaris output goes here" > "%WORKDIR%\polaris.out"

REM Process main ABM run
set i=-1
for /L %%S in (1,1,%NUM_ABM_RUNS%) do (
	set /a i+=1
	set out_local=%OUT_DIR_MAIN%%%S
	if [%%S]==[1] (
		set out_local=%OUT_DIR_MAIN%
	)
	
	REM run polaris
	if [%%S]==[1] (
		echo Running Polaris SCENARIO_MAIN_INIT instance %%S - see "polaris.out"
		START %tail_app% "%WORKDIR%\polaris.out"
		%POLARIS_EXE% %SCENARIO_MAIN_INIT% %THREADS% > "%WORKDIR%\polaris.out" 2>&1
		xcopy "%WORKDIR%\polaris.out" "%WORKDIR%\!out_local!\" /y /i
	) else (
		REM call python script to modify the routing weight parameter
	 	call "%python_path%pythonw.exe" modify_scenario.py %SCENARIO_MAIN% time_dependent_routing_weight_factor 1.0
		echo Running Polaris SCENARIO_MAIN instance %%S - see "polaris.out"
		START %tail_app% "%WORKDIR%\polaris.out"
		%POLARIS_EXE% %SCENARIO_MAIN% %THREADS% > "%WORKDIR%\polaris.out" 2>&1
		xcopy "%WORKDIR%\polaris.out" "%WORKDIR%\!out_local!\" /y /i
	)
	
	REM copy local results back to the main run directory for the next run
	xcopy "%WORKDIR%\!out_local!\%DB%-Demand.sqlite" %WORKDIR% /y /i
	xcopy "%WORKDIR%\!out_local!\%DB%-Result.sqlite" %WORKDIR% /y /i
	xcopy "%WORKDIR%\!out_local!\highway_skim_file.bin" %WORKDIR% /y /i	
	
	%sqlite3_path%sqlite3.exe "%WORKDIR%\%DB%-Demand.sqlite" < clean_db_after_abm_for_abm.sql		
	
	ren %WORKDIR%\!out_local!\summary.csv summary_abm_%%S.csv	
	%sqlite3_path%sqlite3.exe "%WORKDIR%\!out_local!\%DB%-Demand.sqlite" < wtf_baseline_analysis_25Per_calibrate.sql	
	%sqlite3_path%sqlite3.exe "%WORKDIR%\!out_local!\%DB%-Demand.sqlite" < gap_calculate.sql	
	%sqlite3_path%sqlite3.exe "%WORKDIR%\!out_local!\%DB%-Demand.sqlite" < output_to_csv.sql
		
	type artificial_count_temp.csv >> artificial_count.csv
	type gap_calculations_temp.csv >> gap_calculations.csv
	type gap_breakdown_temp.csv >> gap_breakdown.csv
	
	del artificial_count_temp.csv
	del gap_calculations_temp.csv
	del gap_breakdown_temp.csv
	
	xcopy "artificial_count.csv" "%cloud_backup_path%" /y /i
	xcopy "gap_calculations.csv" "%cloud_backup_path%" /y /i
	xcopy "gap_breakdown.csv" "%cloud_backup_path%" /y /i
	xcopy "%WORKDIR%\!out_local!\summary_abm_%%S.csv" "%cloud_backup_path%" /y /i
	xcopy "%WORKDIR%\!out_local!\%DB%-Demand.sqlite" "%cloud_backup_path%\!out_local!\" /y /i
	xcopy "%WORKDIR%\!out_local!\%DB%-Result.sqlite" "%cloud_backup_path%\!out_local!\" /y /i
	xcopy "%WORKDIR%\!out_local!\highway_skim_file.bin" "%cloud_backup_path%\!out_local!\" /y /i
)

