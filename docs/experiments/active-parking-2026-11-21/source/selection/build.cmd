@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 exit /b 1
cl /nologo /EHsc /O2 /std:c++17 /Isrc /Isrc/nn /Fobuild\quiet-gait-v2\quiet.obj /Febuild\quiet-gait-v2\quiet.exe build\quiet-gait-v2\quiet.cpp build\front-pad-output-bias-v4\checkpoint.obj build\front-pad-output-bias-v4\world.obj
exit /b %errorlevel%
