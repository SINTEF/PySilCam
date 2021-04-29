set expected_miniconda_path=%UserProfile%\Miniconda3\Scripts\activate.bat
set silcam_env_path=%conda_path%..\..\..\envs\silcam
call %conda_path% %silcam_env_path%
pause