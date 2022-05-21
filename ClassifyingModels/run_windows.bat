@echo off
cmd /k "echo creating env... & py -m venv env & echo activating env... & (cd .\env\Scripts || cd .\env\bin) & activate & cd .. & cd .. & echo installing requirements... & py -m pip install -r requirements.txt"
PAUSE
