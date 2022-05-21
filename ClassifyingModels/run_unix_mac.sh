#!/bin/sh
echo creating env... & py -m venv env & echo activating env... & cd (cd ./env/Scripts/ ||  cd ./env/bin/) & source activate & cd .. & cd .. & echo installing requirements... & py -m pip install -r requirements.txt
$SHELL
