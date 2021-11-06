#!/bin/bash

cd ..
python src/organizefss1000.py $1
python src/verifyfss1000.py
cd -
