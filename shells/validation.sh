#! /usr/bin/env bash

ssh crest-super-01 "cd Codes/ssai; python scripts/batch_evaluation.py"
python scripts/collect_results.py
python scripts/analize_results.py
python scripts/comparing_curves.py
