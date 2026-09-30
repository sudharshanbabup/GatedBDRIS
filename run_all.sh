#!/usr/bin/env bash
# One shot reproduction of every number in the article.
set -e
export NREAL=${NREAL:-50} NHI=${NHI:-30} NCDF=${NCDF:-100} NPART=${NPART:-100} WORKERS=${WORKERS:-2}
cd "$(dirname "$0")/src"
python3 runner.py 123456789             # base campaign -> ../results
python3 runner_ext.py samkfdiclerx      # ablation, scaling, sensitivity, CSI, IL, echo, access, 20 inits
python3 make_figs.py all                # ../results/summary.json and quick look figures (../tex/figs)
python3 make_figs_ext.py
python3 verify.py all                   # analytic checks -> ../results/verify.json
python3 make_facts.py                   # every quoted number -> ../tex/facts.tex
python3 make_facts_ext.py
echo "Done. Numbers in tex/facts.tex, checks in results/verify.json"
