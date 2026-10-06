#!/bin/bash
# GitHub Actions çalışmazsa yerel yedek: katalog güncelle, (varsa) yarının pencerelerini yayınla, bitenleri puanla.
# Kullanım: bash prospektif/yerel_calistir.sh [YYYY-MM-DD]
set -e
cd "$(dirname "$0")/.."
python3 src/prospective/live_catalog.py
python3 src/prospective/schedule.py $1
python3 src/prospective/evaluate_forecasts.py
