# GTA VI Trophy Rarity Forecast

Forecasts each PlayStation 5 trophy's earn rate on Exophase one week after launch, as five quantiles (5%, 25%, 50%, 75%, 95%) per trophy.

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kui123456789/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kui123456789/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/competitions/gta6-trophy-rarity-forecast-01)

## Method

Damped log-time trend in logit space, with a frozen-page refresh prior and a horizon-growing spread calibrated on the practice refresh. See the docstring in `main.py` for the details.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `<id>.in` in the input directory and writes `<id>.out` per case. Python 3.13, standard library only; offline, deterministic, no clock, no randomness.
