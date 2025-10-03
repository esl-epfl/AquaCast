# My New Repo

Synth & real *Data versions* and names tables

Build the docker image:
docker build -t synth:latest .

# AquaCast — Transformer-based Time-Series Forecasting (Research Code)

**What:** DL models (MyTransformer, PatchTST variants) for multi-variate, multi-horizon forecasting.  
**Paper / Preprint:** (add link)  
**Status:** Research code; reproducible on a tiny synthetic/sample dataset. Real/partner data is **not** included.

## Install
```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

## Dataset Style
Dataset structure for time-series forecasting in CSV columns' format

1. only endogenous time-series

    date, node1, node2, ...

2. Including endogenous time-series history

    date, node1, node2, ..., rain

3. using perfect exogenous forecast

    a. AquaCast

        date, node1, node2, ..., rain

    b. PatchTST (rain_forecast_steps should be equal to history steps)
    
        date, node1, node2, ..., rain, rain_forecast_steps



## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST