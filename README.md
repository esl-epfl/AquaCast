# AquaCast — Transformer-based Time-Series Forecasting (Research Code)

Deep learning models for **multi-variate, multi-horizon time-series forecasting**, with a focus on urban water and hydrological dynamics.

This repository contains:
- Research code for **AquaCast**, and **PatchTST**
- Synthesized datasets for public reproducibility
- Scripts and Docker support for controlled experimentation

---

## 📄 Paper / Preprint

**AquaCast: Precipitation-Informed Transformer for Urban Water Dynamics Forecasting**  
arXiv: https://arxiv.org/abs/2509.09458  

**Status:**  
This work is **submitted and currently under revision**.  
(The arXiv version should be cited when referring to this repository.)

---

## 🧪 Repository Status

- **Research code**
- Fully reproducible on **synthetic datasets**
- **Real-world and partner datasets are not included** due to non-disclosure agreements (NDAs)

---

## 📁 Data Overview

This repository includes **synthetic and real data *schemas*** (names and versions only).

- Synthesized datasets:  
  `AquaCast/Data/`  
  (see `AquaCast/Data/README.md` for full documentation)

---

## 🐳 Docker

To build the Docker image:

```bash
docker build -t synth:latest .
```

## Install (local)
```bash
python -m venv .venv && source .venv/bin/activate   # or conda
pip install -r requirements.txt
```

## ▶️ Running Experiments

All experiments are executed via bash scripts located under the scripts/ directory,
organized by model.

To run any experiment, use the following command pattern:
```bash
bash scripts/[model]/[script].sh
```
Examples:
```bash
bash scripts/AquaCast/synthesized.sh
bash scripts/PatchTST/traiLausanneCity.sh
```
## Citation
If you use this repository, code, or synthesized datasets in academic work, please cite:
```bibtex
@misc{abdollahinejad2025aquacast,
  title         = {AquaCast: Precipitation-Informed Transformer for Urban Water Dynamics Forecasting},
  author        = {AbdollahiNejad, Golnoosh and collaborators},
  year          = {2025},
  eprint        = {2509.09458},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  note          = {Submitted, under revision}
}
```

## Acknowledgement

We appreciate the following github repo very much for the valuable code base and datasets:

https://github.com/yuqinie98/PatchTST
