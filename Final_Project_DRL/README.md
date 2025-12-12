# Demo Instructions

This README explains how to demo the Final_Project_DRL project locally. It assumes you already have the ROM and savestates placed under `env/assets/` as provided.

## Setup
1) Create a Python 3.11+ virtual environment and activate it.
2) Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

## Running a Quick Demo
- Start a short slim run (lightweight):
```
python training/train_slim.py --config configs/config_slim.yaml --total-steps 5000 --record-video-every 1 --log-dir runs/slim_demo
```
This writes logs and videos under `runs/slim_demo/`.

- Start a small big-model run (heavier):
```
python training/train_big.py --config configs/config_big.yaml --total-steps 20000 --record-video-every 1 --log-dir runs/big_demo
```
Logs and checkpoints go to `runs/big_demo/`.

## Viewing Outputs
- Demo videos: check `runs/<run_name>/videos/`.
- Logs: events/episodes/progress CSVs inside each run directory.

## Regenerating Figures
After runs are available:
```
python analysis/final_analysis_rnd.py --run-dirs runs/big_demo runs/slim_demo
python analysis/final_analysis_rewards.py --run-dirs runs/big_demo runs/slim_demo
python analysis/final_map_visitation.py --run-dirs runs/big_demo runs/slim_demo
python analysis/metrics.py --run-dirs runs/big_demo runs/slim_demo
```
Figures save to `report/figs/final/` and tables to `analysis/tables/`.

## Compiling the Report
```
cd report
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
Ensure figures under `report/figs/final/` are present.

## Notes
- Set `--headless` or `--no-headless` as needed; defaults are headless for server use.
- Checkpoints land in `runs/<run>/checkpoints/`; resume with `--resume` or `--checkpoint`.
- ROM and savestates must remain in `env/assets/`.
