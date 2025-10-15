# Full-Graph Wildfire Prediction

## Install Dependencies
```bash
pip install -r requirements.txt
```

## Training a Model

1. **Select which model to train**
Inside `train_models.py`, choose one of the predefined models:
```bash
SELECTED_MODEL = MODEL_NAMES[0]
```
Model options:
- `0` -> **Parametric GTCNN**
- `1` -> **Disjoint Model**
- `2` -> **Vanilla GCNN**

2. **Run the training script**
```bash
python train_models.py
```