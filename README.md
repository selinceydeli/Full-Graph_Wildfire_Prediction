# Full-Graph Wildfire Prediction

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Prepare Data

Run the notebook that builds the graph/data artifacts:

```
prep_data.ipynb
```

Artifacts saved under `data/`:

* `timeseries_data.npy` — shape `(N, T)`
* `distance_matrix.npy` — shape `(N, N)` (projected-CRS centroid distances)
* `days.npy` — timeline for the columns of `timeseries_data`

## 3) Train a Model

### 3.1 Choose a model

In `train_models.py`, set:

```bash
SELECTED_MODEL = MODEL_NAMES[0]
```

Options:

* `0` → **Parametric GTCNN**
* `1` → **Disjoint Model**
* `2` → **Vanilla GCNN**

### 3.2 Run the training script

```bash
python train_models.py
```
