# Full-Graph Wildfire Prediction

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Prepare Data

Place the pre-generated pkl file `graphs_per_day.pkl` under the `data` folder:
`data/graphs_per_day.pkl`

Run the notebook that builds the graph/data artifacts:

```
prep_data.ipynb
```

Artifacts saved under `data/`:

* `days.npy` — timeline for the columns of `timeseries_data` [shape `(T,)`]
* `timeseries_data.npy` — node features per day [shape `(N, T, F)`]
* `labels.npy` — binary wildfire labels for each node/day [shape `(N, T)`]
* `distance_matrix.npy` — pairwise node distances between projected-CRS centroids [shape `(N, N)`]

## 3) Train a Model

### 3.1 Choose a model

In `train_models.py`, set:

```bash
SELECTED_MODEL = MODEL_NAMES[0]
```

Options:

* `0` → **Vanilla GCNN**
* `1` → **Parametric GTCNN**
* `2` → **Event Based Parametric GTCNN**

### 3.1.1 Scalability: training using clustering

If you choose one of the GTCNN models (model `1` or `2`), you can perform training using clusterGCN by setting:

```bash
CLUSTERING = True
```

For Vanilla GCNN (model `0`), training using clustering is not an option.

### 3.2 Run the training script

```bash
python train_models.py
```
