# Full-Graph Wildfire Prediction

## 1) Install Dependencies

```bash
pip install -r requirements.txt
```

## 2) Prepare Data

Place the pre-generated pkl file `graphs_per_day.pkl` under the `data` folder:
`data/graphs_per_day.pkl`

Make sure to delete all previously generated adjacency matrices (i.e., all `.npy` files in the `data` folder).

Run the notebook that builds the graph/data artifacts:

```
prep_data.ipynb
```

Artifacts saved under `data/`:

- `days.npy` — timeline for the columns of `timeseries_data` [shape `(T,)`]
- `timeseries_data.npy` — node features per day [shape `(N, T, F)`]
- `labels.npy` — binary wildfire labels for each node/day [shape `(N, T)`]
- `distance_matrix.npy` — pairwise node distances between projected-CRS centroids [shape `(N, N)`]

## 3) Train a Model

`train_models.py` is a CLI entry point that trains one of three models on your wildfire dataset and evaluates on the test split.

### 3.1 Model Options

#### 1) Vanilla GCNN

A fast spatial baseline. It applies graph convolutions on node features where the temporal window is flattened into the feature axis.

- Input shape used internally: [B, N, F × T] (we flatten the last two dims)
- Captures spatial correlations well, treats time as extra features

#### 2) Parametric GTCNN

A spatio-temporal graph model with compact, parameterized temporal filters.

- Input shape: [B, F, N, T] (keeps time explicit)
- Learns temporal kernels and spatial graph filters jointly

#### 3) Event-based GTCNN

Extends the Parametric GTCNN to focus learning around event times.

- Input shape: [B, F, N, T] plus `event_times` of shape [B, T]
- Uses an event-centric kernel (e.g., exponential with `tau`, `max_back_hops`) to weigh temporal edges

### 3.2 CLI arguments

| Flag                                    | Type     | Default                  | Choices/Format                                         | Description                                                                   |
| --------------------------------------- | -------- | ------------------------ | ------------------------------------------------------ | ----------------------------------------------------------------------------- |
| --days_data_path                        | str      | data/days.npy            | path                                                   | Path to the timeline array used to derive event windows                       |
| --timeseries_data_path                  | str      | data/timeseries_data.npy | path                                                   | Path to the feature tensor shaped (N, T, F)                                   |
| --labels_path                           | str      | data/labels.npy          | path                                                   | Path to the label tensor shaped (N, T) with 0/1 labels                        |
| <nobr>--distance_matrix_filepath</nobr> | str      | data/distance_matrix.npy | path                                                   | Path to pairwise distances shaped (N, N) used to build the kNN graph          |
| --pred_horizon                          | int      | 1                        | positive integer                                       | How many steps ahead to predict                                               |
| --obs_window                            | int      | 4                        | positive integer                                       | Temporal window size used by the models                                       |
| --k                                     | int      | 4                        | positive integer                                       | Number of neighbors kept when building the kNN graph                          |
| --num_epochs                            | int      | 50                       | positive integer                                       | Training epochs                                                               |
| --batch_size                            | int      | 16                       | positive integer                                       | Minibatch size for training and validation                                    |
| --selected_loss_function                | str      | bce                      | bce, weighted_bce, focal, dice                         | Training loss; weighted_bce auto-computes pos_weight from train labels        |
| --selected_model                        | str      | vanilla_gcnn             | vanilla_gcnn, parametric_gtcnn, parametric_gtcnn_event | Model to train                                                                |
| --train_val_test_split                  | 3 floats | 0.6 0.2 0.2              | three numbers summing to 1                             | Fractions for train, validation, test in time order                           |
| --threshold_tp                          | float    | 0.5                      | 0.0–1.0                                                | Threshold used by the F1 metric (does not affect the loss)                    |
| --clustering                            | bool     | False                    | True or False                                          | Train cluster by cluster; only for parametric_gtcnn or parametric_gtcnn_event |

### 3.3 Train the Selected Model

Here are some example commands to train the selected model:

#### 1) Vanilla GCNN (full-batch training)

```bash
python3 train_models.py \
  --selected_model vanilla_gcnn \
  --selected_loss_function bce \
  --obs_window 4 \
  --pred_horizon 1 \
  --k 4 \
  --num_epochs 10 \
  --batch_size 16
```

#### 2) Parametric GTCNN with weighted BCE loss and mini-batch training

```bash
python3 train_models.py \
  --selected_model parametric_gtcnn \
  --selected_loss_function weighted_bce \
  --num_epochs 10 \
  --batch_size 16 \
  --clustering True
```

#### 3) Event-based GTCNN with Focal Loss and mini-batch training

```bash
python3 train_models.py \
  --selected_model parametric_gtcnn_event \
  --clustering True \
  --selected_loss_function focal \
  --num_epochs 10 \
  --batch_size 16 \
  --clustering True
```
