# Eye Region Detection

*A The Turing Way inspired project to enable reproducibility in data science.*

## About this Repository

This repository is for the detection of eye regions in images. The data the model is trained on is
all very similar to the image shown below (and as such is very constrained):


## Repo Structure

Inspired by [Cookie Cutter Data Science](https://github.com/drivendata/cookiecutter-data-science)

```

├── README.md          <- The top-level README for users of this project.
├── data
│   ├── processed      <- The final, canonical data sets for modeling. Not stored on the repo.
│   └── raw            <- The original, immutable data dump. This is not stored on the repo.
│
├── models             <- Trained and serialized models, model predictions, or model summaries.
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── src                <- Source code for use in this project.
│   │
│   ├── data           <- Scripts to download or generate data
│   │   └── construct_dataset.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── predict_model.py
│   │   ├── train_model.py
|   |   ├── model_utils.py
|   |   └──run_file.sh
│   │
│   └── visualisation  <- Scripts to create exploratory and results oriented visualisations
|       |
|       ├── inspect_predicted_results.py
|       ├── inspect_region_labels.py
│       └── utils.py

```
