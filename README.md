# AutoGluon – Applied Notebooks & Mini-Projects

End-to-end examples using **[AutoGluon](https://auto.gluon.ai/)** for tabular machine learning (classification & regression), plus a few experimental side projects.  
Each folder contains a ready-to-run Jupyter notebook that demonstrates data loading, feature engineering, model training, leaderboard analysis, and Kaggle-ready predictions.

---

## 📁 Repository Structure

```

AutoGluon/
├── california housing price/
│   └── <notebook>.ipynb               # Regression on California Housing dataset
├── fraud_detection/
│   └── <notebook>.ipynb               # Binary classification for fraud detection
├── knot_theory/
│   └── <notebook>.ipynb               # Experimental / research sandbox
├── pet_finder/
│   └── <notebook>.ipynb               # PetFinder use case (classification/regression)
└── titanic_feature_engineering/
└── <notebook>.ipynb               # Feature engineering showcase on Titanic dataset

````

---

## ⚙️ Setup & Installation

### Requirements
- Python 3.9 or later
- Jupyter Notebook or JupyterLab
- Optional GPU acceleration

Install dependencies:
```bash
pip install -U autogluon.tabular pandas ipywidgets kaggle
````

If running on **Kaggle**, most packages are preinstalled.
If using the **Kaggle API**, make sure your API key (`kaggle.json`) is set in `~/.kaggle/`.

---

## 🚀 Quick Start

Clone the repository and open the desired project folder:

```bash
git clone https://github.com/ykalathiya-2/AutoGluon.git
cd AutoGluon

# Example: Titanic
cd titanic_feature_engineering
jupyter lab
```

Each notebook will:

1. Load the dataset (via Kaggle API or local `./data/` directory).
2. Train a baseline AutoGluon model.
3. Apply feature engineering and compare multiple model configurations.
4. Display leaderboards and feature importances.
5. Export predictions as `submission.csv`.

---

## 📊 Data Access

Datasets should be placed in a local `./data/` folder.

Example for Kaggle Titanic:

```bash
kaggle competitions download -c titanic -p data
unzip -o data/titanic.zip -d data
```

---

## 🧠 Typical Notebook Workflow

1. **Load & Inspect Data**
   Explore the dataset using `pandas`.

2. **Train Baseline Model**

   ```python
   from autogluon.tabular import TabularPredictor
   predictor = TabularPredictor(label='target', eval_metric='accuracy').fit(train_data)
   predictor.leaderboard(train_data)
   ```

3. **Feature Engineering**
   Create domain features or use AutoGluon’s `FeatureGenerator` (e.g., `TextNgramFeatureGenerator`, `FillNaFeatureGenerator`).

4. **Evaluate & Interpret**
   Check model performance and feature importance.

5. **Full Refit & Prediction**

   ```python
   _ = predictor.refit_full()
   preds = predictor.predict(test_data)
   ```

---

## ⚠️ Notes

* `refit_full()` returns a dictionary of `_FULL` models but updates the same predictor.
  Keep using `predictor.predict()` rather than calling methods on the dict.
* Use `IdentityFeatureGenerator()` if you wish to disable AutoGluon’s automatic feature engineering.

---

## 🧩 Useful Settings

* **Quality presets:** `'medium_quality_faster_train'`, `'high_quality'`, `'best_quality'`
* **Cross-validation:** `num_bag_folds`, `num_stack_levels`
* **Minimal preprocessing:**

  ```python
  from autogluon.features.generators import IdentityFeatureGenerator
  predictor = TabularPredictor(label='target').fit(train_data, feature_generator=IdentityFeatureGenerator())
  ```

---

## 📚 References

* [AutoGluon Documentation](https://auto.gluon.ai/stable/index.html)
* [AutoGluon GitHub](https://github.com/autogluon/autogluon)

---
