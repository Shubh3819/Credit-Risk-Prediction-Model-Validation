# Credit Risk Portfolio Analytics & Model Validation

An interactive Streamlit dashboard for exploring a credit portfolio, reviewing data quality, comparing baseline credit-risk models, and flagging unusual account profiles. The project also includes a local **Ask the Portfolio** assistant powered by Ollama.

> **Important:** This is an educational prototype built with synthetic data. It is not approved for live lending decisions, customer treatment, or production use. Model metrics are preliminary and must be independently validated before any real-world use.

## Features

- **Portfolio Overview & Analytics** — explore portfolio KPIs and segment-level patterns using the available dataset fields.
- **Model Evaluation** — review Logistic Regression and XGBoost evaluation outputs, including saved metrics and plots.
- **Anomaly Detection** — use Isolation Forest to flag unusual origination/profile combinations for human review. An anomaly is not proof of fraud or default.
- **Account Explorer** — inspect available account-level information and prediction outputs.
- **Data Quality** — review data-quality checks and known data limitations.
- **Ask the Portfolio (optional)** — ask natural-language questions about supported portfolio calculations using a local Ollama model. Numerical calculations are performed by the application; the language model provides explanatory text.

## Project structure

```text
credit-risk-dashboard/
├── app.py
├── requirements.txt
├── Model_File.ipynb
├── data/
│   ├── customers.csv
│   ├── loan_accounts.csv
│   ├── bureau_credit.csv
│   ├── repayment_history.csv
│   └── collections_events.csv
├── model_artifacts/
│   ├── model_metrics.csv
│   ├── test_set_predictions.csv
│   ├── model_df_engineered.csv
│   ├── logistic_regression_pipeline.joblib
│   ├── xgboost_pipeline.joblib
│   ├── isolation_forest_origination.joblib
│   └── evaluation plots and supporting notes
├── product_brief.docx
├── product_brief.pdf
├── architecture_diagram.png
├── assumptions_log.docx
└── cro_pitch_deck.pptx
```

Some generated or local-only files may be omitted from the repository by `.gitignore`.

## Requirements

- Windows, macOS, or Linux
- Python version compatible with the packages pinned in `requirements.txt` (Python 3.10–3.12 is a reasonable starting point; use the version that matches the saved model artifacts and installed dependencies)
- Git (optional, for cloning the repository)
- Ollama only if you want to use the local Ask the Portfolio assistant

## Installation (Windows PowerShell)

### 1. Clone the repository

```powershell
git clone https://github.com/Shubh3819/Credit-Risk-Prediction-Model-Validation.git
cd Credit-Risk-Prediction-Model-Validation
```

If you already downloaded the repository as a ZIP, extract it and open PowerShell in the extracted project folder instead.

### 2. Create and activate a virtual environment

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation for this terminal session, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Install Python dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Start the dashboard

From the project root (the folder containing `app.py`), run:

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints in the terminal, usually `http://localhost:8501`.

Keep the terminal open while using the dashboard. To stop it, press `Ctrl+C`.

## Enable the Ask the Portfolio assistant (optional)

The dashboard's chat requires Ollama to be installed and running locally. The rest of the dashboard may be usable without the chat, depending on the current app configuration.

1. Install Ollama from [https://ollama.com](https://ollama.com).
2. In PowerShell, download the model:

   ```powershell
   ollama pull llama3.2:3b
   ```

3. Confirm the model is installed:

   ```powershell
   ollama list
   ```

4. Make sure the Ollama service is running. The app expects the local API at:

   ```text
   http://localhost:11434
   ```

5. Test the API:

   ```powershell
   Invoke-RestMethod -Uri "http://localhost:11434/api/tags"
   ```

6. Start or restart the Streamlit app:

   ```powershell
   streamlit run app.py
   ```

Open the floating **Ask the Portfolio** chat and use its connection test if available.

**Local vs hosted:** `localhost:11434` refers to the computer running the dashboard. A hosted Streamlit deployment cannot reach Ollama on your personal computer through that address. A hosted setup needs a separately deployed, secured model service and appropriate network/security controls.

## Data and model artifacts

The dashboard expects the CSV files and model artifacts in the project-relative `data/` and `model_artifacts/` folders. Keep their folder names and filenames unchanged unless you also update the paths in `app.py`.

The saved `.joblib` models were created in a particular Python and package environment. If loading fails, check the versions in `requirements.txt` and use a compatible Python/scikit-learn/XGBoost environment rather than blindly retraining or replacing the files.

The project data is synthetic. Do not replace it with real customer or bank data in a public repository. Real data would require authorization, data minimization, masking or anonymization, access controls, and an approved storage location.

## Model validation and interpretation notes

- The current Logistic Regression and XGBoost results are preliminary and should not be treated as production validation.
- The target definition, observation window, label maturity, and scoring point must be confirmed before interpreting model performance.
- Repayment features observed after origination can introduce target leakage if the intended use is an origination-time prediction.
- Evaluate on a leakage-safe, time-based holdout and review calibration, segment stability, and business costs of false positives and false negatives.
- Isolation Forest flags unusual profiles for investigation; it does not establish fraud, default, or misconduct.
- Use human review for exceptions. Do not use this prototype to automatically approve or decline credit.

See `assumptions_log.docx` and `model_artifacts/model_limitations.md` for project-specific assumptions and limitations.

## Monitoring approach

A proposed post-deployment monitoring plan is to review data quality, score distributions, portfolio composition, and PSI/CSI monthly. Investigate PSI above 0.10 as an early warning and PSI above 0.25 as a potentially material shift, subject to bank-approved thresholds. Once outcomes have matured, monitor AUC, KS, Gini, precision, recall, calibration, and realized bad rates. Start with a quarterly retraining review, or investigate earlier when drift or sustained performance degradation appears. Retraining should only follow leakage-safe validation, model-risk review, documented approval, and a rollback plan.

These are proposed starting points, not bank-approved policy thresholds or evidence that the prototype has been deployed.

## Assignment deliverables

- `product_brief.docx` and `product_brief.pdf` — product brief
- `architecture_diagram.png` — architecture overview
- `cro_pitch_deck.pptx` — 10-minute CRO briefing
- `assumptions_log.docx` — assumptions, limitations, and bank readiness requirements
- `Model_File.ipynb` — model development and evaluation notebook

## Troubleshooting

**`streamlit` is not recognized**
```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

**CSV or model file not found**
- Run commands from the project root.
- Confirm the `data/` and `model_artifacts/` directories exist and contain the expected files.
- Do not rename expected files without updating the application paths.

**Ollama connection error**
- Confirm Ollama is running.
- Run `ollama list` and confirm `llama3.2:3b` is installed.
- Test `http://localhost:11434/api/tags`.
- Remember that Ollama must be reachable from the same machine/environment where Streamlit runs.

## Disclaimer

This repository is a prototype for learning and portfolio analytics. It does not constitute financial, lending, legal, or regulatory advice and has not been certified for production credit-risk decisioning.
