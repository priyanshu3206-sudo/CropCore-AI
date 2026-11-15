# CropCare AI — Crop Disease Detection Web App

A Streamlit application scaffold for the PlantVillage CNN classifier trained in Google Colab.

## 1. Project structure

```text
CropCare-AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── model/
│   ├── README.txt
│   ├── crop_disease_model.keras   # add after training
│   └── class_names.json           # add after training
└── assets/
    └── README.txt
```

## 2. Requirements

- Python 3.10 or 3.11 is recommended.
- VS Code with the Python extension.
- A trained model exported from the supplied Colab notebook.

## 3. Setup in VS Code

Open the `CropCare-AI` folder in VS Code, then open Terminal.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Add your trained model (integration point)

Copy the two output files downloaded from Google Colab into the `model/` folder:

- `crop_disease_model.keras`
- `class_names.json`

The JSON file must be the class-name list exported from the **same training run** as the model. The supplied notebook saves both files with these names.

The app is intentionally usable before model integration. Without the two files, the upload interface loads and displays a clear “integration pending” message when Analyze is clicked.

## 5. Run the application

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## 6. Model preprocessing compatibility

The app reads the expected image dimensions from the Keras model input shape, resizes the uploaded image, converts it to RGB, and sends pixel values in the 0–255 range.

This matches the provided Colab notebook, where `Rescaling(1./127.5, offset=-1)` is included **inside the model**. Do not add another external normalization step unless you change the model architecture and training preprocessing.

## 7. Troubleshooting

- **Model integration pending:** confirm both model files exist in `model/`.
- **Class count mismatch:** use the `class_names.json` generated alongside that exact `.keras` model.
- **TensorFlow import/load error:** check your Python version and install a TensorFlow version compatible with it.
- **PowerShell activation error:** use Command Prompt activation shown above, or configure your local execution policy if permitted by your organization.
- **No GPU:** the app performs inference on CPU; it may simply take longer.

## 8. Important project limitation

PlantVillage is a controlled image dataset. Performance on real farm photographs can be lower due to background, lighting, camera and disease-stage differences. This application is an academic screening prototype, not a field-validated diagnostic or treatment recommendation system.
