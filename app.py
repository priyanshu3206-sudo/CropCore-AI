
from pathlib import Path
import json

import numpy as np
import streamlit as st
from PIL import Image, ImageOps

# ---------------- PATHS ----------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "crop_disease_model.keras"
CLASS_PATH = BASE_DIR / "model" / "class_names.json"

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌿",
    layout="wide"
)

# ---------------- SIMPLE DESIGN ----------------

st.title("🌿 CropCare AI")
st.subheader("AI-powered Plant Disease Detection")

st.write(
    "Upload a photograph of a plant leaf to identify "
    "possible diseases using your trained deep-learning model."
)

st.divider()

# ---------------- MODEL STATUS ----------------

if MODEL_PATH.exists() and CLASS_PATH.exists():
    st.success("Trained model and class labels detected.")
else:
    st.warning("Model files are missing. Check your model folder.")

# ---------------- NAVIGATION ----------------

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "How it works", "About project"]
)

# ---------------- MODEL LOADER ----------------

@st.cache_resource(show_spinner="Loading trained model...")
def load_model():
    import tensorflow as tf
    return tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )


@st.cache_data
def load_labels():
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        labels = json.load(file)

    if isinstance(labels, dict):
        labels = list(labels.values())

    if not isinstance(labels, list) or not labels:
        raise ValueError("Invalid class_names.json format.")

    return [str(label) for label in labels]


# ---------------- DASHBOARD ----------------

if page == "Dashboard":

    left, right = st.columns(2)

    with left:
        st.header("01. Upload leaf image")

        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG or PNG image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = ImageOps.exif_transpose(
                Image.open(uploaded_file)
            ).convert("RGB")

            st.image(
                image,
                caption="Uploaded leaf image",
                use_container_width=True
            )

    with right:
        st.header("02. AI analysis")

        if uploaded_file is None:
            st.info("Upload a leaf image to begin analysis.")

        elif not MODEL_PATH.exists() or not CLASS_PATH.exists():
            st.error("Please add both model files to the model folder.")

        elif st.button("Analyze leaf image", type="primary"):
            try:
                with st.spinner("Analyzing leaf image..."):

                    model = load_model()
                    labels = load_labels()

                    input_shape = model.input_shape

                    if isinstance(input_shape, list):
                        input_shape = input_shape[0]

                    height = int(input_shape[1] or 160)
                    width = int(input_shape[2] or 160)

                    processed = image.resize((width, height))

                    array = np.asarray(
                        processed,
                        dtype=np.float32
                    )

                    # IMPORTANT:
                    # This assumes the saved model accepts RGB
                    # pixel values in the 0-255 range.
                    batch = np.expand_dims(array, axis=0)

                    predictions = model.predict(
                        batch,
                        verbose=0
                    )[0]

                    if len(predictions) != len(labels):
                        raise ValueError(
                            f"Model outputs {len(predictions)} classes, "
                            f"but the label file contains {len(labels)}."
                        )

                    top_indices = np.argsort(predictions)[::-1][:3]
                    best_index = int(top_indices[0])

                    result = labels[best_index]
                    confidence = float(predictions[best_index])

                    st.success("Analysis completed!")

                    if "healthy" in result.lower():
                        st.markdown("### 🌱 Healthy leaf predicted")
                    else:
                        st.markdown("### ⚠️ Potential disease predicted")

                    st.subheader(result.replace("___", " — ").replace("_", " "))

                    st.metric(
                        "Prediction confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    st.progress(
                        max(0.0, min(1.0, confidence))
                    )

                    st.write("### Top 3 predictions")

                    for index in top_indices:
                        label = labels[int(index)]
                        score = float(predictions[int(index)])

                        st.write(
                            f"**{label.replace('___', ' — ').replace('_', ' ')}** "
                            f"— {score * 100:.2f}%"
                        )

                    st.caption(
                        "AI screening result only. Confirm suspected diseases "
                        "with an agricultural expert before treatment."
                    )

            except Exception as error:
                st.error("Prediction failed.")
                st.exception(error)


# ---------------- HOW IT WORKS ----------------

elif page == "How it works":

    st.header("How CropCare AI works")

    st.write("1. Upload a plant leaf photograph.")
    st.write("2. The image is resized for the trained model.")
    st.write("3. The CNN predicts the most likely classification.")
    st.write("4. The application displays the prediction and confidence.")


# ---------------- ABOUT ----------------

elif page == "About project":

    st.header("About CropCare AI")

    st.write(
        "CropCare AI is an academic project that uses a "
        "convolutional neural network to classify plant leaf images."
    )

    st.write("**Technologies used:**")
    st.write("- Python")
    st.write("- Streamlit")
    st.write("- TensorFlow / Keras")
    st.write("- NumPy")
    st.write("- Pillow")

    st.warning(
        "This application is an academic prototype, not a substitute "
        "for professional plant disease diagnosis."
    )

st.divider()
st.caption("CropCare AI | Plant Health Screening")
