
from pathlib import Path
import json

import numpy as np
import streamlit as st
from PIL import Image, ImageOps


# =====================================================
# 1. FILE PATHS
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "crop_disease_model.keras"
CLASS_NAMES_PATH = BASE_DIR / "class_names.json"


# =====================================================
# 2. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="CropCare AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 3. CUSTOM HEADER
# =====================================================

st.title("🌿 CropCare AI")

st.subheader("AI-powered Plant Disease Detection")

st.write(
    "Upload a photograph of a plant leaf to identify "
    "possible diseases using your trained deep-learning model."
)

st.divider()


# =====================================================
# 4. MODEL FILE STATUS
# =====================================================

model_exists = MODEL_PATH.is_file()
classes_exist = CLASS_NAMES_PATH.is_file()

if model_exists and classes_exist:
    st.success("Trained model and class labels detected.")
else:
    st.warning("Model files are missing.")

    if not model_exists:
        st.error(f"Missing model file: {MODEL_PATH.name}")

    if not classes_exist:
        st.error(f"Missing class file: {CLASS_NAMES_PATH.name}")


# =====================================================
# 5. SIDEBAR NAVIGATION
# =====================================================

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "How it works", "About project"]
)


# =====================================================
# 6. LOAD TRAINED MODEL
# =====================================================

@st.cache_resource(show_spinner="Loading trained AI model...")
def load_model():

    import tensorflow as tf

    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False
    )

    return model


# =====================================================
# 7. LOAD CLASS NAMES
# =====================================================

@st.cache_data
def load_labels():

    with open(
        CLASS_NAMES_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        labels = json.load(file)

    if isinstance(labels, dict):

        try:
            labels = [
                labels[key]
                for key in sorted(labels, key=lambda x: int(x))
            ]

        except (ValueError, TypeError):
            labels = list(labels.values())

    if not isinstance(labels, list) or len(labels) == 0:
        raise ValueError(
            "Invalid class_names.json format."
        )

    return [str(label) for label in labels]


# =====================================================
# 8. IMAGE PREPROCESSING
# =====================================================

def preprocess_image(image, model):

    input_shape = model.input_shape

    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = int(input_shape[1] or 160)
    width = int(input_shape[2] or 160)

    image = image.resize((width, height))

    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Assumes the trained model expects RGB pixels
    # in the 0-255 range.

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# =====================================================
# 9. DASHBOARD
# =====================================================

if page == "Dashboard":

    left, right = st.columns(2)

    with left:

        st.header("01. Upload leaf image")

        uploaded_file = st.file_uploader(
            "Choose a JPG, JPEG or PNG image",
            type=["jpg", "jpeg", "png"]
        )

        image = None

        if uploaded_file is not None:

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

            st.info(
                "Upload a leaf image to begin analysis."
            )

        elif not model_exists or not classes_exist:

            st.error(
                "Model files are unavailable. "
                "Please check the GitHub repository."
            )

        elif st.button(
            "Analyze leaf image",
            type="primary",
            use_container_width=True
        ):

            try:

                with st.spinner(
                    "AI is analyzing your leaf image..."
                ):

                    # Load model and labels

                    model = load_model()
                    labels = load_labels()

                    # Preprocess uploaded image

                    processed_image = preprocess_image(
                        image,
                        model
                    )

                    # Make prediction

                    predictions = model.predict(
                        processed_image,
                        verbose=0
                    )

                    predictions = np.asarray(
                        predictions
                    ).reshape(-1)

                    if len(predictions) != len(labels):

                        raise ValueError(
                            f"Model predicts {len(predictions)} "
                            f"classes, but class_names.json "
                            f"contains {len(labels)} labels."
                        )

                    # Get top predictions

                    top_indices = np.argsort(
                        predictions
                    )[::-1][:3]

                    best_index = int(top_indices[0])

                    result = labels[best_index]

                    confidence = float(
                        predictions[best_index]
                    )

                    # Display results

                    st.success(
                        "Analysis completed successfully!"
                    )

                    if "healthy" in result.lower():

                        st.markdown(
                            "### 🌱 Healthy leaf predicted"
                        )

                    else:

                        st.markdown(
                            "### ⚠️ Potential disease detected"
                        )

                    display_name = (
                        result
                        .replace("___", " — ")
                        .replace("_", " ")
                    )

                    st.subheader(display_name)

                    # Confidence score

                    st.metric(
                        "Prediction confidence",
                        f"{confidence * 100:.2f}%"
                    )

                    st.progress(
                        max(
                            0.0,
                            min(1.0, confidence)
                        )
                    )

                    # Top three predictions

                    st.write("### Top 3 predictions")

                    for index in top_indices:

                        label = labels[int(index)]

                        score = float(
                            predictions[int(index)]
                        )

                        formatted_label = (
                            label
                            .replace("___", " — ")
                            .replace("_", " ")
                        )

                        st.write(
                            f"**{formatted_label}** "
                            f"— {score * 100:.2f}%"
                        )

                    st.divider()

                    st.caption(
                        "AI screening result only. Confirm "
                        "suspected diseases with an agricultural "
                        "expert before treatment."
                    )

            except Exception as error:

                st.error("Prediction failed.")

                st.exception(error)


# =====================================================
# 10. HOW IT WORKS
# =====================================================

elif page == "How it works":

    st.header("How CropCare AI works")

    st.write(
        "CropCare AI uses a trained Convolutional Neural "
        "Network (CNN) to identify plant leaf diseases."
    )

    st.markdown(
        """
        **Step 1: Upload an image**

        Upload a photograph of a plant leaf.

        **Step 2: Image preprocessing**

        The image is resized according to the trained model's
        input dimensions.

        **Step 3: AI prediction**

        The CNN model analyzes the image and predicts
        the most likely class.

        **Step 4: Display results**

        The application displays the predicted disease,
        confidence score, and top three predictions.
        """
    )


# =====================================================
# 11. ABOUT PROJECT
# =====================================================

elif page == "About project":

    st.header("About CropCare AI")

    st.write(
        "CropCare AI is an academic project designed to "
        "identify possible plant diseases from leaf images "
        "using deep learning."
    )

    st.subheader("Technologies used")

    st.write("- Python")
    st.write("- Streamlit")
    st.write("- TensorFlow / Keras")
    st.write("- NumPy")
    st.write("- Pillow")

    st.subheader("Project objective")

    st.write(
        "The objective is to assist in early plant disease "
        "screening through an AI-powered image classification system."
    )

    st.warning(
        "This application is an academic prototype, "
        "not a substitute for professional plant disease diagnosis."
    )


# =====================================================
# 12. FOOTER
# =====================================================

st.divider()

st.caption(
    "🌿 CropCare AI | AI-powered Plant Health Screening"
)
