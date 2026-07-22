import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
import cv2
import os
import io

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader

# ------------------------------------
# ⚙️ Paths
# ------------------------------------
MODEL_PATH = "pneumonia_model1.h5"
# TODO: 🔁 UPDATE THIS to your real train data path
TRAIN_DATA_PATH = r"C:\Users\muham\Desktop\aiml\pneunomiadata\chest_xray\train"

# ------------------------------------
# 🩺 Page Setup
# ------------------------------------
st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁", layout="wide")

st.markdown(
    """
    <h1 style='text-align:center; font-size:40px;'>🩺 Pneumonia Detection from Chest X-Ray</h1>
    <p style='text-align:center; font-size:18px;'>
    Upload a chest X-ray, fill optional patient details, and receive an AI-assisted pneumonia evaluation with anomaly detection & Grad-CAM.
    </p>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------
# 🌗 Dark / Light Mode
# ------------------------------------
if "theme" not in st.session_state:
    st.session_state.theme = "light"

theme_checkbox = st.sidebar.checkbox("🌗 Dark Mode", value=(st.session_state.theme == "dark"))
st.session_state.theme = "dark" if theme_checkbox else "light"

if st.session_state.theme == "light":
    modern_css = """
    <style>
    :root {
      --bg: #f6f8fa;
      --card: #ffffff;
      --muted: #6b7280;
      --primary: #0f62fe;
      --radius: 14px;
    }
    body, .block-container, .stApp {
      background: var(--bg) !important;
      color: #5DADE2 !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    h1, h2, h3, h4, h5, h6, p, label, span {
      color: #5DADE2 !important;
    }
    .stButton>button {
      border-radius: 10px;
      padding: 8px 14px;
      background: linear-gradient(90deg, #0f62fe, #4f8dff) !important;
      color: white !important;
      border: none !important;
    }
    .stButton>button:hover {
      background: linear-gradient(90deg, #0d4ad6, #3869d4) !important;
      color: white !important;
    }
    .stDownloadButton>button {
      background: linear-gradient(90deg, #0f62fe, #4f8dff) !important;
      color: white !important;
    }
    .stDownloadButton>button:hover {
      background: linear-gradient(90deg, #0d4ad6, #3869d4) !important;
      color: white !important;
    }
    </style>
    """
else:  # Dark mode
    modern_css = """
    <style>
    body, .block-container, .stApp {
      background: #0e1117 !important;
      color: #5DADE2 !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
    }
    h1, h2, h3, h4, h5, h6, p, label, span, div {
      color: #5DADE2 !important;
    }
    input, textarea, select {
      background-color: #1b1f27 !important;
      color: #5DADE2 !important;
      border: 1px solid #333 !important;
    }
    .stButton>button {
      border-radius: 10px;
      padding: 8px 14px;
      background: linear-gradient(90deg, #1d7dd4, #4f9cff) !important;
      color: white !important;
      border: none !important;
    }
    .stButton>button:hover {
      background: linear-gradient(90deg, #1862a8, #3876cc) !important;
      color: white !important;
    }
    .stDownloadButton>button {
      background: linear-gradient(90deg, #1d7dd4, #4f9cff) !important;
      color: white !important;
    }
    .stDownloadButton>button:hover {
      background: linear-gradient(90deg, #1862a8, #3876cc) !important;
      color: white !important;
    }
    </style>
    """
st.markdown(modern_css, unsafe_allow_html=True)

# ------------------------------------
# 🧍 Patient Details (Sidebar)
# ------------------------------------
st.sidebar.header("🧍 Patient Details")
name = st.sidebar.text_input("Patient Name", placeholder="Enter name...", key="name")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30, key="age")
sex = st.sidebar.selectbox("Sex", ["Male", "Female"], key="sex")
st.sidebar.markdown("**Symptoms**")
fever = st.sidebar.checkbox("Fever", key="fever")
breathing_difficulty = st.sidebar.checkbox("Breathing Difficulty", key="breathing_difficulty")
cough = st.sidebar.checkbox("Cough", key="cough")
st.sidebar.markdown("---")

# ------------------------------------
# ✅ Utility: Confidence Bars
# ------------------------------------
def show_conf_bar_dual(confidence: float, label: str, color: str):
    """Bar used for Pneumonia vs Normal (with explicit color)."""
    bg = "#3c3c3c" if st.session_state.theme == "dark" else "#dcdcdc"
    st.markdown(
        f"""
        <div style="margin-bottom: 12px; font-size:18px;">
            <div style="display:flex; justify-content:space-between;">
                <b>{label}</b>
                <span>{confidence:.2f}%</span>
            </div>
            <div style="background-color:{bg}; border-radius:8px; width:100%; height:28px; margin-top:4px;">
                <div style="width:{confidence}%; background-color:{color}; height:28px; border-radius:8px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def show_conf_bar_single(confidence, label):
    """Single bar used for overall confidence (image-only)."""
    color = "red" if label == "Pneumonia" else "green"
    st.markdown(f"""
        <div style="background-color: lightgray; border-radius: 10px; width: 100%; height: 25px;">
            <div style="width: {confidence}%; background-color: {color}; height: 25px; border-radius: 10px; 
            border-top-left-radius: 10px; border-bottom-left-radius: 10px;
            border-top-right-radius: {'10px' if confidence == 100 else '0px'};
            border-bottom-right-radius: {'10px' if confidence == 100 else '0px'};"></div>
        </div>
    """, unsafe_allow_html=True)

def get_severity(confidence):
    """Determines the severity based on the prediction confidence (0-100)."""
    if confidence < 60:
        return "Mild"
    elif confidence < 85:
        return "Moderate"
    else:
        return "Severe"

# ------------------------------------
# 🧠 Load Model
# ------------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure the file is in the correct directory.")
    st.stop()

@st.cache_resource
def load_model():
    """Loads the pre-trained Sequential Keras model."""
    try:
        import logging
        logging.getLogger('tensorflow').setLevel(logging.ERROR)

        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

seq_model = load_model()
if seq_model is None:
    st.stop()

# We'll also use it as `model` for feature extraction
model = seq_model

# Ensure model is built
if not model.built:
    model.build(input_shape=(None, 150, 150, 1))

# ------------------------------------
# 🔹 Feature extractor
# ------------------------------------
def extract_features(img_array):
    """
    Extracts features up to (but not including) the final classification layer.
    """
    x = img_array
    for layer in model.layers[:-1]:
        x = layer(x)
    return x

# ------------------------------------
# 🔹 Compute training stats for Mahalanobis distance
# ------------------------------------
@st.cache_data
def compute_training_stats():
    """Computes the mean vector and inverse covariance matrix from training data features."""
    try:
        feature_size = model.layers[-2].output.shape[-1]
    except Exception as e:
        st.error(f"Could not determine feature size from model architecture: {e}")
        return np.zeros((1024,)), np.eye(1024)

    if not os.path.isdir(TRAIN_DATA_PATH):
        st.warning(f"Training data path not found: {TRAIN_DATA_PATH}. Mahalanobis distance for outlier detection is disabled.")
        return np.zeros((feature_size,)), np.eye(feature_size)

    try:
        train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
        train_set = train_gen.flow_from_directory(
            TRAIN_DATA_PATH,
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary',
            color_mode='grayscale',
            shuffle=False
        )

        if train_set.n == 0:
            st.warning(f"No images found in the training directory: {TRAIN_DATA_PATH}. Mahalanobis distance for outlier detection is disabled.")
            return np.zeros((feature_size,)), np.eye(feature_size)

        features_list = []
        steps = len(train_set)

        with st.spinner("Computing training statistics for anomaly detection..."):
            for i in range(steps):
                x_batch, _ = train_set[i]
                feat_batch = extract_features(x_batch)
                features_list.append(feat_batch.numpy())

        features = np.vstack(features_list)

        mean_vector = np.mean(features, axis=0)
        cov_matrix = np.cov(features, rowvar=False)

        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-5  # Regularization

        try:
            inv_cov = inv(cov_matrix)
        except np.linalg.LinAlgError:
            st.error("Covariance matrix is singular. Mahalanobis distance will be inaccurate.")
            inv_cov = np.eye(cov_matrix.shape[0])

        return mean_vector, inv_cov

    except Exception as e:
        st.error(f"Error computing training statistics: {e}")
        return np.zeros((feature_size,)), np.eye(feature_size)

mean_vector, inv_cov = compute_training_stats()

# ------------------------------------
# 🔹 Prediction w/ anomaly detection
# ------------------------------------
def predict_xray_or_pneumonia(img_array):
    """
    Predicts the class and uses Mahalanobis distance for outlier detection.
    Returns: label_class, confidence_prob (0-1)
    label_class ∈ {"Pneumonia", "NORMAL", "Not an X-ray", "Error"}
    """
    try:
        feat = extract_features(img_array)[0].numpy()

        if np.allclose(mean_vector, 0) and np.allclose(inv_cov, np.eye(inv_cov.shape[0])):
            dist = 0
        else:
            dist = mahalanobis(feat, mean_vector, inv_cov)

        threshold = 10  # you can tune this

        if dist > threshold:
            return "Not an X-ray", 0.0
        else:
            pred_prob = seq_model.predict(img_array, verbose=0)[0][0]

            if pred_prob > 0.5:
                return "Pneumonia", float(pred_prob)
            else:
                return "NORMAL", float(1 - pred_prob)

    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0.0

# ------------------------------------
# 🔹 Grad-CAM
# ------------------------------------
def generate_gradcam(img_array, original_model, last_conv_layer_name):
    """Generates the Grad-CAM heatmap for model interpretability."""
    try:
        img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)

        conv_layer_idx = None
        for idx, layer in enumerate(original_model.layers):
            if layer.name == last_conv_layer_name:
                conv_layer_idx = idx
                break

        if conv_layer_idx is None:
            st.warning(f"Could not find layer {last_conv_layer_name}")
            return np.zeros((150, 150, 3), dtype=np.uint8)

        with tf.GradientTape() as tape:
            x = img_tensor
            for idx, layer in enumerate(original_model.layers):
                if idx <= conv_layer_idx:
                    x = layer(x, training=False)
                    if idx == conv_layer_idx:
                        conv_outputs = x
                        tape.watch(conv_outputs)
                else:
                    x = layer(x, training=False)

            predictions = x
            pred_value = predictions[0][0]

        grads = tape.gradient(pred_value, conv_outputs)

        if grads is None:
            st.warning("Could not compute gradients for Grad-CAM.")
            return np.zeros((150, 150, 3), dtype=np.uint8)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_outputs_np = conv_outputs[0].numpy()
        pooled_grads_np = pooled_grads.numpy()

        for i in range(len(pooled_grads_np)):
            conv_outputs_np[:, :, i] *= pooled_grads_np[i]

        heatmap = np.mean(conv_outputs_np, axis=-1)
        heatmap = np.maximum(heatmap, 0)

        max_val = np.max(heatmap)
        if max_val > 0:
            heatmap = heatmap / max_val

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.resize(heatmap, (150, 150))
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        return heatmap

    except Exception as e:
        st.warning(f"Error generating Grad-CAM: {e}")
        import traceback
        st.code(traceback.format_exc())
        return np.zeros((150, 150, 3), dtype=np.uint8)

# ------------------------------------
# 📤 Upload File
# ------------------------------------
st.markdown("<h2 style='font-size:28px;'>📤 Upload X-Ray Image</h2>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload a chest X-ray image (JPG or PNG)", type=["jpg", "jpeg", "png"])

# ------------------------------------
# 🧾 PDF Generator
# ------------------------------------
def generate_pdf(
    img,
    name,
    age,
    sex,
    fever,
    breathing_difficulty,
    cough,
    pneumonia_conf,
    normal_conf,
    final_risk,
    final_label,
    label_class_image_only,
    severity_image_only
):
    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, 760, "Pneumonia Detection Report")

    pdf.setFont("Helvetica", 13)
    y = 720
    pdf.drawString(50, y, f"Patient Name: {name if name else 'N/A'}")
    y -= 20
    pdf.drawString(50, y, f"Age: {age}")
    y -= 20
    pdf.drawString(50, y, f"Sex: {sex}")
    y -= 30

    # Symptoms
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Symptoms:")
    y -= 20

    pdf.setFont("Helvetica", 13)
    pdf.drawString(70, y, f"Fever: {'Yes' if fever else 'No'}")
    y -= 20
    pdf.drawString(70, y, f"Breathing Difficulty: {'Yes' if breathing_difficulty else 'No'}")
    y -= 20
    pdf.drawString(70, y, f"Cough: {'Yes' if cough else 'No'}")
    y -= 40

    # Results
    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Results:")
    y -= 20

    pdf.setFont("Helvetica", 13)
    pdf.drawString(70, y, f"Image-only Prediction: {label_class_image_only}")
    y -= 20
    if label_class_image_only == "Pneumonia":
        pdf.drawString(70, y, f"Image-only Severity: {severity_image_only}")
        y -= 20

    pdf.drawString(70, y, f"Pneumonia Confidence (image): {pneumonia_conf:.2f}%")
    y -= 20
    pdf.drawString(70, y, f"Normal Confidence (image): {normal_conf:.2f}%")
    y -= 20
    pdf.drawString(70, y, f"Final Risk Score (image + symptoms): {(final_risk * 100):.2f}%")
    y -= 20
    pdf.drawString(70, y, f"Final Assessment: {final_label}")
    y -= 40

    pdf.setFont("Helvetica-Bold", 14)
    pdf.drawString(50, y, "Chest X-Ray Image:")
    y -= 25

    img_buffer = io.BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    pil_img = Image.open(img_buffer)
    w, h = pil_img.size

    max_width = 5 * inch
    max_height = 5 * inch

    ratio = min(max_width / w, max_height / h)
    final_w = w * ratio
    final_h = h * ratio

    img_y = y - final_h
    if img_y < 40:
        y = 480
        img_y = y - final_h

    img_buffer.seek(0)
    img_reader = ImageReader(img_buffer)

    pdf.drawImage(img_reader, 50, img_y, width=final_w, height=final_h, preserveAspectRatio=True)

    pdf.save()
    buffer.seek(0)
    return buffer

# ------------------------------------
# 🔧 Main Logic
# ------------------------------------
if uploaded_file is not None:
    try:
        # Preprocess
        img = Image.open(uploaded_file).convert("L")
        img_resized = img.resize((150, 150))
        img_array = image.img_to_array(img_resized)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Show uploaded image
        st.image(img, caption="Uploaded Chest X-ray", use_container_width=True)

        # Prediction with anomaly detection
        label_class, confidence_prob = predict_xray_or_pneumonia(img_array)

        if label_class == "Error":
            st.error("An error occurred during prediction. Please try again.")
            st.stop()

        if label_class == "Not an X-ray":
            st.error("❌ The uploaded image does not appear to be a valid chest X-ray (Anomaly Detected).")
            st.caption("Please upload a proper chest X-ray image for analysis.")
            st.stop()

        # Image-only pneumonia vs normal probabilities
        if label_class == "Pneumonia":
            pneumonia_prob = confidence_prob
            normal_prob = 1 - confidence_prob
        else:  # "NORMAL"
            normal_prob = confidence_prob
            pneumonia_prob = 1 - confidence_prob

        pneumonia_conf = pneumonia_prob * 100
        normal_conf = normal_prob * 100

        # Severity based on image-only pneumonia confidence
        severity = get_severity(pneumonia_conf) if label_class == "Pneumonia" else "Low"

        # Grad-CAM
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

        if last_conv_layer_name is None:
            st.warning("Could not find a Conv2D layer for Grad-CAM generation.")
            superimposed_img = np.array(img_resized)
        else:
            heatmap = generate_gradcam(img_array, seq_model, last_conv_layer_name)
            original_img = cv2.cvtColor(np.array(img_resized), cv2.COLOR_GRAY2BGR)
            heatmap = cv2.resize(heatmap, (150, 150))
            superimposed_img = cv2.addWeighted(original_img, 0.4, heatmap, 0.6, 0)
            superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

        # Layout: Original vs Grad-CAM
        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Original X-ray (Preprocessed)", use_container_width=True)
        with col2:
            st.image(superimposed_img, caption="Grad-CAM Heatmap (Model Focus)", use_container_width=True)

        st.markdown("---")

        # Symptoms impact (like first app)
        symptom_score = (0.1 if fever else 0) + (0.15 if breathing_difficulty else 0) + (0.05 if cough else 0)
        final_risk = min(1.0, max(0.0, pneumonia_prob + symptom_score))
        final_label = "Pneumonia Detected" if final_risk > 0.5 else "Normal"

        # 🧪 Final Assessment Section (image + symptoms)
        st.markdown("<h2 style='font-size:28px;'>🧪 Final Assessment (Image + Symptoms)</h2>", unsafe_allow_html=True)
        color = "red" if final_label == "Pneumonia Detected" else "green"
        st.markdown(f"<h3 style='color:{color}; font-size:26px;'>{final_label}</h3>", unsafe_allow_html=True)
        st.markdown(
            f"<p style='font-size:20px;'><b>Final Risk Score:</b> {(final_risk * 100):.2f}%</p>",
            unsafe_allow_html=True,
        )

        # Image-only confidence section (for transparency)
        st.markdown("<h3 style='font-size:24px;'>Model Confidence (Image Only)</h3>", unsafe_allow_html=True)
        show_conf_bar_dual(pneumonia_conf, "Pneumonia", "red")
        show_conf_bar_dual(normal_conf, "Normal", "green")

        st.caption("⚠️ This tool is for educational purposes only and is not a medical diagnosis.")

        st.markdown("---")

        # Detailed prediction info (image-only)
        st.markdown("### ➡️ Image-only Prediction Summary")
        st.markdown(f"**Prediction (image):** :blue[{label_class}]")
        if label_class == "Pneumonia":
            st.markdown(f"**Severity (image):** :red[{severity}]")
        else:
            st.markdown(f"**Severity (image):** :green[Low]")
        st.write(f"**Confidence (image):** **{confidence_prob * 100:.2f}%**")
        show_conf_bar_single(confidence_prob * 100, label_class)

        st.markdown("---")

        # Guidance messages (based primarily on image prediction)
        if label_class == "Pneumonia":
            if confidence_prob * 100 > 80:
                st.error("🚨 **High Confidence:** Strong signs of pneumonia detected from the X-ray. **Seek medical attention immediately.**")
            elif confidence_prob * 100 > 50:
                st.warning("⚠️ **Moderate Confidence:** Possible signs of pneumonia. **Consult a doctor** for professional diagnosis.")
            else:
                st.info("ℹ️ **Low Confidence:** Model suggests possible pneumonia, but results are uncertain. Consider **further testing**.")

            with st.expander("🩺 More Information about Pneumonia"):
                st.write("""
                - **Pneumonia** is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus.
                - **Common Symptoms:** Cough (often with phlegm), fever, chills, and difficulty breathing.
                - **Disclaimer:** This tool is for informational purposes only and is **not a substitute for professional medical advice**.
                """)

            st.markdown("#### Suggested Next Steps:")
            st.write("""
            * Schedule a visit with your primary care physician or pulmonologist.
            * Share the image and this prediction with your healthcare provider.
            * Avoid self-medicating and follow professional medical guidance.
            """)
        else:
            st.success("✅ **No strong signs of pneumonia detected from the X-ray.**")
            st.markdown("#### Health Tips for Normal Results:")
            st.write("""
            * **Maintain lung health** by avoiding smoking and exposure to respiratory irritants.
            * **Stay up-to-date** on vaccinations (e.g., flu and pneumococcal vaccines).
            * **Practice good hygiene** to prevent respiratory infections.
            """)

        # ------------------------------------
        # 📄 PDF Download
        # ------------------------------------
        pdf_buffer = generate_pdf(
            img=img,
            name=name,
            age=age,
            sex=sex,
            fever=fever,
            breathing_difficulty=breathing_difficulty,
            cough=cough,
            pneumonia_conf=pneumonia_conf,
            normal_conf=normal_conf,
            final_risk=final_risk,
            final_label=final_label,
            label_class_image_only=label_class,
            severity_image_only=severity,
        )

        st.download_button(
            label="📄 Download Report (PDF)",
            data=pdf_buffer,
            file_name=f"{name if name else 'patient'}_pneumonia_report.pdf",
            mime="application/pdf",
        )

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        st.write("Please ensure you've uploaded a valid image file and try again.")
