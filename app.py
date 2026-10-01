import os
import time
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import streamlit as st
import gdown
from mtcnn import MTCNN
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

st.set_page_config(
    page_title="Down Syndrome AI Screening Platform",
    page_icon="🧬",
    layout="wide"
)

MODEL_PATH = "best_custom_cnn_v2.keras"
DRIVE_FILE_ID = "1u87-TfGbSPDrydMz6pyNdYMlNTCOkT81"

@st.cache_resource
def load_screening_models():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading clinical model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    detector = MTCNN()
    model = tf.keras.models.load_model(MODEL_PATH)
    return detector, model

detector, model = load_screening_models()

# Grad-CAM Heatmap Generator
def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None):
    if last_conv_layer_name is None:
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break

    grad_model = tf.keras.models.Model(
        inputs=[model.inputs],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def generate_gradcam_overlay(img_rgb, heatmap, alpha=0.4):
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    jet = cv2.cvtColor(jet, cv2.COLOR_BGR2RGB)
    jet = cv2.resize(jet, (img_rgb.shape[1], img_rgb.shape[0]))
    
    superimposed_img = jet * alpha + img_rgb * (1 - alpha)
    return np.uint8(superimposed_img)

# Pipeline: MTCNN Eye Alignment + 10% Margin Crop + CLAHE Normalization
def align_and_preprocess_face(img_rgb, face_data):
    keypoints = face_data['keypoints']
    left_eye, right_eye = keypoints['left_eye'], keypoints['right_eye']
    dY, dX = right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))
    
    eye_center = (int((left_eye[0] + right_eye[0]) // 2), int((left_eye[1] + right_eye[1]) // 2))
    h, w, _ = img_rgb.shape
    M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
    aligned_img = cv2.warpAffine(img_rgb, M, (w, h), flags=cv2.INTER_CUBIC)

    x, y, box_w, box_h = face_data['box']
    margin = int(0.10 * max(box_w, box_h))
    x1, y1 = max(0, x - margin), max(0, y - margin)
    x2, y2 = min(w, x + box_w + margin), min(h, y + box_h + margin)
    cropped = aligned_img[y1:y2, x1:x2]

    lab = cv2.cvtColor(cropped, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l_channel)
    eq_rgb = cv2.cvtColor(cv2.merge((l_eq, a_channel, b_channel)), cv2.COLOR_LAB2RGB)

    face_resized = cv2.resize(eq_rgb, (224, 224))
    return cropped, np.expand_dims(face_resized.astype(np.float32), axis=0)

def generate_pdf_report(patient_id, class_result, prob_down, prob_typical, cropped_img_path):
    report_filename = f"{patient_id}_Clinical_Report.pdf"
    c = canvas.Canvas(report_filename, pagesize=letter)
    width, height = letter
    
    c.setFillColor(colors.HexColor("#1e293b"))
    c.rect(0, height - 80, width, 80, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(30, height - 45, "Non-Invasive Down Syndrome AI Screening Report")
    
    c.setFillColor(colors.black)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 110, f"Patient ID: {patient_id}")
    c.setFont("Helvetica", 10)
    c.drawString(30, height - 130, f"Date/Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawString(30, height - 150, "Pipeline: CLAHE Normalization + MTCNN Alignment + Grad-CAM")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 190, f"Classification Result: {class_result}")
    c.setFont("Helvetica", 11)
    c.drawString(30, height - 210, f"Down Syndrome Risk Score: {prob_down:.2f}%")
    c.drawString(30, height - 230, f"Neurotypical Probability: {prob_typical:.2f}%")
    
    if os.path.exists(cropped_img_path):
        c.drawImage(cropped_img_path, 30, height - 460, width=200, height=200)
        
    c.save()
    return report_filename

# SIDEBAR MENU
st.sidebar.title("🧬 Navigation")
mode = st.sidebar.radio("Select Workflow Mode:", ["Single Patient Screening", "Batch Processing Pipeline"])

# MODE 1: SINGLE PATIENT SCREENING
if mode == "Single Patient Screening":
    st.title("🧬 Single Patient Biometric Diagnostic Screening")

    patient_id = st.text_input("Patient ID", value="PATIENT-1001")
    uploaded_file = st.file_uploader("Upload Frontal Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, 1)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_rgb, caption="Uploaded Image", use_container_width=True)

        if st.button("🚀 Run Diagnostic Analysis", type="primary"):
            faces = detector.detect_faces(img_rgb)
            if len(faces) > 0:
                cropped_face, img_tensor = align_and_preprocess_face(img_rgb, faces[0])
            else:
                cropped_face = img_rgb
                img_tensor = np.expand_dims(cv2.resize(img_rgb, (224, 224)).astype(np.float32), axis=0)

            raw_pred = float(model.predict(img_tensor, verbose=0)[0][0])
            prob_typical = max(0.0, min(100.0, raw_pred * 100.0))
            prob_down = max(0.0, min(100.0, 100.0 - prob_typical))

            outcome_str = "Down Syndrome Phenotype" if prob_down >= 50.0 else "Neurotypical Profile"

            try:
                heatmap = make_gradcam_heatmap(img_tensor, model)
                gradcam_overlay = generate_gradcam_overlay(cv2.resize(cropped_face, (224, 224)), heatmap)
            except Exception:
                gradcam_overlay = cropped_face

            with col2:
                st.subheader(f"Result: {outcome_str}")
                
                m1, m2 = st.columns(2)
                m1.metric("Down Syndrome Risk", f"{prob_down:.2f}%")
                m2.metric("Neurotypical Probability", f"{prob_typical:.2f}%")

                st.progress(prob_down / 100.0)

                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(cropped_face, caption="Biometric Face Crop", use_container_width=True)
                with img_col2:
                    st.image(gradcam_overlay, caption="Grad-CAM Biomarker Heatmap", use_container_width=True)

                temp_img_path = f"temp_{patient_id}.png"
                cv2.imwrite(temp_img_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
                pdf_path = generate_pdf_report(patient_id, outcome_str, prob_down, prob_typical, temp_img_path)

                with open(pdf_path, "rb") as f:
                    st.download_button("📥 Download PDF Clinical Report", f, file_name=f"{patient_id}_Report.pdf", mime="application/pdf")

# MODE 2: BATCH PROCESSING PIPELINE
elif mode == "Batch Processing Pipeline":
    st.title("📦 Multi-Patient Batch Screening Pipeline")
    st.write("Upload multiple patient facial photographs simultaneously for automated batch screening and CSV summary export.")

    uploaded_files = st.file_uploader("Upload Multiple Patient Images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        if st.button(f"⚡ Process Batch ({len(uploaded_files)} Images)", type="primary"):
            results_list = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for idx, file in enumerate(uploaded_files):
                status_text.text(f"Processing image {idx+1}/{len(uploaded_files)}: {file.name}")
                
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img_bgr = cv2.imdecode(file_bytes, 1)
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                faces = detector.detect_faces(img_rgb)
                if len(faces) > 0:
                    cropped_face, img_tensor = align_and_preprocess_face(img_rgb, faces[0])
                else:
                    img_tensor = np.expand_dims(cv2.resize(img_rgb, (224, 224)).astype(np.float32), axis=0)

                raw_pred = float(model.predict(img_tensor, verbose=0)[0][0])
                prob_typical = max(0.0, min(100.0, raw_pred * 100.0))
                prob_down = max(0.0, min(100.0, 100.0 - prob_typical))
                outcome = "Down Syndrome Phenotype" if prob_down >= 50.0 else "Neurotypical Profile"

                results_list.append({
                    "Filename": file.name,
                    "Prediction": outcome,
                    "Down Syndrome Risk (%)": round(prob_down, 2),
                    "Neurotypical Prob (%)": round(prob_typical, 2)
                })

                progress_bar.progress((idx + 1) / len(uploaded_files))

            status_text.text("✅ Batch Analysis Completed!")
            
            df_results = pd.DataFrame(results_list)
            st.subheader("📊 Batch Diagnostic Results Summary")
            st.dataframe(df_results, use_container_width=True)

            csv_data = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Batch Results CSV",
                data=csv_data,
                file_name="batch_screening_results.csv",
                mime="text/csv"
            )
