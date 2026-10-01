import os
import time
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown
from mtcnn import MTCNN
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors

st.set_page_config(page_title="Down Syndrome AI Screening", layout="wide")

MODEL_PATH = "best_custom_cnn_v2.keras"
DRIVE_FILE_ID = "1u87-TfGbSPDrydMz6pyNdYMlNTCOkT81"

@st.cache_resource
def load_screening_models():
    # Download model from Google Drive automatically if not present
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights from Google Drive..."):
            url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
            
    detector = MTCNN()
    model = tf.keras.models.load_model(MODEL_PATH)
    return detector, model

detector, model = load_screening_models()

# Preprocessing: Geometric Eye Alignment + 10% Margin Crop + CLAHE Lighting Normalization
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
    c.drawString(30, height - 150, "Pipeline: CLAHE Normalization + MTCNN Geometric Eye Alignment")
    
    c.setFont("Helvetica-Bold", 12)
    c.drawString(30, height - 190, f"Classification Result: {class_result}")
    c.setFont("Helvetica", 11)
    c.drawString(30, height - 210, f"Down Syndrome Risk Score: {prob_down:.2f}%")
    c.drawString(30, height - 230, f"Neurotypical Probability: {prob_typical:.2f}%")
    
    if os.path.exists(cropped_img_path):
        c.drawImage(cropped_img_path, 30, height - 460, width=200, height=200)
        
    c.save()
    return report_filename

# UI Layout
st.title("🧬 Down Syndrome AI Screening & Biometric Diagnostic Platform")

patient_id = st.text_input("Patient ID", value="PATIENT-1001")
uploaded_file = st.file_uploader("Upload Frontal Photograph", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption="Uploaded Patient Image", use_container_width=True)

    if st.button("🚀 Run Analysis", type="primary"):
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

        with col2:
            st.subheader(f"Result: {outcome_str}")
            st.progress(prob_down / 100.0, text=f"Down Syndrome Risk: {prob_down:.2f}%")
            st.progress(prob_typical / 100.0, text=f"Neurotypical Profile: {prob_typical:.2f}%")
            st.image(cropped_face, caption="Processed Biometric Face Crop", width=200)

            temp_img_path = f"temp_{patient_id}.png"
            cv2.imwrite(temp_img_path, cv2.cvtColor(cropped_face, cv2.COLOR_RGB2BGR))
            pdf_path = generate_pdf_report(patient_id, outcome_str, prob_down, prob_typical, temp_img_path)

            with open(pdf_path, "rb") as f:
                st.download_button("📥 Download Clinical PDF Report", f, file_name=f"{patient_id}_Report.pdf", mime="application/pdf")
