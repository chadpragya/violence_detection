import streamlit as st
import tempfile
import uuid
import cv2
import os
import torch
from ultralytics import YOLO

#Device Selection
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#Load Model
@st.cache_resource
def load_model():
    model = YOLO("best.pt")  # path to your fine-tuned model
    model.to(DEVICE)
    return model

model = load_model()

#Page Config
st.set_page_config(page_title="Violence Detection Streamlit App")
st.title("🔍 Violence Detection using YOLOv8")
st.write(f"Model is running on: **{DEVICE.upper()}**")

#Tabs for Video Upload and Webcam Detection
tab1, tab2 = st.tabs(["Video Upload", "Live Webcam"])

#VIDEO UPLOAD
with tab1:
    st.subheader("Upload a video")
    uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        st.video(video_path)

        if st.button("Run Detection on Uploaded Video"):
            st.info("Processing... Please wait")

            cap = cv2.VideoCapture(video_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            output_path = os.path.join(tempfile.gettempdir(), f"output_{uuid.uuid4().hex[:8]}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model.predict(frame, imgsz=640, conf=0.8, device=DEVICE, verbose=False)[0]
                annotated_frame = results.plot()
                out.write(annotated_frame)

            cap.release()
            out.release()

            st.success("Detection complete!")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name="annotated_output.mp4",
                    mime="video/mp4"
                )

# WEBCAM DETECTION
with tab2:
    st.subheader("Live Webcam Detection")
    run_webcam = st.button("Start Webcam")

    if run_webcam:
        stframe = st.empty()  # placeholder for displaying video frames

        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("Could not access webcam.")
        else:
            st.info("Press 'Stop' to end detection.")
            stop_button = st.button("Stop")

            while cap.isOpened() and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Failed to read frame from webcam.")
                    break

                results = model.predict(frame, imgsz=640, conf=0.8, device=DEVICE, verbose=False)[0]
                annotated_frame = results.plot()

                # Convert BGR to RGB for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

            cap.release()
            st.success("Webcam detection stopped.")