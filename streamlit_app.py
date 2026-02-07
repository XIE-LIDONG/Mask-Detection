import os
import subprocess
import time
import webbrowser

def install_ultralytics():
    print(" Installing YOLO dependency: ultralytics...")
    install_cmd = "pip install ultralytics opencv-python streamlit numpy --upgrade -i https://pypi.tuna.tsinghua.edu.cn/simple"
    try:
        result = subprocess.run(
            install_cmd, shell=True, check=True, capture_output=True, text=True
        )
        print(" ultralytics and dependencies installed successfully!")
    except Exception as e:
        print(f" Installation warning (can ignore): {e}")
        # Fallback: Conda installation for conda environments
        try:
            subprocess.run("conda install -c conda-forge ultralytics -y", shell=True, check=True)
            print(" ultralytics installed successfully via conda!")
        except:
            raise Exception(" Installation failed! Please run manually: pip install ultralytics")

# Run installation 
install_ultralytics()

PROJECT_PATH = r"C:\Users\lenovo\jupyter"
os.chdir(PROJECT_PATH)


streamlit_code = '''
import streamlit as st
import cv2
import numpy as np
import os  # Added missing os import
try:
    from ultralytics import YOLO
except ImportError:
    st.error(" ultralytics not installed! Please run: pip install ultralytics")
    st.stop()

# Page configuration
st.set_page_config(page_title="YOLO Mask Detection", layout="wide")
st.title("😷 YOLO Real-Time Mask Detection ")
st.markdown("---")

@st.cache_resource
def load_yolo_model():
    # Replace with your downloaded model filename (e.g., yolo11m.pt/Mask_detector.pt)
    model_name = "Mask_detector.pt"
    model_path = os.path.join(os.getcwd(), model_name)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f" Model file not found: {model_path}")
        st.error(f"Files in current directory: {os.listdir(os.getcwd())}")
        st.stop()
    
    # Load the model
    try:
        model = YOLO(model_path)
        st.success(f" Model loaded successfully: {model_name}")
        return model
    except Exception as e:
        st.error(f" Model loading failed: {e}")
        st.stop()

model = load_yolo_model()

# Sidebar configuration
st.sidebar.header("🔧 Detection Settings")
conf = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.05)

# Session state management
if "run" not in st.session_state:
    st.session_state.run = False

# Control buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("📹 Start YOLO Detection", type="primary", use_container_width=True):
        st.session_state.run = True
with col2:
    if st.button("🛑 Stop Detection", use_container_width=True):
        st.session_state.run = False

# Video stream display placeholder
frame_placeholder = st.empty()

if st.session_state.run:
    # Open camera (DirectShow for Windows compatibility)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera read failed (may be in use)")
            break
        
        # Mirror flip the frame
        frame = cv2.flip(frame, 1)
        
        # YOLO inference
        results = model(frame, conf=conf, verbose=False, device="cpu")
        
        # Auto-draw detection boxes
        frame_with_boxes = results[0].plot()
        
        # Convert color space for Streamlit display
        frame_rgb = cv2.cvtColor(frame_with_boxes, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
    
    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
    frame_placeholder.empty()
    st.info("🛑 Detection stopped")

'''

with open("yolo_streamlit.py", "w", encoding="utf-8") as f:
    f.write(streamlit_code)
print("✅ Generated final YOLO file: yolo_streamlit.py")

# Step 6: Launch Streamlit (port 8502 to avoid conflicts)
start_cmd = f"streamlit run yolo_streamlit.py --server.port 8502"
streamlit_process = subprocess.Popen(
    start_cmd, shell=True, cwd=PROJECT_PATH
)

# Wait for startup and open browser
time.sleep(4)  # Extra second after installation for model loading
webbrowser.open_new("http://localhost:8502")

# Final status message
print("="*60)
print("YOLO11 Mask Detection System Started Successfully!")
print("🌐 Access the app via your browser (default: http://localhost:8502)")
print("💡 To stop the app: press Ctrl+C in the terminal")
print("="*60)


globals()['streamlit_process'] = streamlit_process
