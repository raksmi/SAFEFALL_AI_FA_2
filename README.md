AI-Powered Fall Detection and Monitoring System

SafeFall AI is an intelligent fall detection and monitoring application designed to analyse images and videos and identify potential falls. The application combines YOLOv8 Pose Estimation with an AI-based fall detection system to provide a visual and user-friendly monitoring experience.

Built using Python and Streamlit, SafeFall AI provides a simple interface for analysing uploaded media and displaying clear fall-detection results.

✨ Features
🖼️ Image Analysis – Upload an image for AI-based analysis.
🎥 Video Analysis – Analyse uploaded videos for potential falls.
🦴 YOLOv8 Pose Estimation – Detects human body keypoints and posture.
🚨 Fall Detection Alerts – Clearly identifies whether a fall was detected.
📸 Fall Frame Display – Displays the detected fall frame instead of showing every analysed frame.
📊 Monitoring Analytics – Tracks and visualises detection results.
🧑‍⚕️ User-Friendly Dashboard – Designed with a clean health-tech monitoring interface.
🧠 How It Works
For Images
The user uploads an image.
YOLOv8 Pose detects the person's body position and keypoints.
The fall detection model analyses the input.
The application displays the result and confidence.
For Videos
The user uploads a video and clicks Run Analysis.
Selected frames are sampled from the video.
YOLOv8 Pose analyses the person's posture in each sampled frame.
The system determines whether a potential fall is detected.
If a fall is found, the application displays the relevant fall frame and provides an alert.
If no fall is detected, the application clearly displays No Fall Detected.
🛠️ Technologies Used
Technology	Purpose
Python	Core programming language
Streamlit	Web application interface
TensorFlow / Keras	Machine learning model
YOLOv8 Pose	Human pose estimation
OpenCV	Image processing
NumPy	Numerical operations
Pandas	Data analysis
Matplotlib	Data visualisation
ImageIO / FFmpeg	Video processing
📁 Project Structure
safefall_ai_fa_2/
│
├── streamlit_app.py          # Main Streamlit application
├── fall_detection_model.h5   # Trained fall detection model
├── class_labels.json         # Activity class labels
├── requirements.txt          # Python dependencies
├── packages.txt              # System dependencies
├── runtime.txt               # Python runtime configuration
└── README.md                 # Project documentation
🚀 Running the Project Locally
1. Clone the repository
git clone <your-repository-url>
cd safefall_ai_fa_2
2. Install dependencies
pip install -r requirements.txt
3. Run the Streamlit application
streamlit run streamlit_app.py

The application should then open in your web browser.

☁️ Deployment

SafeFall AI is deployed using Streamlit Community Cloud.

The deployment requires:

Python 3.12
The dependencies listed in requirements.txt
Required system packages listed in packages.txt
The trained model file and class labels included in the repository
⚠️ Limitations

SafeFall AI is designed as a prototype and demonstration of AI-assisted fall detection. Its performance may be affected by:

Poor lighting conditions
Unusual camera angles
Multiple people in the frame
Occlusion of the person's body
Low-quality or blurry video
Postures that visually resemble a fall

Therefore, the system should not be considered a replacement for professional medical or emergency monitoring systems.

🔮 Future Improvements

Possible future developments include:

📱 Real-time camera monitoring
🔔 Automatic caregiver notifications
📍 Location-based emergency alerts
👥 Improved multi-person tracking
🧠 Training on a larger and more diverse dataset
☁️ Cloud-based event logging and monitoring history
👨‍💻 Project

SafeFall AI was developed as an Artificial Intelligence project exploring the use of computer vision and machine learning for fall detection and safety monitoring.
