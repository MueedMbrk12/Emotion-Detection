import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import cv2
import numpy as np
from deepface import DeepFace
import base64
from PIL import Image
import io

# Initialize the Dash app
app = dash.Dash(__name__)

# Global variable for capturing frames from the webcam
cap = cv2.VideoCapture(0)  # 0 is the default camera

# Function to process the frame and detect emotion
def detect_emotion(frame):
    try:
        # Convert the frame from BGR (OpenCV default) to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Analyze the frame with DeepFace
        result = DeepFace.analyze(rgb_frame, actions=['emotion'], enforce_detection=False)
        return result['dominant_emotion']
    except Exception as e:
        return f"Error: {str(e)}"

# Function to capture a frame from the webcam and encode it as base64
def capture_frame():
    ret, frame = cap.read()  # Read a frame from the webcam
    if not ret:
        return None, None
    
    # Convert the frame to JPEG format and encode to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Detect emotion in the current frame
    emotion = detect_emotion(frame)

    return frame_base64, emotion

# Define the layout of the app
app.layout = html.Div([
    html.H1("Real-Time Emotion Detection from Webcam", style={'textAlign': 'center'}),
    
    # Image display area
    html.Div([
        html.Img(id='live-image', style={'width': '60%', 'display': 'block', 'margin': 'auto'}),
        html.H4(id='emotion-result', style={'textAlign': 'center', 'color': 'blue', 'margin': '20px'})
    ]),
    
    # Interval component for real-time updates
    dcc.Interval(
        id='interval-component',
        interval=1000,  # 1 second in milliseconds
        n_intervals=0  # Start at 0
    )
])

# Define the callback to update the image and emotion result every second
@app.callback(
    [Output('live-image', 'src'), Output('emotion-result', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_image(n_intervals):
    # Capture a frame from the webcam and analyze it
    frame_base64, emotion = capture_frame()

    if frame_base64 is not None:
        # Convert the base64-encoded frame to a format Dash can display
        frame_data = f"data:image/jpeg;base64,{frame_base64}"
        
        return frame_data, f"Detected Emotion: {emotion}"
    
    return None, "Unable to capture image"

# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)

# Release the webcam when the app is stopped
cap.release()
cv2.destroyAllWindows()
