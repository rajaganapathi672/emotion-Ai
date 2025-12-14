from flask import Flask, render_template, Response
import cv2
from deepface import DeepFace
import numpy as np

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break
        
        try:
            # DeepFace analysis
            # enforce_detection=False prevents crash if no face found
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False, silent=True)
            
            # results is a list of dictionaries if multiple faces (or single dict in older versions, but current is list)
            if not isinstance(results, list):
                results = [results]

            for result in results:
                # Region
                region = result['region']
                x, y, w, h = region['x'], region['y'], region['w'], region['h']
                
                # Emotion
                dominant_emotion = result['dominant_emotion']
                confidence = result['emotion'][dominant_emotion]

                # Draw
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Label
                label = f"{dominant_emotion} ({confidence:.1f}%)"
                cv2.rectangle(frame, (x, y-25), (x+w, y), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, label, (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        except Exception as e:
            # DeepFace might throw errors or return empty
            pass

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    import webbrowser
    import threading

    def open_browser():
        webbrowser.open_new('http://127.0.0.1:5000/')

    print("--------------------------------------------------")
    print("Starting Web Application...")
    print("Please wait for the server to start.")
    print("If the browser does not open automatically, visit:")
    print("http://127.0.0.1:5000/")
    print("--------------------------------------------------")

    # Schedule browser to open after 1.5 seconds
    threading.Timer(1.5, open_browser).start()
    
    app.run(debug=True, use_reloader=False)
