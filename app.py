import os
from flask import Flask, render_template, request, redirect, url_for, Response
from werkzeug.utils import secure_filename
import cv2
from ultralytics import solutions

import matplotlib.pyplot as plt
import numpy as np
from io import BytesIO
from flask import send_file

app = Flask(__name__)

# Configure upload folder and allowed extensions
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Ensure the uploads folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

grid_count =[]

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "file" not in request.files:
            return "No file part", 400
        file = request.files["file"]
        if file.filename == "":
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(filepath)

            # Redirect to the streaming page
            return redirect(url_for('stream',video_path=filename))

    return render_template("index.html")




def generate(video_path):

    global grid_count

    # Initialize YOLO heatmap
    heatmap = solutions.Heatmap(colormap=cv2.COLORMAP_PARULA, show=False, model="models/yolo11n.pt")

    # Grid configuration
    grid_rows, grid_cols = 5, 5

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    h, w, _ = frame.shape
    cell_height = h // grid_rows
    cell_width = w // grid_cols
    grid_count = [[0 for _ in range(grid_cols)] for _ in range(grid_rows)]

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            break

        im0 = heatmap.generate_heatmap(im0)
        results = heatmap.model(im0)
        boxes = results[0].boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf, cls = box.conf[0], box.cls[0]
            if cls == 0:
                grid_x = int(x1 // cell_width)
                grid_y = int(y1 // cell_height)
                grid_count[grid_y][grid_x] += 1

        for y in range(grid_rows):
            for x in range(grid_cols):
                cv2.rectangle(im0, (x * cell_width, y * cell_height),
                                ((x + 1) * cell_width, (y + 1) * cell_height), (255, 255, 255), 1)
                cv2.putText(im0, str(grid_count[y][x]),
                            (x * cell_width + 10, y * cell_height + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route('/stream/<video_path>')
def stream(video_path):
    video_path = os.path.join(app.config["UPLOAD_FOLDER"], video_path)

    return render_template('stream.html',video_path=video_path)


@app.route('/stream_video/<video_path>')
def stream_video(video_path):

    return Response(generate(video_path), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route('/grid_results')
def grid_results():
    global grid_count
    return render_template('grid_results.html', grid_count=grid_count, enumerate=enumerate)


@app.route('/grid_heatmap')
def grid_heatmap():
    global grid_count
    if not grid_count:
        return "No grid data available to generate heatmap.", 400

    # Convert grid_count to a numpy array for plotting
    grid_array = np.array(grid_count)

    # Create the heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(grid_array, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Count')
    plt.title('Heatmap of Grid Activity')
    plt.xlabel('Grid Columns')
    plt.ylabel('Grid Rows')
    plt.xticks(range(len(grid_count[0])))
    plt.yticks(range(len(grid_count)))

    # Save the plot to a BytesIO object
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()

    return send_file(buf, mimetype='image/png')


if __name__ == "__main__":
    app.run(debug=True)
