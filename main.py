import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from flask import Flask, request, render_template, send_from_directory
import os
import webcolors
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/assets/img'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed extensions for images
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Create the 'uploads' folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Route for the home page (upload form)
@app.route('/')
def index():
    return render_template('index.html')  # Render HTML from templates folder


# Route to handle the image upload
@app.route('/upload', methods=['POST'])
def upload_and_process():
    # Check if a file is part of the request
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']

    # If the user does not select a file, the browser may submit an empty part without a filename
    if file.filename == '':
        return 'No selected file', 400

    # If the file is valid, save it
    if file and allowed_file(file.filename):

        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Check if it is a file (not a directory)
            if os.path.isfile(file_path):
                os.remove(file_path)  # Delete the file
                print(f"Deleted file: {file_path}")

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load image using OpenCV
        image = cv2.imread(file_path)

        # Convert the image from BGR (OpenCV default) to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Reshape the image to a 2D array of pixels (height * width, 3)
        pixels = image_rgb.reshape((-1, 3))

        # Define the number of dominant colors to extract (10 colors)
        n_colors = 10

        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_colors, random_state=100)
        kmeans.fit(pixels)

        # Get the RGB values of the cluster centers (dominant colors)
        dominant_colors = kmeans.cluster_centers_

        # Round the RGB values to integers
        dominant_colors = np.round(dominant_colors).astype(int)

        # Generate the color palette image
        plt.figure(figsize=(8, 4))
        plt.imshow([dominant_colors])
        plt.axis('off')  # Hide axes
        color_palette_path = os.path.join(app.config['UPLOAD_FOLDER'], 'color_palette.png')
        plt.savefig(color_palette_path)  # Save the color palette to a file
        plt.close()  # Close the plot to free up memory

        def rgb_to_color_name(rgb):
            try:
                # Convert the RGB tuple to a color name
                return webcolors.rgb_to_name(rgb)
            except ValueError:
                # If RGB value does not match a known color name, return a message
                return "No exact color name found"

        color_list = []
        # Print the RGB values of the dominant colors
        print("Top 10 dominant colors (RGB):")
        for color in dominant_colors:
            color_name = rgb_to_color_name(color)
            color_list.append(color_name)
        print(color_list)

        # Return the uploaded image URL and color palette URL to display them
        return render_template('index.html', filename=filename, color_palette=True, color_list=color_list)

    return 'File type not allowed', 400


# Route to serve the uploaded image
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == '__main__':
    app.run(debug=True)



