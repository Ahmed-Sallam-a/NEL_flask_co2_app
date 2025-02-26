from flask import Flask, request, jsonify, send_file, render_template
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import segmentation_models_pytorch as smp
import tifffile
from ultralytics import YOLO
import cv2
import io
from flask_cors import CORS
import zipfile
import matplotlib.pyplot as plt

app = Flask(__name__)
CORS(app)

########################################################I N D U S T R Y   C O D E #######################################################

# Define the transformation function for the input image
def transform_image(image):
    pad_transform = transforms.Compose([
        transforms.Pad((2, 2, 2, 2)),  # Pad by 2 pixels on each side (top, bottom, left, right)
        transforms.Normalize([0.5], [0.5])
    ])
    image_tensor = torch.tensor(image, dtype=torch.float32)
    image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
    image_tensor = pad_transform(image_tensor)
    print("Transformed image \n")
    return image_tensor

# Function to load the model
def load_industry_model():
    model = smp.DeepLabV3Plus(
        encoder_name="efficientnet-b1",
        encoder_weights="imagenet",
        in_channels=13,
        classes=1,
        activation="sigmoid"
    )
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()
    print("Model loaded \n")
    return model

# Function to make prediction
def predict_industry(model, image_tensor):
    with torch.no_grad():
        print("image_tensor shape = ", image_tensor.shape)
        output = model(image_tensor)  # Get the model output
        print("output shape = ", output.shape)
        prediction = output.squeeze().numpy()  # Remove batch dimension and convert to numpy
        print("Prediction made \n")
        print("prediction shape = ", prediction.shape)

    return prediction

# Load the model and make prediction
industry_model = load_industry_model()

######################################################F O R E S T   C O D E ########################################################

def load_forest_model():
    yolo_model = YOLO("best.pt")  # Load your trained YOLO model
    return yolo_model

def detect_objects(model, image):
    print("Detecting objects \n")
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    results = model(img)
    detections = results[0].boxes.data.cpu().numpy()
    print("Objects detected \n")
    print(detections)
    return detections, results[0].names


def draw_boxes(image, detections, names):

    img = np.array(image)
    for box in detections:
        x1, y1, x2, y2, confidence, class_id = box[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        label = f"{names[int(class_id)]}: {confidence:.1f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
        print("Drawing boxes \n")
        print(img)
    return img

forest_model = load_forest_model()

######################################################         A P I       ############################################################

@app.route('/')
def home():
    return render_template('index.html')  # Serve the HTML file

@app.route('/predict', methods=['POST'])
def predict():
    print("Request received \n")
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    model_type = request.form.get('model_type', 'Industry')
    print(f"Model type: {model_type}")

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        if model_type == "Industry":
            # Read .tif file
            image = tifffile.imread(file)
            height, width, channels = image.shape

            # Extract RGB bands
            band1, band2, band3 = image[:, :, 0], image[:, :, 1], image[:, :, 2]
            rgb_image = np.stack([band1, band2, band3], axis=-1)
            rgb_image = (rgb_image / np.max(rgb_image) * 255).astype(np.uint8)
            print("rgb_image shape: ", rgb_image.shape)
            # Convert to PIL format
            rgb_image_pil = Image.fromarray(rgb_image)
            image_tensor = transform_image(image)
            prediction = predict_industry(industry_model, image_tensor)

            # Ensure prediction is a NumPy array
            if isinstance(prediction, torch.Tensor):
                prediction = prediction.detach().cpu().numpy()

            print("Prediction image shape: ", prediction.shape)

            # Show the prediction image
            #cv2.imshow("Prediction", prediction)  # Ensure correct format
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

             # ✅ Ensure prediction is properly formatted before converting to PIL
            if np.issubdtype(prediction.dtype, np.floating):
                if prediction.max() > 1.0:
                    prediction = prediction / prediction.max()  # Normalize to [0,1]
                prediction = (prediction * 65535).clip(0, 65535).astype(np.uint16)  # Scale to 16-bit

            elif np.issubdtype(prediction.dtype, np.integer):
                # Ensure integer values are within the 16-bit range
                if prediction.max() > 65535:
                    raise ValueError("Integer prediction values exceed 16-bit range.")

            # ✅ Convert NumPy array to PIL Image with 16-bit depth
            pred_image_pil = Image.fromarray(prediction, mode='I;16')  # 16-bit grayscale image

            # Save images in a ZIP file
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zipf:
                with io.BytesIO() as img_byte:
                    rgb_image_pil.save(img_byte, format='PNG')  # Save original RGB image
                    zipf.writestr("original.png", img_byte.getvalue())

                with io.BytesIO() as pred_byte:
                    pred_image_pil.save(pred_byte, format='PNG')  # Save prediction in 16-bit
                    zipf.writestr("prediction.png", pred_byte.getvalue())

            zip_buffer.seek(0)

            print("Sending Industry original + prediction images")
            return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name="results.zip")

        elif model_type == "Forest":
            # Handle Forest model (unchanged)
            image = Image.open(file)
            detections, names = detect_objects(forest_model, image)
            image_with_boxes = draw_boxes(image, detections, names)

            # Convert image to bytes
            img_byte_arr = io.BytesIO()
            Image.fromarray(image_with_boxes).save(img_byte_arr, format='PNG')
            img_byte_arr.seek(0)
            print("Divided image successfully \n")
            return send_file(img_byte_arr, mimetype='image/png')

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)