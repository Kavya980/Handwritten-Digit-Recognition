from flask import Flask, render_template, request, jsonify
from PIL import Image
import numpy as np
import joblib
import base64
import io


app = Flask(__name__)


# Load trained MNIST SVM model
model = joblib.load("digit_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recognize")
def recognize():
    return render_template("recognize.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        print("\nPrediction request received")


        # ==========================================
        # 1. Get image from browser
        # ==========================================

        data = request.json["image"]

        # Remove base64 header
        data = data.split(",")[1]

        # Decode image
        image_bytes = base64.b64decode(data)


        # ==========================================
        # 2. Open image and convert to grayscale
        # ==========================================

        image = Image.open(
            io.BytesIO(image_bytes)
        ).convert("L")


        # Convert to NumPy array
        image = np.array(image)


        # ==========================================
        # 3. Find the digit
        # ==========================================

        coordinates = np.argwhere(image > 30)


        # Empty canvas
        if coordinates.size == 0:

            return jsonify({
                "prediction": "Please draw a digit"
            })


        # Find digit boundaries
        y_min, x_min = coordinates.min(axis=0)
        y_max, x_max = coordinates.max(axis=0)


        # ==========================================
        # 4. Crop digit
        # ==========================================

        cropped = image[
            y_min:y_max + 1,
            x_min:x_max + 1
        ]


        # ==========================================
        # 5. Make digit square
        # ==========================================

        height, width = cropped.shape

        size = max(height, width)


        square = np.zeros(
            (size, size),
            dtype=np.uint8
        )


        # Center digit inside square
        y_offset = (size - height) // 2
        x_offset = (size - width) // 2


        square[
            y_offset:y_offset + height,
            x_offset:x_offset + width
        ] = cropped


        # ==========================================
        # 6. Resize digit to 20 × 20
        # ==========================================

        digit = Image.fromarray(square)


        digit = digit.resize(
            (20, 20),
            Image.Resampling.LANCZOS
        )


        digit = np.array(digit)


        # ==========================================
        # 7. Create 28 × 28 MNIST image
        # ==========================================

        mnist_image = np.zeros(
            (28, 28),
            dtype=np.float32
        )


        # Put digit in center
        mnist_image[
            4:24,
            4:24
        ] = digit


        # ==========================================
        # 8. Calculate center of mass
        # ==========================================

        total_pixels = mnist_image.sum()


        if total_pixels > 0:

            # Create coordinate grids
            y_grid, x_grid = np.indices(
                mnist_image.shape
            )


            # Calculate center of mass
            center_y = (
                (y_grid * mnist_image).sum()
                / total_pixels
            )

            center_x = (
                (x_grid * mnist_image).sum()
                / total_pixels
            )


            # Desired center of MNIST image
            target_y = 13.5
            target_x = 13.5


            # Calculate shift
            shift_y = int(
                round(target_y - center_y)
            )

            shift_x = int(
                round(target_x - center_x)
            )


            # Shift image
            shifted = np.zeros_like(
                mnist_image
            )


            source_y_start = max(
                0,
                -shift_y
            )

            source_y_end = min(
                28,
                28 - shift_y
            )

            source_x_start = max(
                0,
                -shift_x
            )

            source_x_end = min(
                28,
                28 - shift_x
            )


            target_y_start = max(
                0,
                shift_y
            )

            target_y_end = (
                target_y_start
                + (
                    source_y_end
                    - source_y_start
                )
            )


            target_x_start = max(
                0,
                shift_x
            )

            target_x_end = (
                target_x_start
                + (
                    source_x_end
                    - source_x_start
                )
            )


            shifted[
                target_y_start:target_y_end,
                target_x_start:target_x_end
            ] = mnist_image[
                source_y_start:source_y_end,
                source_x_start:source_x_end
            ]


            mnist_image = shifted


        # ==========================================
        # 9. Normalize pixels
        # ==========================================

        mnist_image = mnist_image / 255.0


        # ==========================================
        # 10. Flatten 28 × 28 → 784
        # ==========================================

        mnist_image = mnist_image.reshape(
            1,
            784
        )


        print(
            "Input shape:",
            mnist_image.shape
        )


        # ==========================================
        # 11. Predict
        # ==========================================

        prediction = model.predict(
            mnist_image
        )


        print(
            "Prediction:",
            prediction[0]
        )


        # ==========================================
        # 12. Send result to browser
        # ==========================================

        return jsonify({
            "prediction": int(prediction[0])
        })


    except Exception as e:

        print(
            "ERROR:",
            e
        )


        return jsonify({
            "prediction": "Error: " + str(e)
        })


if __name__ == "__main__":

    app.run(
        debug=True
    )