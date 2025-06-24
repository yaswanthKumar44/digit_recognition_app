from flask import Flask, request, jsonify, render_template_string
from keras.models import load_model
import numpy as np
import re
import base64
from PIL import Image
from io import BytesIO

app = Flask(__name__)

# Load trained model
model = load_model("model/digit_recognizer.h5")

# HTML template inline
HTML_PAGE = '''
<!doctype html>
<title>Handwritten Digit Recognition</title>
<h2 style="text-align:center">📝 Draw a Digit (0-9)</h2>
<div style="text-align:center">
<canvas id="canvas" width="280" height="280" style="border:1px solid #000;"></canvas><br><br>
<button onclick="clearCanvas()">Clear</button>
<button onclick="predict()">Predict</button>
<p id="result" style="font-size:20px"></p>
</div>
<script>
var canvas = document.getElementById('canvas');
var ctx = canvas.getContext('2d');
ctx.fillStyle = "black";
ctx.fillRect(0,0,canvas.width,canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 20;
ctx.lineCap = "round";
var drawing = false;
canvas.onmousedown = () => { drawing = true; ctx.beginPath(); };
canvas.onmouseup = () => { drawing = false; };
canvas.onmousemove = (e) => {
  if (drawing) {
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();
  }
};
function clearCanvas() {
  ctx.fillStyle = "black";
  ctx.fillRect(0,0,canvas.width,canvas.height);
  document.getElementById('result').innerText = "";
}
function predict() {
  var dataURL = canvas.toDataURL('image/png');
  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ image: dataURL })
  })
  .then(response => response.json())
  .then(data => {
    document.getElementById('result').innerText = "Prediction: " + data.prediction;
  });
}
</script>
'''

@app.route('/')
def home():
    return render_template_string(HTML_PAGE)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img = Image.open(BytesIO(base64.b64decode(img_data))).convert('L')  # to grayscale

    # Resize to 28x28 and normalize
    img = img.resize((28,28))
    img_array = np.array(img).astype('float32') / 255.0

    # Reshape to (1,28,28,1)
    img_array = img_array.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))

    return jsonify({'prediction': predicted_digit})

if __name__ == '__main__':
    app.run(debug=True)
