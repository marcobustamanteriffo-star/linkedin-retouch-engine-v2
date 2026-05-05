from flask import Flask, request, send_file
import cv2
import numpy as np

app = Flask(__name__)

def cinematic_linkedin_system(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    l = cv2.convertScaleAbs(l, alpha=1.1, beta=-15)

    img_light = cv2.merge((l, a, b))
    img_bgr = cv2.cvtColor(img_light, cv2.COLOR_LAB2BGR)

    b_ch, g_ch, r_ch = cv2.split(img_bgr)
    b_ch = cv2.add(b_ch, 6)
    r_ch = cv2.subtract(r_ch, 4)

    img_cool = cv2.merge((b_ch, g_ch, r_ch))
    hsv = cv2.cvtColor(img_cool, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.9).astype(np.uint8)

    final_color = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    gaussian = cv2.GaussianBlur(final_color, (0, 0), 2.0)
    final_output = cv2.addWeighted(final_color, 1.5, gaussian, -0.5, 0)

    cv2.imwrite(output_path, final_output)

@app.route('/retouch', methods=['POST'])
def retouch():
    file = request.files['image']

    input_path = "input.jpg"
    output_path = "output.jpg"

    file.save(input_path)
    cinematic_linkedin_system(input_path, output_path)

    return send_file(output_path, mimetype='image/jpeg')

import os

port = int(os.environ.get("PORT", 10000))
app.run(host='0.0.0.0', port=port)
