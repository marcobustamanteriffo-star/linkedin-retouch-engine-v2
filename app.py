from flask import Flask, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)

# --- 1. Motor de Procesamiento de Imagen ---
def cinematic_linkedin_system(input_path, output_path):
    img = cv2.imread(input_path)
    if img is None:
        return

    # Ajuste de iluminación (LAB)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    l = clahe.apply(l)
    l = cv2.convertScaleAbs(l, alpha=1.1, beta=-15)

    img_light = cv2.merge((l, a, b))
    img_bgr = cv2.cvtColor(img_light, cv2.COLOR_LAB2BGR)

    # Balance de color frío
    b_ch, g_ch, r_ch = cv2.split(img_bgr)
    b_ch = cv2.add(b_ch, 6)
    r_ch = cv2.subtract(r_ch, 4)

    img_cool = cv2.merge((b_ch, g_ch, r_ch))
    hsv = cv2.cvtColor(img_cool, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 0.9).astype(np.uint8)

    final_color = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    # Filtro de nitidez/suavizado
    gaussian = cv2.GaussianBlur(final_color, (0, 0), 2.0)
    final_output = cv2.addWeighted(final_color, 1.5, gaussian, -0.5, 0)

    cv2.imwrite(output_path, final_output)

# --- 2. Ruta Única de Retoque ---
@app.route('/retouch', methods=['POST'])
def retouch():
    if 'image' not in request.files:
        return "Error: No se envió ninguna imagen bajo la clave 'image'", 400
    
    file = request.files['image']
    if file.filename == '':
        return "Error: El archivo está vacío", 400

    input_path = "input.jpg"
    output_path = "output.jpg"
    
    try:
        file.save(input_path)
        cinematic_linkedin_system(input_path, output_path)
        return send_file(output_path, mimetype='image/jpeg')
    except Exception as e:
        return f"Error en el procesamiento: {str(e)}", 500

# --- 3. Ejecución del Servidor ---
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)
