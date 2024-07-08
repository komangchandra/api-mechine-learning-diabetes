import os
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Tambahkan ini untuk mengaktifkan CORS di seluruh aplikasi

# Path to the model file
model_path = os.path.join(os.path.dirname(__file__), 'model', 'diabetes_model.sav')
print(f"Path model: {model_path}")

# Load model dari file pickle
try:
    with open(model_path, 'rb') as model_file:
        diabetes_model = pickle.load(model_file)
    print("Model berhasil dimuat.")
except FileNotFoundError:
    print("File diabetes_model.sav tidak ditemukan.")
    diabetes_model = None
except Exception as e:
    print("Error saat memuat model:", e)
    diabetes_model = None

# Endpoint untuk index API
@app.route("/")
def index():
    return "<p>API Cek diabetes!</p>"

# Endpoint untuk prediksi diabetes dengan metode POST
@app.route("/diabetes-cek", methods=["POST"])
def diabetes_cek():
    if diabetes_model is None:
        return jsonify({'error': 'Model tidak tersedia'}), 500

    # Mendapatkan data dari body request
    data = request.get_json()
    print("Data diterima:", data)  # Logging data yang diterima

    try:
        # Mendapatkan nilai dari atribut yang diperlukan untuk prediksi
        pregnancies = float(data['Pregnancies'])
        glucose = float(data['Glucose'])
        blood_pressure = float(data['BloodPressure'])
        skin_thickness = float(data['SkinThickness'])
        insulin = float(data['Insulin'])
        bmi = float(data['BMI'])
        diabetes_pedigree_function = float(data['DiabetesPedigreeFunction'])
        age = float(data['Age'])

        # Membuat array numpy untuk input model
        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]

        # Melakukan prediksi dengan model
        prediction = diabetes_model.predict(input_data)

        # Menentukan diagnosis berdasarkan prediksi
        if prediction[0] == 0:
            diagnosis = 'Pasien Tidak Terkena Diabetes'
        else:
            diagnosis = 'Pasien Terkena Diabetes'

        # Mengembalikan hasil prediksi dalam format JSON
        return jsonify({'prediction': int(prediction[0]), 'diagnosis': diagnosis})

    except KeyError as e:
        return jsonify({'error': 'Field ' + str(e) + ' tidak ditemukan'}), 400
    except ValueError:
        return jsonify({'error': 'Harap masukkan nilai numerik yang valid untuk semua input.'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
