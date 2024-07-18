import os
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Path to the model and scaler files
model_dir = os.path.join(os.path.dirname(__file__), 'model')

# Load model and scaler for Random Forest (model 5)
rf_model_path = os.path.join(model_dir, 'diabetes_model_5.pkl')
rf_scaler_path = os.path.join(model_dir, 'scaler5.pkl')

try:
    rf_model = joblib.load(rf_model_path)
    rf_scaler = joblib.load(rf_scaler_path)
    print("Random Forest model dan scaler berhasil dimuat.")
except FileNotFoundError:
    print("File model atau scaler Random Forest tidak ditemukan.")
    rf_model = None
    rf_scaler = None
except Exception as e:
    print("Error saat memuat Random Forest model atau scaler:", e)
    rf_model = None
    rf_scaler = None

# Load model and scaler for Naive Bayes (model 2)
nb_model_path = os.path.join(model_dir, 'diabetes_model_2.pkl')
nb_scaler_path = os.path.join(model_dir, 'scaler.pkl')

try:
    nb_model = joblib.load(nb_model_path)
    nb_scaler = joblib.load(nb_scaler_path)
    print("Naive Bayes model dan scaler berhasil dimuat.")
except FileNotFoundError:
    print("File model atau scaler Naive Bayes tidak ditemukan.")
    nb_model = None
    nb_scaler = None
except Exception as e:
    print("Error saat memuat Naive Bayes model atau scaler:", e)
    nb_model = None
    nb_scaler = None

# Endpoint untuk index API
@app.route("/")
def index():
    return "<p>API Cek diabetes!</p>"

# Function to handle prediction
def make_prediction(model, scaler, data):
    try:
        # Mengambil dan memvalidasi data input
        pregnancies = float(data['Pregnancies'])
        glucose = float(data['Glucose'])
        blood_pressure = float(data['BloodPressure'])
        skin_thickness = float(data['SkinThickness'])
        insulin = float(data['Insulin'])
        bmi = float(data['BMI'])
        diabetes_pedigree_function = float(data['DiabetesPedigreeFunction'])
        age = float(data['Age'])

        input_data = [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
        
        # Transformasi input data dengan scaler
        std_data = scaler.transform(input_data)
        print(f"Data setelah diskalakan: {std_data}")

        prediction = model.predict(std_data)
        print(f"Prediksi: {prediction}")

        if prediction[0] == 0:
            diagnosis = 'Pasien Tidak Terkena Diabetes'
        elif prediction[0] == 1:
            diagnosis = 'Pasien Terkena Diabetes'
        else:
            diagnosis = 'Diagnosis Tidak Diketahui'

        return {'prediction': int(prediction[0]), 'diagnosis': diagnosis}

    except KeyError as e:
        print(f"KeyError: {e}")
        return {'error': f'Field {e} tidak ditemukan'}
    except ValueError as e:
        print(f"ValueError: {e}")
        return {'error': 'Harap masukkan nilai numerik yang valid untuk semua input.'}
    except Exception as e:
        print(f"Exception: {e}")
        return {'error': str(e)}

# Endpoint untuk prediksi diabetes dengan model Random Forest
@app.route("/diabetes-cek-rf", methods=["POST"])
def diabetes_cek_rf():
    if rf_model is None or rf_scaler is None:
        return jsonify({'error': 'Random Forest model atau scaler tidak tersedia'}), 500

    data = request.get_json()
    print("Data diterima:", data)

    result = make_prediction(rf_model, rf_scaler, data)
    if 'error' in result:
        return jsonify(result), 400 if 'Field' in result['error'] else 500
    return jsonify(result)

# Endpoint untuk prediksi diabetes dengan model Naive Bayes
@app.route("/diabetes-cek-nb", methods=["POST"])
def diabetes_cek_nb():
    if nb_model is None or nb_scaler is None:
        return jsonify({'error': 'Naive Bayes model atau scaler tidak tersedia'}), 500

    data = request.get_json()
    print("Data diterima:", data)

    result = make_prediction(nb_model, nb_scaler, data)
    if 'error' in result:
        return jsonify(result), 400 if 'Field' in result['error'] else 500
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
