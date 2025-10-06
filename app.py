from flask import Flask, render_template, request
import sqlite3
import joblib
import pandas as pd
import spacy
import pytesseract
from PIL import Image
import os
import pytesseract

# Explicit path for Windows


app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load ML models and encoders
model = joblib.load("model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_symptom = joblib.load("le_symptom.pkl")
le_disease = joblib.load("le_disease.pkl")

# Load datasets
data = pd.read_csv("Medical_Advice.csv")
med_data = pd.read_csv("medicines.csv")

# Load NLP model
nlp = spacy.load("en_core_web_sm")

SYMPTOM_KEYWORDS = {
    "fever": "fever",
    "cough": "cold",
    "cold": "cold",
    "headache": "headache",
    "pain": "body_pain",
    "tired": "body_pain",
    "fatigue": "body_pain",
    "chest": "chest_pain",
    "bp": "hypertension",
    "sugar": "diabetes",
    "diabetes": "diabetes"
}

def extract_symptoms(text):
    doc = nlp(text.lower())
    found = []
    for token in doc:
        if token.lemma_ in SYMPTOM_KEYWORDS:
            mapped = SYMPTOM_KEYWORDS[token.lemma_]
            if mapped not in found:
                found.append(mapped)
    return found

def init_db():
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            age INTEGER,
            gender TEXT,
            bp REAL,
            temp REAL,
            symptom TEXT,
            disease TEXT,
            suggestion TEXT,
            tablet TEXT,
            symptom_desc TEXT
        )
    """)
    conn.commit()
    conn.close()

init_db()

def get_medicine_info(med_name):
    row = med_data[med_data["name"].str.lower() == med_name.lower()]
    if not row.empty:
        return {
            "name": row.iloc[0]["name"],
            "use": row.iloc[0]["use"],
            "side_effects": row.iloc[0]["side_effects"]
        }
    return None

# ---------------- Patient Symptom Prediction ---------------- #
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            name = request.form.get("name", "Anonymous")
            age = int(request.form["age"])
            gender = request.form["gender"].strip().upper()
            bp = float(request.form["bp"])
            temp = float(request.form["temp"])
            symptom = request.form.get("symptom", "").strip().lower()
            symptom_desc = request.form.get("symptom_desc", "").strip()

            extracted = extract_symptoms(symptom_desc) if symptom_desc else []
            symptoms_to_use = extracted if extracted else [symptom]

            results = []
            for sym in symptoms_to_use:
                if sym not in le_symptom.classes_:
                    continue
                gender_en = le_gender.transform([gender])[0]
                symptom_en = le_symptom.transform([sym])[0]
                input_data = [[age, gender_en, bp, temp, symptom_en]]
                pred = model.predict(input_data)[0]
                disease = le_disease.inverse_transform([pred])[0]
                row = data[data["target_disease"] == disease].iloc[0]
                results.append({
                    "Symptom": sym,
                    "Disease": disease,
                    "Suggestion": row["suggestion"],
                    "Tablet": row["tablet"]
                })
                conn = sqlite3.connect("patients.db")
                c = conn.cursor()
                c.execute("""
                    INSERT INTO records 
                    (name, age, gender, bp, temp, symptom, disease, suggestion, tablet, symptom_desc) 
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                """, (name, age, gender, bp, temp, sym, disease, row["suggestion"], row["tablet"], symptom_desc))
                conn.commit()
                conn.close()

            if not results:
                return "❌ No valid symptom found. Try again."
            return render_template("index.html", results=results)
        except Exception as e:
            return f"❌ Error: {e}"
    return render_template("index.html", results=None)

# ---------------- Medicine OCR Upload ---------------- #
@app.route("/upload_medicine", methods=["GET", "POST"])
def upload_medicine():
    result = None
    error_msg = None

    if request.method == "POST":
        file = request.files.get("medicine_image")
        if not file or file.filename == "":
            error_msg = "❌ No file uploaded."
        else:
            try:
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
                file.save(filepath)
                img = Image.open(filepath)
                text = pytesseract.image_to_string(img).strip().lower()

                if not text:
                    error_msg = "❌ Could not extract text from image."
                else:
                    med_row = med_data[med_data["name"].str.lower().str.contains(text)]
                    if not med_row.empty:
                        med_row = med_row.iloc[0]
                        result = {
                            "name": med_row["name"],
                            "use": med_row["use"],
                            "side_effects": med_row["side_effects"]
                        }
                    else:
                        error_msg = "❌ Medicine not found in database."

            except pytesseract.TesseractNotFoundError:
                error_msg = "❌ Tesseract is not installed or not found. Install it and ensure it's in your PATH."
            except Exception as e:
                error_msg = f"❌ Error: {e}"

    return render_template("upload_medicine.html", result=result, error_msg=error_msg)

# ---------------- View Patient Records ---------------- #
@app.route("/records")
def records():
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("SELECT * FROM records")
    rows = c.fetchall()
    conn.close()
    return render_template("records.html", rows=rows)

@app.route("/delete/<int:record_id>", methods=["POST"])
def delete_record(record_id):
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("DELETE FROM records WHERE id = ?", (record_id,))
    conn.commit()
    conn.close()
    return records()

@app.route("/delete_all", methods=["POST"])
def delete_all_records():
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("DELETE FROM records")
    conn.commit()
    conn.close()
    return records()

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


if __name__ == "__main__":
    app.run(debug=True)
