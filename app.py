from flask import Flask, render_template, request
import sqlite3
import joblib
import pandas as pd

app = Flask(__name__)

# Load model + encoders
model = joblib.load("model.pkl")
le_gender = joblib.load("le_gender.pkl")
le_symptom = joblib.load("le_symptom.pkl")
le_disease = joblib.load("le_disease.pkl")

data = pd.read_csv("Medical_Advice.csv")

def init_db():
    conn = sqlite3.connect("patients.db")
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS records (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 age INTEGER, gender TEXT, bp REAL, temp REAL,
                 symptom TEXT, disease TEXT, suggestion TEXT, tablet TEXT)""")
    conn.commit()
    conn.close()

init_db()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            gender = request.form["gender"].strip().upper()   # normalize to M/F
            bp = float(request.form["bp"])
            temp = float(request.form["temp"])
            symptom = request.form["symptom"].strip().lower() # normalize to lowercase

            # Validate against encoder classes
            if gender not in le_gender.classes_:
                return f"❌ Error: Unknown gender '{gender}'. Valid: {list(le_gender.classes_)}"
            if symptom not in le_symptom.classes_:
                return f"❌ Error: Unknown symptom '{symptom}'. Valid: {list(le_symptom.classes_)}"

            gender_en = le_gender.transform([gender])[0]
            symptom_en = le_symptom.transform([symptom])[0]

            input_data = [[age, gender_en, bp, temp, symptom_en]]
            pred = model.predict(input_data)[0]
            disease = le_disease.inverse_transform([pred])[0]

            row = data[data["target_disease"] == disease].iloc[0]
            suggestion = row["suggestion"]
            tablet = row["tablet"]

            # Save record
            conn = sqlite3.connect("patients.db")
            c = conn.cursor()
            c.execute("""INSERT INTO records 
                         (age, gender, bp, temp, symptom, disease, suggestion, tablet) 
                         VALUES (?,?,?,?,?,?,?,?)""",
                      (age, gender, bp, temp, symptom, disease, suggestion, tablet))
            conn.commit()
            conn.close()

            return render_template("index.html", result={
                "Disease": disease,
                "Suggestion": suggestion,
                "Tablet": tablet
            })
        
        except Exception as e:
            return f"❌ Error: {e}"

    return render_template("index.html", result=None)

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
    c.execute("DELETE FROM records WHERE id=?", (record_id,))
    conn.commit()
    conn.close()
    return ("", 204)  

if __name__ == "__main__":
    app.run(debug=True)
