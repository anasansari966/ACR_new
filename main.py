import csv
import json
import os
import pandas as pd
from dateutil import parser
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

app = Flask(__name__)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChatGPT model with ChatGPT 4.0
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""
    Extract the following information from the provided text and format it as a JSON dictionary:

    - Name
    - Gender
    - Date of Birth (DOB)
    - Height
    - Weight
    - Insurance and Policy Number
    - Medical Record Number
    - Hospital Record Number

    Example JSON format:
    {{
        "first name": "Patient First Name",
        "last name": "Patient Last Name",
        "gender": "Gender",
        "dob": "DOB",
        "height": Height,
        "weight": Weight,
        "insurance": "Insurance",
        "policy_number": "Policy Number",
        "medical_record_number": "Medical Record Number",
        "hospital_record_number": "Hospital Record Number"
    }}

    Text:
    {prompt}
    """
)

# Create a LangChain instance
chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_patient_info(prompt):
    extracted_info = chain.run(prompt=prompt)
    if not extracted_info:
        return {"error": "Model response is empty or invalid."}
    try:
        patient_data = json.loads(extracted_info)
        patient_data['dob'] = format_date(patient_data.get('dob', ''))
    except json.JSONDecodeError:
        return {"error": "Error decoding JSON from the model response."}
    return patient_data

def format_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        return date_str

def generate_patient_id(csv_file_path):
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
        if not df.empty:
            df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')
            df.dropna(subset=['patient_id'], inplace=True)
            if not df.empty:
                max_id = df['patient_id'].max()
                return int(max_id) + 1
    return 1

def validate_and_save_to_csv(patient_data, csv_file_path):
    patient_id = generate_patient_id(csv_file_path)
    patient_data['patient_id'] = patient_id

    patientinfo_fields = [
        'first name', 'last name', 'gender', 'dob', 'height', 'weight',
        'insurance', 'policy_number', 'medical_record_number', 'hospital_record_number'
    ]
    patientinfo_data = {field: patient_data.get(field, '') for field in patientinfo_fields}
    patientinfo_data['patient_id'] = patient_id
    patientinfo_data['dob'] = format_date(patientinfo_data.get('dob', ''))

    missing_fields = [field for field in patientinfo_fields if not patientinfo_data.get(field)]
    if not os.path.isfile(csv_file_path):
        df_patientinfo = pd.DataFrame(columns=patientinfo_fields + ['patient_id'])
    else:
        df_patientinfo = pd.read_csv(csv_file_path)

    new_patientinfo_df = pd.DataFrame([patientinfo_data])
    df_patientinfo = pd.concat([df_patientinfo, new_patientinfo_df], ignore_index=True)
    df_patientinfo.to_csv(csv_file_path, index=False)

    return missing_fields, patientinfo_data

def insert_into_db(patient_data):
    try:
        connection = mysql.connector.connect(
            host='database-2.cpgwwc6uys5f.us-east-1.rds.amazonaws.com',
            port='3306',
            user='admin',
            password='acrroot987654321',
            database='user_information'
        )
        if connection.is_connected():
            cursor = connection.cursor()
            check_query = "SELECT COUNT(*) FROM patientinfo WHERE patient_id = %s"
            cursor.execute(check_query, (patient_data['patient_id'],))
            record_exists = cursor.fetchone()[0]
            if record_exists:
                return "duplicate"
            insert_query = """
            INSERT INTO patientinfo (
                patient_id, first_name, last_name, gender, dob, height, weight, insurance, policy_number, medical_record_number, hospital_record_number
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                patient_data['patient_id'],
                patient_data.get('first name', ''),
                patient_data.get('last name', ''),
                patient_data.get('gender', ''),
                patient_data.get('dob', ''),
                patient_data.get('height', ''),
                patient_data.get('weight', ''),
                patient_data.get('insurance', ''),
                patient_data.get('policy_number', ''),
                patient_data.get('medical_record_number', ''),
                patient_data.get('hospital_record_number', '')
            )
            cursor.execute(insert_query, values)
            connection.commit()
            return "success"
    except Error as e:
        print(f"Error: {e}")
        return "error"
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

@app.route('/process_patient', methods=['POST'])
def process_patient():
    data = request.json
    prompt = data.get('prompt')
    csv_file_path = 'data/patientinfo.csv'  # Adjust path as needed

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    patient_data = extract_patient_info(prompt)
    if 'error' in patient_data:
        return jsonify({"error": patient_data['error']}), 400

    missing_fields, patient_data = validate_and_save_to_csv(patient_data, csv_file_path)

    if missing_fields:
        return jsonify({"patient_id": patient_data['patient_id'], "missing_fields": missing_fields, "data": patient_data}), 400

    save_status = insert_into_db(patient_data)
    if save_status == "success":
        return jsonify({"message": "Data saved successfully in the database.", "patient_id": patient_data['patient_id'], "data": patient_data}), 200
    elif save_status == "duplicate":
        return jsonify({"error": "Record already exists in the database."}), 400
    else:
        return jsonify({"error": "An error occurred while saving data in the database."}), 500

@app.route('/complete_missing_fields', methods=['POST'])
def complete_missing_fields():
    data = request.json
    patient_id = data.get('patient_id')
    csv_file_path = 'data/patientinfo.csv'  # Adjust path as needed

    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400

    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df_patient = df[df['patient_id'] == patient_id]
        if df_patient.empty:
            return jsonify({"error": "Patient ID not found in CSV"}), 400

        patient_data = df_patient.to_dict('records')[0]

        for key, value in data.items():
            if key in patient_data:
                patient_data[key] = value

        missing_fields = [field for field in df.columns if not patient_data.get(field)]

        if missing_fields:
            return jsonify({"patient_id": patient_id, "missing_fields": missing_fields, "data": patient_data}), 400

        df.update(pd.DataFrame([patient_data]))
        df.to_csv(csv_file_path, index=False)

        save_status = insert_into_db(patient_data)
        if save_status == "success":
            return jsonify({"message": "Missing data completed and saved successfully.", "patient_id": patient_id, "data": patient_data}), 200
        elif save_status == "duplicate":
            return jsonify({"error": "Record already exists in the database."}), 400
        else:
            return jsonify({"error": "An error occurred while saving data in the database."}), 500

    return jsonify({"error": "CSV file not found"}), 400

if __name__ == "__main__":
    app.run(debug=True)
