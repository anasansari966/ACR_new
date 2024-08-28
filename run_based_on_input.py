import json
import os
import re
import pandas as pd
from dateutil import parser
import mysql.connector
from mysql.connector import pooling
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)

# Database connection configuration
dbconfig = {
    "host":"database-2.cpgwwc6uys5f.us-east-1.rds.amazonaws.com",
    "port":"3306",
    "user":"admin",
    "password":"acrroot987654321",
    "database":"user_information"
}

# Initialize connection pool
pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **dbconfig)

def connect_to_db():
    return pool.get_connection()

# Ensure you have set your OpenAI API key in the environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize the ChatGPT model with ChatGPT 4.0
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the prompt template for extracting information
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""
    Extract the following information from the provided text and format it as a JSON dictionary:

    - Name (split into "first name" and "last name" if both are available)
    - Age
    - Gender
    - Date of Birth (DOB)
    - Height
    - Weight
    - Insurance and Policy Number
    - Medical Record Number
    - Hospital Record Number
    - Patient ID

    The output should be a JSON dictionary with the following structure:
    {{
        "patient_id": "Patient ID",
        "first name": "Patient First Name",
        "last name": "Patient Last Name",
        "age": "Age",
        "gender": "Gender",
        "dob": "DOB",
        "height": "Height",
        "weight": "Weight",
        "insurance": "Insurance",
        "policy_number": "Policy Number",
        "medical_record_number": "Medical Record Number",
        "hospital_record_number": "Hospital Record Number"
    }}

    Only include fields that are provided in the text.

    Text:
    {prompt}
    """
)

# Create a LangChain instance
chain = LLMChain(llm=llm, prompt=prompt_template)

def extract_patient_info(prompt):
    extracted_info = chain.run(prompt=prompt)

    if isinstance(extracted_info, dict):
        return extracted_info

    if not extracted_info:
        return {"error": "Model response is empty or invalid."}

    try:
        patient_data = json.loads(extracted_info)
    except json.JSONDecodeError:
        return {"error": "Error decoding JSON from the model response."}

    valid_fields = {
        'patient_id': patient_data.get('patient_id'),
        'first name': patient_data.get('first name'),
        'last name': patient_data.get('last name'),
        'gender': patient_data.get('gender'),
        'dob': format_date(patient_data.get('dob', '')),
        'height': clean_value(patient_data.get('height', '')),
        'weight': clean_value(patient_data.get('weight', '')),
        'insurance': patient_data.get('insurance'),
        'policy_number': patient_data.get('policy_number'),
        'medical_record_number': patient_data.get('medical_record_number'),
        'hospital_record_number': patient_data.get('hospital_record_number')
    }

    valid_fields = {k: v for k, v in valid_fields.items() if v is not None and v != ''}

    return valid_fields


def clean_value(value):
    return re.sub(r'[^\d.]', '', value)

def format_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        return date_str

def update_patient_data_csv(patient_data, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return {"error": "CSV file does not exist."}

    df = pd.read_csv(csv_file_path)
    patient_id = patient_data.get('patient_id')

    if patient_id is None or not str(patient_id).isdigit():
        return {"error": "Invalid or missing patient ID."}

    patient_id = int(patient_id)
    if patient_id not in df['patient_id'].values:
        return {"error": f"Patient ID {patient_id} not found in the CSV file."}

    for field, value in patient_data.items():
        if field in df.columns and field != 'patient_id':
            df.loc[df['patient_id'] == patient_id, field] = value

    df.to_csv(csv_file_path, index=False)
    return None

def update_patient_data_db(patient_data):
    conn = connect_to_db()
    cursor = conn.cursor()

    column_name_mapping = {
        'first name': 'first_name',
        'last name': 'last_name',
        'gender': 'gender',
        'dob': 'dob',
        'height': 'height',
        'weight': 'weight',
        'insurance': 'insurance',
        'policy_number': 'policy_number',
        'medical_record_number': 'medical_record_number',
        'hospital_record_number': 'hospital_record_number'
    }

    update_fields = []
    values = []
    for field, value in patient_data.items():
        db_field = column_name_mapping.get(field)
        if db_field and db_field != 'patient_id':
            update_fields.append(f"`{db_field}` = %s")  # Use backticks for column names
            values.append(value)

    if not update_fields:
        return {"error": "No fields to update."}

    update_fields_str = ", ".join(update_fields)
    query = f"UPDATE patientinfo SET {update_fields_str} WHERE `patient_id` = %s"
    values.append(patient_data.get('patient_id'))

    try:
        cursor.execute(query, values)
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()
        return {"error": str(err)}
    finally:
        cursor.close()
        conn.close()
    return None

def add_patient_data_csv(patient_data, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return {"error": "CSV file does not exist."}

    df = pd.read_csv(csv_file_path)

    # Check if patient_id exists to prevent duplicate entries
    if 'patient_id' in patient_data and str(patient_data['patient_id']).isdigit():
        if int(patient_data['patient_id']) in df['patient_id'].values:
            return {"error": "Patient ID already exists in the CSV file."}

    # Convert patient_data to DataFrame and concatenate with the existing DataFrame
    new_row = pd.DataFrame([patient_data])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    return None


def add_patient_data_db(patient_data):
    conn = connect_to_db()
    cursor = conn.cursor()

    column_name_mapping = {
        'first name': 'first_name',
        'last name': 'last_name',
        'gender': 'gender',
        'dob': 'dob',
        'height': 'height',
        'weight': 'weight',
        'insurance': 'insurance',
        'policy_number': 'policy_number',
        'medical_record_number': 'medical_record_number',
        'hospital_record_number': 'hospital_record_number'
    }

    columns = ', '.join(f"`{column_name_mapping.get(key, key)}`" for key in patient_data.keys())  # Quote column names
    placeholders = ', '.join(['%s'] * len(patient_data))
    query = f"INSERT INTO patientinfo ({columns}) VALUES ({placeholders})"
    values = list(patient_data.values())

    try:
        cursor.execute(query, values)
        conn.commit()
    except mysql.connector.Error as err:
        conn.rollback()
        return {"error": str(err)}
    finally:
        cursor.close()
        conn.close()
    return None

@app.route('/process_patient', methods=['POST'])
def process_patient():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    csv_file_path = r"data\patientinfo.csv"  # Path to your CSV file

    if should_update_data(prompt):
        patient_data = extract_patient_info(prompt)

        if 'error' in patient_data:
            return jsonify({"error": patient_data['error']}), 400

        error_message = update_patient_data_csv(patient_data, csv_file_path)
        if error_message:
            return jsonify({"error": error_message['error']}), 400
        else:
            db_error = update_patient_data_db(patient_data)
            if db_error:
                return jsonify({"error": db_error['error']}), 400

        return jsonify({"message": "Patient data updated successfully.", "data": patient_data}), 200

    else:
        patient_data = extract_patient_info(prompt)

        if 'error' in patient_data:
            return jsonify({"error": patient_data['error']}), 400

        error_message = add_patient_data_csv(patient_data, csv_file_path)
        if error_message:
            return jsonify({"error": error_message['error']}), 400
        else:
            db_error = add_patient_data_db(patient_data)
            if db_error:
                return jsonify({"error": db_error['error']}), 400

        return jsonify({"message": "New patient added successfully.", "data": patient_data}), 200

def should_update_data(prompt):
    update_keywords = ['update', 'change', 'modify', 'adjust', 'alter']
    return any(keyword in prompt.lower() for keyword in update_keywords)

if __name__ == "__main__":
    app.run(debug=True)