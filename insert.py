import json
import numpy as np
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
    "host": "database-2.cpgwwc6uys5f.us-east-1.rds.amazonaws.com",
    "port": "3306",
    "user": "admin",
    "password": "acrroot987654321",
    "database": "user_information"
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

# In-memory storage for missing patient data
temporary_storage = {}


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

    if not valid_fields['patient_id']:
        valid_fields['patient_id'] = generate_patient_id(r"data\patientinfo.csv")

    missing_fields = [field for field, value in valid_fields.items() if value is None or value == '']

    if missing_fields:
        patient_id = valid_fields['patient_id']
        temporary_storage[patient_id] = valid_fields
        return {"error": "Missing fields", "missing_fields": missing_fields}

    return valid_fields


def clean_value(value):
    return re.sub(r'[^\d.]', '', value)


def format_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except ValueError:
        return date_str


def generate_patient_id(csv_file_path):
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
        if not df.empty:
            return df['patient_id'].max() + 1
    return 1


def add_patient_data_csv(patient_data, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return {"error": "CSV file does not exist."}

    df = pd.read_csv(csv_file_path)

    # Check if patient_id already exists to prevent duplicate entries
    if int(patient_data['patient_id']) in df['patient_id'].dropna().astype(int).values:
        return {"error": "Patient ID already exists in the CSV file."}

    # Convert patient_data to DataFrame and concatenate with the existing DataFrame
    new_row = pd.DataFrame([patient_data])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    return None


def update_patient_data_csv(patient_data, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return {"error": "CSV file does not exist."}

    df = pd.read_csv(csv_file_path)

    if 'patient_id' not in patient_data or not patient_data['patient_id']:
        return {"error": "Patient ID is required for updating records."}

    patient_id = patient_data['patient_id']
    if patient_id not in df['patient_id'].astype(str).values:
        return {"error": "Patient ID not found in CSV file."}

    for key, value in patient_data.items():
        if value is not None:
            df.loc[df['patient_id'] == patient_id, key] = value

    df.to_csv(csv_file_path, index=False)
    return None


def convert_values(values):
    return [int(value) if isinstance(value, np.int64) else value for value in values]

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
    values = convert_values(list(patient_data.values()))

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

    update_columns = ', '.join(f"`{column_name_mapping.get(key, key)}` = %s" for key in patient_data.keys())
    query = f"UPDATE patientinfo SET {update_columns} WHERE patient_id = %s"
    values = convert_values(list(patient_data.values()))
    values.append(patient_data['patient_id'])  # Append the patient_id for the WHERE clause

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

    update_columns = ', '.join(f"`{column_name_mapping.get(key, key)}` = %s" for key in patient_data.keys())
    query = f"UPDATE patientinfo SET {update_columns} WHERE patient_id = %s"
    values = list(patient_data.values())
    values.append(patient_data['patient_id'])  # Append the patient_id for the WHERE clause

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

    patient_data = extract_patient_info(prompt)

    if 'error' in patient_data and patient_data['error'] == "Missing fields":
        return jsonify(patient_data), 400

    csv_error = add_patient_data_csv(patient_data, r"data\patientinfo.csv")
    if csv_error:
        return jsonify(csv_error), 400

    db_error = add_patient_data_db(patient_data)
    if db_error:
        return jsonify(db_error), 400

    return jsonify({"success": "Patient data processed successfully."}), 200


@app.route('/save_patient', methods=['POST'])
def save_patient():
    data = request.json
    patient_id = data.get('patient_id')

    if not patient_id:
        return jsonify({"error": "Patient ID is required."}), 400

    patient_data = temporary_storage.get(patient_id)

    if not patient_data:
        return jsonify({"error": "No data found for the provided Patient ID."}), 404

    # Check if there are still missing fields
    missing_fields = [field for field in patient_data if not patient_data[field]]

    if missing_fields:
        return jsonify({"error": "Incomplete patient data", "missing_fields": missing_fields}), 400

    csv_error = add_patient_data_csv(patient_data, r"data\patientinfo.csv")
    if csv_error:
        return jsonify(csv_error), 400

    db_error = add_patient_data_db(patient_data)
    if db_error:
        return jsonify(db_error), 400

    # Remove from temporary storage
    temporary_storage.pop(patient_id, None)

    return jsonify({"success": "Patient data saved successfully."}), 200


if __name__ == '__main__':
    app.run(debug=True)