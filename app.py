import json
import os
import re
import pandas as pd
import mysql.connector
from mysql.connector import pooling
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
from dateutil import parser
import numpy as np

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

# Initialize the ChatGPT model with ChatGPT 4.0
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the prompt template for extracting information
prompt_template = PromptTemplate(
    input_variables=["prompt"],
    template="""
    Extract the following information from the provided text and format it as a JSON dictionary:

    - Name (split into "first name" and "last name" if both are available)
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
        return {"error": "Error decoding JSON from the model response. Raw response: " + extracted_info}

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

    if 'patient_id' not in valid_fields:
        valid_fields['patient_id'] = generate_patient_id('data/patientinfo.csv')

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
            df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')
            max_id = df['patient_id'].max()
            if pd.isna(max_id):
                return 1
            return int(max_id) + 1
    return 1

def add_patient_data_csv(patient_data, csv_file_path):
    if not os.path.isfile(csv_file_path):
        return {"error": "CSV file does not exist."}

    df = pd.read_csv(csv_file_path)

    if int(patient_data['patient_id']) in df['patient_id'].dropna().astype(int).values:
        return {"error": "Patient ID already exists in the CSV file."}

    new_row = pd.DataFrame([patient_data])
    df = pd.concat([df, new_row], ignore_index=True)

    df.to_csv(csv_file_path, index=False)
    return None

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
def retrieve_patient_info(patient_id):
    # Specify the correct path to the CSV file
    csv_file_path = 'data/patientinfo.csv'

    # Check if the file exists
    if not os.path.isfile(csv_file_path):
        return {"error": "CSV file not found."}

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Print the contents of the DataFrame for debugging
    print("CSV DataFrame Contents:")
    print(df)

    # Ensure patient_id is compared as an integer
    try:
        patient_id = int(patient_id)
    except ValueError:
        return {"error": "Invalid patient ID format."}

    # Check the column names and data types
    print("Column Names:")
    print(df.columns)
    print("Data Types:")
    print(df.dtypes)

    # Convert patient_id column to integers for comparison
    df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')

    # Retrieve patient information based on patient_id
    patient_info = df[df['patient_id'] == patient_id].to_dict(orient='records')

    if not patient_info:
        return {"error": "Patient ID not found in CSV file."}

    return patient_info[0]


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

    columns = ', '.join(f"`{column_name_mapping.get(key, key)}`" for key in patient_data.keys())
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
    values.append(patient_data['patient_id'])

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

def generate_natural_language_description(patient_data):
    prompt = f"""
    Describe the following patient information in natural language:

    - Patient ID: {patient_data.get('patient_id')}
    - First Name: {patient_data.get('first name')}
    - Last Name: {patient_data.get('last name')}
    - Gender: {patient_data.get('gender')}
    - Date of Birth: {patient_data.get('dob')}
    - Height: {patient_data.get('height')}
    - Weight: {patient_data.get('weight')}
    - Insurance: {patient_data.get('insurance')}
    - Policy Number: {patient_data.get('policy_number')}
    - Medical Record Number: {patient_data.get('medical_record_number')}
    - Hospital Record Number: {patient_data.get('hospital_record_number')}

    Provide the description in a readable and informative way.
    """
    chain_description = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["prompt"], template=prompt))
    description = chain_description.run(prompt=prompt)
    return description

def determine_operation(prompt):
    # Expanded lists of keywords for different operations
    add_keywords = [
        'add', 'new', 'create', 'insert', 'include', 'register', 'enlist', 'submit',
        'add in', 'introduce', 'record', 'incorporate', 'append', 'attach', 'enter'
    ]
    update_keywords = [
        'update', 'modify', 'change', 'alter', 'revise', 'adjust', 'amend', 'refresh',
        'edit', 'correct', 'patch', 'rework', 'upgrade', 'replace', 'enhance', 'fix',
        'overhaul', 'redefine'
    ]
    retrieve_keywords = [
        'retrieve', 'get', 'fetch', 'show', 'display', 'list', 'present', 'access',
        'obtain', 'query', 'extract', 'report', 'uncover', 'reveal', 'bring up',
        'check', 'fetch details', 'summarize', 'provide'
    ]

    prompt_lower = prompt.lower()

    if any(keyword in prompt_lower for keyword in add_keywords):
        return 'add'
    elif any(keyword in prompt_lower for keyword in update_keywords):
        return 'update'
    elif any(keyword in prompt_lower for keyword in retrieve_keywords):
        return 'retrieve'
    else:
        return 'unknown'



@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    prompt = data.get('text')

    if not prompt:
        return jsonify({"error": "Text prompt is required."}), 400

    # Determine the operation type
    operation = determine_operation(prompt)

    if operation == 'retrieve':
        patient_id_match = re.search(r'patient id (\d+)', prompt, re.IGNORECASE)
        if patient_id_match:
            patient_id = patient_id_match.group(1)
            patient_info = retrieve_patient_info(patient_id)
            if 'error' in patient_info:
                return jsonify(patient_info), 400

            description = generate_natural_language_description(patient_info)
            return jsonify({"description": description}), 200
        else:
            return jsonify({"error": "Patient ID not found in the prompt."}), 400

    # Extract patient information from the prompt
    extracted_info = extract_patient_info(prompt)

    if 'error' in extracted_info:
        return jsonify(extracted_info), 400

    patient_data = extracted_info

    if operation == 'add':
        csv_error = add_patient_data_csv(patient_data, 'data/patientinfo.csv')
        if csv_error:
            return jsonify(csv_error), 400

        db_error = add_patient_data_db(patient_data)
        if db_error:
            return jsonify(db_error), 400

        return jsonify({"message": "Patient data added successfully."}), 200

    elif operation == 'update':
        csv_error = update_patient_data_csv(patient_data, 'data/patientinfo.csv')
        if csv_error:
            return jsonify(csv_error), 400

        db_error = update_patient_data_db(patient_data)
        if db_error:
            return jsonify(db_error), 400

        return jsonify({"message": "Patient data updated successfully."}), 200

    else:
        return jsonify({"error": "Invalid operation. Use 'add', 'update', or 'retrieve'."}), 400



if __name__ == '__main__':
    app.run(debug=True)
