import json
import os
import re
import csv
import pandas as pd
from dateutil import parser
from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv
import logging

app = Flask(__name__)
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection configuration
dbconfig = {
    "host": "localhost",
    "port": "3306",
    "user": "root",
    "password": "anas123",
    "database": "acr1"
}

# Initialize connection pool
pool = pooling.MySQLConnectionPool(pool_name="mypool", pool_size=5, **dbconfig)

# Initialize the ChatGPT model with ChatGPT 4.0
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the prompt templates for extracting and querying data
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

query_prompt_template = PromptTemplate(
    input_variables=["prompt", "patient_data"],
    template="""
    Based on the following prompt: "{prompt}",
    search through the provided patient data and return any relevant information in natural language.

    Patient Data:
    {patient_data}

    Please provide a clear and concise response.
    """
)

chain = LLMChain(llm=llm, prompt=prompt_template)
query_chain = LLMChain(llm=llm, prompt=query_prompt_template)

def connect_to_db():
    return pool.get_connection()

def extract_patient_info(prompt):
    logging.debug(f"Extracting patient info from prompt: {prompt}")
    extracted_info = chain.run(prompt=prompt)

    if not extracted_info:
        logging.error("Model response is empty or invalid.")
        return {"error": "Model response is empty or invalid."}

    try:
        patient_data = json.loads(extracted_info)
        required_fields = [
            'first name', 'last name', 'gender', 'dob', 'height', 'weight',
            'insurance', 'policy_number', 'medical_record_number', 'hospital_record_number'
        ]
        patient_data['dob'] = format_date(patient_data.get('dob', ''))
    except json.JSONDecodeError:
        logging.error("Error decoding JSON from the model response.")
        return {"error": "Error decoding JSON from the model response."}

    return patient_data

def format_date(date_str):
    try:
        parsed_date = parser.parse(date_str)
        return parsed_date.strftime('%Y-%m-%d')
    except (ValueError, TypeError):
        logging.warning(f"Date parsing failed for: {date_str}")
        return date_str

def generate_patient_id(csv_file_path):
    if os.path.isfile(csv_file_path):
        df = pd.read_csv(csv_file_path)
        df['patient_id'] = pd.to_numeric(df['patient_id'], errors='coerce')
        df.dropna(subset=['patient_id'], inplace=True)
        if not df.empty:
            max_id = df['patient_id'].max()
            return int(max_id) + 1
    return 1

def save_to_db(patient_data):
    try:
        connection = connect_to_db()
        cursor = connection.cursor()

        insert_query = """
        INSERT INTO patientinfo (
            patient_id, first_name, last_name, gender, dob, height, weight,
            insurance, policy_number, medical_record_number, hospital_record_number
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        data = (
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

        cursor.execute(insert_query, data)
        connection.commit()
        logging.info(f"Patient data inserted into DB: {patient_data}")

    except mysql.connector.Error as e:
        logging.error(f"Database error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

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
        'age': 'age',
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
        db_field = column_name_mapping.get(field, field)
        if db_field != 'patient_id':
            update_fields.append(f"{db_field} = %s")
            values.append(value)

    update_fields_str = ", ".join(update_fields)
    query = f"UPDATE patientinfo SET {update_fields_str} WHERE patient_id = %s"
    values.append(patient_data.get('patient_id'))

    try:
        cursor.execute(query, values)
        conn.commit()
        logging.info("Patient data updated successfully in DB.")
    except mysql.connector.Error as err:
        logging.error(f"Error: {err}")
        conn.rollback()
    finally:
        cursor.close()
        conn.close()

def search_patient_data(prompt, patient_data):
    patient_data_json = json.dumps(patient_data, indent=4)
    response = query_chain.run(prompt=prompt, patient_data=patient_data_json)
    return response

def load_all_patient_data():
    patientinfo_file = r"data\patientinfo.csv"
    all_patient_data = []

    try:
        with open(patientinfo_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                all_patient_data.append(row)
    except FileNotFoundError:
        logging.error(f"Error: {patientinfo_file} not found.")

    return all_patient_data

# Load patient data when the app starts
patient_data = load_all_patient_data()

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/process', methods=['POST'])
def process():
    user_input = request.json.get('input')
    csv_file_path = "data/patientinfo.csv"

    logging.debug(f"Processing input: {user_input}")

    patient_data = extract_patient_info(user_input)

    if 'error' in patient_data:
        logging.error(f"Extraction error: {patient_data['error']}")
        return jsonify({"error": patient_data['error']})

    required_fields = [
        'first name', 'last name', 'gender', 'dob', 'height', 'weight',
        'insurance', 'policy_number', 'medical_record_number', 'hospital_record_number'
    ]

    missing_fields = [field for field in required_fields if not patient_data.get(field)]

    if missing_fields:
        logging.info(f"Missing fields: {missing_fields}")
        return jsonify({
            "status": "missing",
            "missing_fields": missing_fields,
            "patient_data": patient_data
        })

    patient_data['patient_id'] = generate_patient_id(csv_file_path)
    csv_error = update_patient_data_csv(patient_data, csv_file_path)

    if csv_error:
        return jsonify(csv_error)

    save_to_db(patient_data)
    return jsonify({
        "status": "success",
        "patient_data": patient_data
    })

@app.route('/update_missing_data', methods=['POST'])
def update_missing_data():
    data = request.json

    if not data:
        return jsonify({"error": "No data provided."})

    # Extract patient data and missing fields
    missing_fields = data.get('missing_fields', [])
    patient_data = data.get('patient_data', {})

    # Log the missing fields and the data received
    logging.debug(f"Received missing fields: {missing_fields}")
    logging.debug(f"Received patient data: {patient_data}")

    # Fill in the missing fields if they are not already provided in patient_data
    for field in missing_fields:
        if field not in patient_data:
            patient_data[field] = ''  # Set to empty string or a default value

    # Ensure patient ID is provided and valid
    patient_id = patient_data.get('patient_id')
    if not patient_id or not str(patient_id).isdigit():
        return jsonify({"error": "Invalid or missing patient ID."})

    patient_id = int(patient_id)
    csv_file_path = "data/patientinfo.csv"

    # Update the CSV file
    csv_error = update_patient_data_csv(patient_data, csv_file_path)
    if csv_error:
        return jsonify(csv_error)

    # Update the database
    update_patient_data_db(patient_data)

    return jsonify({
        "status": "success",
        "patient_data": patient_data
    })


@app.route('/update', methods=['POST'])
def update():
    user_input = request.json.get('input')
    csv_file_path = "data/patientinfo.csv"

    logging.debug(f"Updating data with input: {user_input}")

    patient_data = extract_patient_info(user_input)
    update_error = update_patient_data_csv(patient_data, csv_file_path)

    if update_error:
        logging.error(f"Update error: {update_error}")
        return jsonify(update_error)

    update_patient_data_db(patient_data)
    return jsonify({
        "status": "success",
        "patient_data": patient_data
    })

@app.route('/retrieve', methods=['GET'])
def retrieve():
    query = request.args.get('query')

    if not query:
        return jsonify({"error": "No query parameter provided."})

    response = search_patient_data(query, patient_data)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)
