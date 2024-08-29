import json
import os
import pandas as pd
from dateutil import parser
from flask import Flask, request, jsonify, render_template
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
import logging
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

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

chain = LLMChain(llm=llm, prompt=prompt_template)


@app.route('/')
def chat():
    return render_template('chat.html')


@app.route('/process', methods=['POST'])
def process():
    user_input = request.json.get('prompt')
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
        return jsonify({"missing_fields": missing_fields, "preview": patient_data})

    # Proceed to validation and saving
    missing_fields, patient_data = validate_and_save(patient_data, csv_file_path)

    return jsonify({"preview": patient_data})


@app.route('/update', methods=['POST'])
def update():
    user_input = request.json.get('input')
    missing_field = request.json.get('field')
    patient_data = request.json.get('data')

    logging.debug(f"Updating field '{missing_field}' with value: {user_input}")

    # Update the patient_data with the new input
    patient_data[missing_field] = user_input

    # Check if there are still missing fields
    required_fields = [
        'first name', 'last name', 'gender', 'dob', 'height', 'weight',
        'insurance', 'policy_number', 'medical_record_number', 'hospital_record_number'
    ]
    missing_fields = [field for field in required_fields if not patient_data.get(field)]

    if missing_fields:
        logging.info(f"Missing fields after update: {missing_fields}")
        return jsonify({"missing_fields": missing_fields, "preview": patient_data})

    # Proceed to validation and saving
    csv_file_path = "data/patientinfo.csv"
    missing_fields, patient_data = validate_and_save(patient_data, csv_file_path)

    return jsonify({"preview": patient_data})


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


def validate_and_save(patient_data, csv_file_path):
    patientinfo_fields = [
        'first name', 'last name', 'gender', 'dob', 'height', 'weight',
        'insurance', 'policy_number', 'medical_record_number', 'hospital_record_number'
    ]

    # Ensure all fields are present
    patientinfo_data = {field: patient_data.get(field, '') for field in patientinfo_fields}

    # Generate patient ID
    patient_id = generate_patient_id(csv_file_path)
    patientinfo_data['patient_id'] = patient_id

    # Format date
    patientinfo_data['dob'] = format_date(patientinfo_data.get('dob', ''))

    # Load existing CSV or create a new DataFrame
    if os.path.isfile(csv_file_path):
        df_patientinfo = pd.read_csv(csv_file_path)
    else:
        df_patientinfo = pd.DataFrame(columns=patientinfo_fields + ['patient_id'])

    # Append the new patient data to the DataFrame
    new_patientinfo_df = pd.DataFrame([patientinfo_data])
    df_patientinfo = pd.concat([df_patientinfo, new_patientinfo_df], ignore_index=True)

    # Save DataFrame to CSV
    df_patientinfo.to_csv(csv_file_path, index=False)
    logging.info(f"Patient data saved to CSV: {patientinfo_data}")

    # Save to the database
    save_to_db(patientinfo_data)

    return [], patientinfo_data


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
        connection = mysql.connector.connect(
            host='database-2.cpgwwc6uys5f.us-east-1.rds.amazonaws.com',
            port='3306',
            user='admin',
            password='acrroot987654321',
            database='user_information'
        )

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

    except Error as e:
        logging.error(f"Database error: {e}")

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


if __name__ == '__main__':
    app.run(debug=True, port=8000, host="0.0.0.0")