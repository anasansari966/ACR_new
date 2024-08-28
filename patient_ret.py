import csv
import json
import os
import re
import datetime
from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

# Initialize Flask app
app = Flask(__name__)

# Set OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize ChatOpenAI model
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the prompt template for generating natural language description
prompt_template = PromptTemplate(
    input_variables=["patient_data"],
    template="""
    Convert the following patient data into a natural language description:

    Patient Data:
    {patient_data}

    Please provide a clear and conversational summary of the patient's information.
    """
)

# Create a RunnableSequence instance
sequence = RunnableSequence([prompt_template, llm])


def convert_dates_to_strings(obj):
    if isinstance(obj, dict):
        return {k: convert_dates_to_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dates_to_strings(i) for i in obj]
    elif isinstance(obj, datetime.date):
        return obj.isoformat()  # Convert date to ISO format string
    else:
        return obj


def generate_natural_language_description(patient_data):
    # Convert date objects in patient_data to strings
    patient_data = convert_dates_to_strings(patient_data)

    # Convert the patient_data dictionary to a JSON string
    patient_data_json = json.dumps(patient_data, indent=4)

    # Generate the natural language description using ChatOpenAI
    response = sequence.run(patient_data=patient_data_json)
    return response


def retrieve_patient_info(patient_id):
    patientinfo_file = r"D:\Jupiter\New_Model\data\patientinfo.csv"

    patient_info = None

    try:
        with open(patientinfo_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('patient_id') == str(patient_id):
                    patient_info = row
                    break
            if not patient_info:
                print(f"Debug: patient_id '{patient_id}' not found in {patientinfo_file}")
    except FileNotFoundError:
        print(f"Error: {patientinfo_file} not found.")

    if patient_info:
        return patient_info

    return {"error": "Patient ID not found in the Database."}


def extract_patient_id_from_prompt(prompt):
    # Use regex to find the patient ID in the prompt
    match = re.search(r'patient\s+id\s+(\d+)', prompt, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def is_retrieve_operation(prompt):
    # Define keywords that indicate a retrieve operation
    keywords = ['show', 'give', 'all']
    return any(keyword in prompt.lower() for keyword in keywords)


@app.route('/process_text', methods=['POST'])
def process_text():
    data = request.json
    prompt = data.get('text', '')

    if not prompt:
        return jsonify({"error": "Text prompt is required."}), 400

    if is_retrieve_operation(prompt):
        patient_id = extract_patient_id_from_prompt(prompt)

        if patient_id is not None:
            patient_data = retrieve_patient_info(patient_id)

            if 'error' in patient_data:
                return jsonify(patient_data), 404
            else:
                description = generate_natural_language_description(patient_data)
                return jsonify({
                    "patient_data": patient_data,
                    "description": description
                }), 200
        else:
            return jsonify({"error": "Could not extract a valid patient ID from the input."}), 400
    else:
        return jsonify({"error": "Operation not recognized. Use 'show', 'give', or 'all' to retrieve data."}), 400


if __name__ == "__main__":
    app.run(debug=True)
