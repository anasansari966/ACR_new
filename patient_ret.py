import csv
import json
import os
import re
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables from .env file
load_dotenv()

# Get the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

app = Flask(__name__)

# Initialize the ChatGPT model with ChatGPT 4.0
llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the prompt template
prompt_template = PromptTemplate(
    input_variables=["patient_data"],
    template="""
    Convert the following patient data into a natural language description:

    Patient Data:
    {patient_data}

    Please provide a clear and conversational summary of the patient's information.
    """
)

# Create a LangChain instance
chain = LLMChain(llm=llm, prompt=prompt_template)

def generate_natural_language_description(patient_data):
    # Convert the patient_data dictionary to a JSON string
    patient_data_json = json.dumps(patient_data, indent=4)

    # Generate the natural language description using ChatOpenAI
    response = chain.run(patient_data=patient_data_json)
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

@app.route('/process_prompt', methods=['POST'])
def process_prompt():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided."}), 400

    patient_id = extract_patient_id_from_prompt(prompt)

    if patient_id is not None:
        patient_data = retrieve_patient_info(patient_id)

        if 'error' in patient_data:
            return jsonify({"error": patient_data['error']}), 404
        else:
            # Generate natural language description
            description = generate_natural_language_description(patient_data)
            return jsonify({"patient_data": patient_data, "description": description}), 200
    else:
        return jsonify({"error": "Could not extract a valid patient ID from the input."}), 400

if __name__ == "__main__":
    app.run(debug=True)
