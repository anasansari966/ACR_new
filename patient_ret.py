from flask import Flask, request, jsonify
import csv
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

app = Flask(__name__)

# Load the OpenAI API key from the environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model_name="gpt-4", temperature=0.0)

# Define the updated prompt template for querying patient information
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

query_chain = LLMChain(llm=llm, prompt=query_prompt_template)

# Load patient data from CSV file
def load_all_patient_data():
    patientinfo_file = r"D:\Jupiter\New_Model\data\patientinfo.csv"
    all_patient_data = []

    try:
        with open(patientinfo_file, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                all_patient_data.append(row)
    except FileNotFoundError:
        print(f"Error: {patientinfo_file} not found.")

    return all_patient_data

# Load patient data when the app starts
patient_data = load_all_patient_data()

def search_patient_data(prompt, patient_data):
    # Convert the patient_data list to a JSON string
    patient_data_json = json.dumps(patient_data, indent=4)

    # Generate the search results using ChatOpenAI
    response = query_chain.run(prompt=prompt, patient_data=patient_data_json)
    return response

@app.route('/')
def index():
    return "Welcome to the Patient Data Retrieval System"

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()  # Get JSON data from the request
    prompt = data.get('prompt')  # Extract the 'prompt' field from the JSON data

    if not patient_data:
        return jsonify({"error": "No patient data available."})

    search_results = search_patient_data(prompt, patient_data)
    return jsonify({"result": search_results})


if __name__ == "__main__":
    app.run(debug=True)
