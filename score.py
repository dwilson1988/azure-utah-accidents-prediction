# This script generates the scoring and schema files
# necessary to operationalize your model
from azureml.api.schema.dataTypes import DataTypes
from azureml.api.schema.sampleDefinition import SampleDefinition
from azureml.api.realtime.services import generate_schema
from azureml.assets import get_local_path

# Import frameworks
import pandas as pd
import xgboost
import arcgis

import numpy as np
import pickle
import json

# Prepare the web service definition by authoring
# init() and run() functions. Test the functions
# before deploying the web service.

model = None
wrangler = None

def init():
    """
    Initializes the model and any supporting data required.
    * Credentials
    * Road Static Features
    * Data Transfomations
    * XGBoost Model File
    :return: None
    """
    global model, wrangler


    # Load model.
    with open('wrangler.pkl', 'rb') as fp:
        wrangler = pickle.load(fp)
    model = xgboost.Booster(model_file='0001.xgbmodel')

def run(input_df):
    import json
    
    # Predict using appropriate functions
    # prediction = model.predict(input_df)

    prediction = "%s %d" % (str(input_df), model)
    return json.dumps(str(prediction))

def generate_api_schema():
    import os
    print("create schema")
    df = pd.read_csv("sample.csv")
    inputs = {"input_df": SampleDefinition(DataTypes.PANDAS, df)}
    os.makedirs('outputs', exist_ok=True)
    print(generate_schema(inputs=inputs, filepath="outputs/schema.json", run_func=run))

# Implement test code to run in IDE or Azure ML Workbench
if __name__ == '__main__':
    # Import the logger only for Workbench runs
    from azureml.logging import get_azureml_logger

    logger = get_azureml_logger()

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate', action='store_true', help='Generate Schema')
    args = parser.parse_args()

    if args.generate:
        generate_api_schema()

    init()
    input = "{}"
    result = run(input)
    logger.log("Result",result)
