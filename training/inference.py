#!/usr/local/bin/python3.7
import boto3
import io
import csv
from boto3.session import Session
from sagemaker.predictor import RealTimePredictor
from sagemaker.session import Session
from sagemaker import get_execution_role
import boto3
import json
import time

boto3_session = boto3.Session(profile_name='Test')

sagemaker_sess = Session(boto_session=boto3_session)

bt_endpoint = RealTimePredictor("blazingtext-2018-10-08-14-55-37-874",sagemaker_sess)

words = ["awesome"]

payload = {"instances" : words}

i = 0
while True:
    response = bt_endpoint.predict(json.dumps(payload))
    vecs = json.loads(response)
    i = i+1
    print("First vec value : " + str(vecs[0]['vector'][0]) + " & " + str(i))
    time.sleep(1)

