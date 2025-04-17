# Pandas, Machine Learning starter files

Pandas, Machine Learning starter files

### MLAS - Rekog
- Starter Code
    ```python
    # pip install boto3
    # export aws_id=AKIA___________PJPOY
    # export aws_secret_access_key=VaLL______________________________cvF9qC

    import os
    import boto3
    from urllib.request import urlopen

    awid = os.environ['aws_id']
    awkey = os.environ['aws_secret_access_key']

    session = boto3.Session(
    aws_access_key_id=awid,
    aws_secret_access_key=awkey
    )

    urltopen = f"https://raw.githubusercontent.com/ckibank/web/main/dogbag/0{ctr}.jpg"
    ```

### ML
- ML bootstrap code
    ```python

    import pandas as pd
    # import sklearn
    from sklearn.naive_bayes import GaussianNB    
    # Dictionary for Non ML prediction

    InvFeatureDict = {
    "1111" : "Invest ESG",
    "1110" : "Monitor ESG",
    "0001" : "Prof Non ESG",
    "0010" : "No Prof No ESG",
    "0011" : "Prof Non ESG",
    "0101" : "Prof Non ESG",
    "0111" : "Prof Non ESG",
    "1001" : "Prof Non ESG",
    "1011" : "Prof Non ESG",
    "1101" : "Prof Non ESG",
    "0000" : "No Prof No ESG",
    "0100" : "No Prof No ESG",
    "0110" : "No Prof No ESG",
    "1000" : "No Prof No ESG",
    "1100" : "No Prof No ESG",
    "0000" : "No Prof No ESG",
    "1010" : "No Prof No ESG"
    }

    # Dictionary for Prediction answer
    InvType = {
    "1": "Invest_ESG",
    "2": "Monitor_ESG",
    "3": "Prof_Non_ESG",
    "4": "No_Prof_No_ESG"
    }

    ```
