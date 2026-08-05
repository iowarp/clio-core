#!/usr/bin/env python3
"""Create the YCSB 'usertable' on DynamoDB Local (issue #862).

DynamoDB Local accepts any credentials but the table must exist before the
YCSB dynamodb binding can load into it. boto3 is provisioned by the
devcontainer; primary key name matches the binding's dynamodb.primaryKey.
"""
import sys

import boto3

port = sys.argv[1] if len(sys.argv) > 1 else "8000"
ddb = boto3.client(
    "dynamodb",
    endpoint_url=f"http://127.0.0.1:{port}",
    region_name="us-east-1",
    aws_access_key_id="fake",
    aws_secret_access_key="fake",
)
existing = ddb.list_tables()["TableNames"]
if "usertable" in existing:
    print("usertable already exists")
    sys.exit(0)
ddb.create_table(
    TableName="usertable",
    KeySchema=[{"AttributeName": "firstname", "KeyType": "HASH"}],
    AttributeDefinitions=[{"AttributeName": "firstname", "AttributeType": "S"}],
    BillingMode="PAY_PER_REQUEST",
)
ddb.get_waiter("table_exists").wait(TableName="usertable")
print("usertable created")
