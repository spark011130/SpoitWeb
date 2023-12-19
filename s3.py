import boto3
from botocore.exceptions import ClientError
import logging
import os

s3 = boto3.client("s3")
BUCKET_NAME = 'spoits3'

def list_all_buckets(s3):
    '''List all buckets list in the S3 server
    
    :param s3: s3 client
    '''
    try:
        buckets_resp = s3.list_buckets()
        for bucket in buckets_resp["Buckets"]:
            print(bucket)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def list_all_objects(s3, bucket_name):
    '''List all buckets list in the S3 server
    
    :param s3: s3 client
    :param bucket_name: 
    '''
    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        for obj in response["Contents"]:
            print(obj)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_file(file_name, bucket, object_name = None):
    '''Upload a file to a S3 bucket
    
    :param file_name: File path to upload
    :param bucket: Bucket to upload to 
    :param object_name: File name to be uploaded. If not specified, then file_name is used.
    '''

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_file(file_name, bucket, object_name = None, download_path = None):
    '''Download a file from S3 bucket
    
    :param file_name: File name to download
    :param bucket: Bucket to download to
    :param object_name: File name to be downloaded. If not specified, then file_name is used.
    :param download_path: File path to be downloaded. If not specified, then current working directory is used.
    '''
    if object_name is None:
        object_name = file_name

    if download_path is None:
        download_path = os.getcwd()
    
    s3_client = boto3.client('s3')
    try:
        response = s3_client.download_file(bucket, file_name, os.path.join(download_path, object_name))
    except ClientError as e:
        logging.error(e)
        return False
    return True

# list_all_buckets(s3)
# list_all_objects(s3, BUCKET_NAME)

query = input().rstrip()
if query == 'download':
    download_file('Pizza.jpeg', 'spoits3', 'Pizza2.jpeg')
if query == 'upload':
    upload_file('/Users/spoit/Desktop/SPOit Computer Vision Projects/SPOit/AWS/pizza.jpeg', 'spoits3', 'Pizza.jpeg')
