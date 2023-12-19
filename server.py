from flask import Flask, render_template, request, redirect
import boto3
import logging
from botocore.exceptions import NoCredentialsError, ClientError

s3 = boto3.client('s3')
BUCKET_NAME = 'spoits3'
app = Flask(__name__)

def upload_to_s3(s3, file, bucket):
    try:
        s3.upload_fileobj(file, bucket, file.filename)
        return True
    except NoCredentialsError:
        return False
    except ClientError as e:
        logging.error(e)
        return False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explanation/')
def explanation():
    return render_template('explanation.html')

@app.route('/upload/', methods = ['GET', 'POST'])
def upload():
    if request.method == 'GET':
        return render_template('upload.html')
    elif request.method == 'POST':
        files = request.files.getlist('files')
        for file in files:
            print(file)
            if file.filename == '':
                return redirect(request.url)

            if not upload_to_s3(s3, file, BUCKET_NAME):
                return 'Failed to upload one or more files to AWS S3.'
        return 'Files Uploaded!'

@app.route('/analysis/')
def anaylsis():
    return "뭐시기"

if __name__ == '__main__':
    app.run(debug=True)