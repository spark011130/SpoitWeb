from flask import Flask, render_template, request, redirect
from werkzeug.utils import secure_filename
import boto3
from PIL import Image


import logging
from botocore.exceptions import NoCredentialsError, ClientError

s3 = boto3.client('s3')
BUCKET_NAME = 'spoits3'
DIRECTORY_NAME = 'user/'
app = Flask(__name__)

def upload_to_s3(s3, file, bucket):
    try:
        s3.upload_fileobj(file, bucket, (DIRECTORY_NAME + file.filename))
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

        #객체를 삽입하기 이전 객체를 초기화하는 작업
        for elem in s3.list_objects_v2(Bucket = BUCKET_NAME)["Contents"]:
            if elem['Key'].startswith("user/"):
                s3.delete_object(Bucket=BUCKET_NAME, Key=elem['Key'])

        #객체 삽입 & temp 경로에 이미지 저장
        for file in files:
            print(file)
            if file.filename == '':
                return redirect(request.url)
            
            secure_filename_str = secure_filename(file.filename)

            save_path = f'SpoitWeb/static/temp/{secure_filename_str}'
            file.save(save_path)

            if not upload_to_s3(s3, file, BUCKET_NAME):
                return 'Failed to upload one or more files to AWS S3.'

        #업로드가 완료되었습니다 창과 함께 모든 사진들을 html에 갤러리 모드로 보여주기 및 아이콘 생성: 모든 영상 다운로드, 분석으로 가는 버튼도 지정
        return render_template('uploaded.html')

@app.route('/analysis/')
def anaylsis():
    return "뭐시기"

if __name__ == '__main__':
    app.run(debug=True)