from flask import Flask, render_template, request, redirect # 웹 프레임워크 관련 모듈
from werkzeug.utils import secure_filename #링크 보안 관련 모듈
import boto3 # 서버 관련 모듈
import zipfile # 압축 관련 모듈
import os # 경로 관련 모듈
import shutil # 디렉토리의 내용 삭제 관련 모듈
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

def urlGenerate(s3, bucket_name, object_key):
    try:
        presigned_url = s3.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_key},
            ExpiresIn = 3600
        )
        return presigned_url
    except NoCredentialsError:
        print('Credentials not available')
    except Exception as e:
        print(f'Error: {e}')

def zip_images(folder_path, zip_filename):
    # 폴더 내의 모든 파일 목록을 가져옵니다.
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # Zip 파일을 생성하고 이미지 파일들을 압축합니다.
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in file_list:
            # 이미지 파일인 경우에만 압축합니다. 여기서는 확장자가 .jpg, .png, .gif로 가정합니다.
            if file.lower().endswith(('.jpg', '.png', '.gif')):
                file_path = os.path.join(folder_path, file)
                zipf.write(file_path, os.path.basename(file_path))


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

        # 디렉토리를 초개화하는 작업 (user login 시스템을 갖출 경우, 각 user의 초기화가 필요)
        directory_path = 'SpoitWeb/static/temp'
        try:
            os.makedirs(directory_path)
        except FileExistsError:
            # 디렉토리가 이미 존재하는 경우, 덮어쓰기
            shutil.rmtree(directory_path)
            os.makedirs(directory_path)

        #객체를 삽입하기 이전 객체를 초기화하는 작업
        if s3.list_objects_v2(Bucket = BUCKET_NAME)['KeyCount'] > 0:
            for elem in s3.list_objects_v2(Bucket = BUCKET_NAME)["Contents"]:
                if elem['Key'].startswith("user/"):
                    s3.delete_object(Bucket=BUCKET_NAME, Key=elem['Key'])

        # 객체 삽입 & temp 경로에 이미지 저장
        for file in files:
            print(file)
            if file.filename == '':
                return redirect(request.url)
            
            secure_filename_str = secure_filename(file.filename)

            save_path = f'SpoitWeb/static/temp/{secure_filename_str}'
            file.save(save_path)

            if not upload_to_s3(s3, file, BUCKET_NAME):
                return 'Failed to upload one or more files to AWS S3.'
        
        # zip 파일 생성
        img_dir_path = f'SpoitWeb/static/temp'
        zip_img_name = img_dir_path + '/images.zip'
        zip_images(img_dir_path, zip_img_name)
        
        s3.upload_file(img_dir_path+"/images.zip", BUCKET_NAME, DIRECTORY_NAME + "images.zip")

        url = urlGenerate(s3, BUCKET_NAME, "user/images.zip")
        #업로드가 완료되었습니다 창과 함께 모든 사진들을 html에 갤러리 모드로 보여주기 및 아이콘 생성: 모든 영상 다운로드, 분석으로 가는 버튼도 지정
        return render_template('uploaded.html', URL = url)

@app.route('/analysis/')
def anaylsis():
    return "뭐시기"

if __name__ == '__main__':
    app.run(host = '0.0.0.0')

# http://127.0.0.1:5000
