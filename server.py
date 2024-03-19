# 골 디택션 모듈 => KTCC에서는 사용 X
# from __future__ import annotations
# from typing import List, Tuple
# from supervision import VideoSink
# import torch
# from supervision import VideoInfo
# from supervision import get_video_frames_generator
# from shapely.geometry import Point, Polygon
from shapely.geometry import box
import pickle
import numpy as np
from ultralytics import YOLO # 객체 탐지 모듈
import supervision as sv # 객체 탐지 라벨링 모듈
import cv2 # 컴퓨터 비전 관련 모듈
from flask import Flask, render_template, request, redirect, jsonify # 웹 프레임워크 관련 모듈
from werkzeug.utils import secure_filename # 링크 보안 관련 모듈
import boto3 # 서버 관련 모듈
import zipfile # 압축 관련 모듈
import os # 경로 관련 모듈
import shutil # 디렉토리의 내용 삭제 관련 모듈
import logging # 서버 로그 모듈
from botocore.exceptions import NoCredentialsError, ClientError # 서버 에러 핸들링 관련 모듈
import pandas as pd # 데이터 분석 모듈
import mplsoccer # 축구 데이터 분석용 모듈
import warnings # pandas warning 무시용
from tqdm import tqdm # 로딩창 시각화
from markupsafe import Markup # FLASK => HTML

os.chdir('../')
s3 = boto3.client('s3')
BUCKET_NAME = 'spoits3'
DIRECTORY_NAME = 'user/'
app = Flask(__name__)
warnings.filterwarnings(action='ignore')

ball_pt_path ='SpoitWeb/static/models/best.pt'
detection_model = YOLO(ball_pt_path)
position_pt_path ='SpoitWeb/static/models/best_position_v2.pt'
position_model = YOLO(position_pt_path)

def upload_to_s3(s3, file, bucket):
    try:
        secure_filename_str = secure_filename(file.filename)
        s3.upload_fileobj(file, bucket, (DIRECTORY_NAME + secure_filename_str))
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
            # 이미지 파일인 경우에만 압축합니다. 여기서는 확장자가 .jpg, .png, .gif .jpeg로 가정합니다.
            if file.lower().endswith(('.jpg', '.png', '.gif', 'jpeg', '.pickle')):
                file_path = os.path.join(folder_path, file)
                zipf.write(file_path, os.path.basename(file_path))

def detected_image_generator(image_path, save_path):
    image = cv2.imread(image_path)

    result = detection_model(image)[0]
    detections = sv.Detections.from_ultralytics(result)

    bounding_box_annotator = sv.BoundingBoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position = sv.Position.TOP_LEFT)

    annotated_frame_box = bounding_box_annotator.annotate(
        scene = image.copy(),
        detections = detections
    )

    annotated_frame_label = label_annotator.annotate(
        scene = annotated_frame_box,
        detections = detections,
        labels = [
            result.names[class_id] for class_id in detections.class_id
        ]
    )
    ids = np.array(result.boxes.cls.cpu())
    boxes = np.array(result.boxes.xyxy.cpu())
    boxesByClass = [[] for _ in range(4)]
    for box, id in zip(boxes, ids): boxesByClass[int(id)].append(list(box))
    with open(save_path + '/' + os.path.basename(image_path).strip('.jpg') + '.pickle', 'wb') as f: pickle.dump(boxesByClass, f)
    cv2.imwrite(save_path + '/' + os.path.basename(image_path), annotated_frame_label)

def get_images(folder_path):
    file_list = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    ret = []
    for file in file_list:
        if file.lower().endswith(('jpg', 'png', 'gif', 'jpeg')):
            file_path = os.path.join(folder_path, file)
            ret.append(file_path)

    return ret

def long_running_task_detection(files):
    # 디렉토리를 초기화하는 작업 (user login 시스템을 갖출 경우, 각 user의 초기화가 필요)
    directory_paths = ['SpoitWeb/static/temp', 'SpoitWeb/static/detected']
    for directory_path in directory_paths:
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
        print(secure_filename_str)
        save_path = f'SpoitWeb/static/temp/{secure_filename_str}'
        file.save(save_path)

        if not upload_to_s3(s3, file, BUCKET_NAME):
            return 'Failed to upload one or more files to AWS S3.'

    # zip 파일 생성
    img_dir_path = f'SpoitWeb/static/temp'
    zip_img_name = img_dir_path + '/images.zip'
    zip_images(img_dir_path, zip_img_name)

    s3.upload_file(img_dir_path+"/images.zip", BUCKET_NAME, DIRECTORY_NAME + "images.zip")

    url1 = urlGenerate(s3, BUCKET_NAME, "user/images.zip")

    # 객체 탐지
    detected_dir_path = 'SpoitWeb/static/detected'

    img_file_paths = get_images(img_dir_path)
    for img_file_path in tqdm(img_file_paths):
        detected_image_generator(img_file_path, detected_dir_path)

    # 객체 탐지 zip 파일 생성
    zip_images(detected_dir_path, detected_dir_path + '/detected_images.zip')

    s3.upload_file(detected_dir_path + '/detected_images.zip', BUCKET_NAME, DIRECTORY_NAME + "detected_images.zip")

    url2 = urlGenerate(s3, BUCKET_NAME, "user/detected_images.zip")
    return [url1, url2]

class Plot():
  def __init__(self):
    self.pitch = mplsoccer.Pitch(pitch_color = 'grass', line_color = 'white')
    self.fig, self.ax = self.pitch.draw(figsize = (20,15))
  def call_plot(self):
    return self.pitch, self.fig, self.ax
  def set_title(self, title):
    self.ax.set_title(title)
  def save_image(self, path):
    self.fig.savefig(path)

def heatmap(pass_df, plot):
  pitch, fig, ax = plot.call_plot()
  kde = pitch.kdeplot(
      x = pass_df['x'], y = pass_df['y'], ax = ax,
      fill = True, thresh = 0.05, alpha = .5, levels = 50, cmap = 'viridis'
  )

def sided(position):
    position = position.replace('Right', 'Side')
    position = position.replace('Left', 'Side')
    return position

def limitations(probs):
    '''
    {0: 'Center Attacking Midfield', 1: 'Center Defensive Midfield', 2: 'Goalkeeper', 3: 'Left Back', 4: 'Left Center Back', 5: 'Left Center Midfield', 6: 'Left Defensive Midfield', 7: 'Left Wing', 8: 'Right Back', 9: 'Right Center Back', 10: 'Right Center Midfield', 11: 'Right Defensive Midfield', 12: 'Right Wing'}
    '''
    probs[0] *= 3
    probs[1] *= 2.5 
    probs[7] *= 5
    probs[12] *= 5
    make100 = sum(probs)
    for i in range(len(probs)):
        probs[i] = probs[i] / make100
    return probs

def make_percentage(probs):
    for i in range(len(probs)):
        probs[i] *= 100
        probs[i] = f'{probs[i]:.0f}%'

def position_prediction_YOLO(given_pos, heatmap_path):
    results = position_model.predict(heatmap_path, save=False)
    for result in results:
        probs = result.probs.data.tolist()
        probs = limitations(probs)
    ranks = sorted(probs, reverse = True)
    first =  position_model.names[probs.index(ranks[0])]
    second = position_model.names[probs.index(ranks[1])]
    third = position_model.names[probs.index(ranks[2])]
    make_percentage(probs)
    print(probs, flush = True)
    print(position_model.names, flush = True)
    print(first, second, third, flush = True)
    position = sided(given_pos); first = sided(first); second = sided(second); third = sided(third)
    if first == 'Goalkeeper': return 'Goalkeeper'
    if first == second:
        return third
    return second
def get_coach_recommendation(pos):
    if pos == 'Center Attacking Midfield' or 'Side Wing':
        pos = 'Forward'
    with open(f'SpoitWeb/static/coachDB/coach/{pos}.txt', 'r') as file:
        data = file.read().replace('\n', '<br />')
    return data

def long_running_task_coach(files):
    # 디렉토리를 초기화하는 작업 (user login 시스템을 갖출 경우, 각 user의 초기화가 필요)
    directory_paths = ['SpoitWeb/static/original']
    for directory_path in directory_paths:
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

    # original 경로에 csv 파일 저장
    for file in files:
        print(file)
        if file.filename == '':
            return redirect(request.url)

        secure_filename_str = secure_filename(file.filename)

        save_path = f'SpoitWeb/static/original/{secure_filename_str}'
        file.save(save_path)

        if not upload_to_s3(s3, file, BUCKET_NAME):
            return 'Failed to upload one or more files to AWS S3.'

    # 파일 업로드
    s3.upload_file(save_path, BUCKET_NAME, DIRECTORY_NAME + f"{secure_filename_str}")
    url1 = urlGenerate(s3, BUCKET_NAME, DIRECTORY_NAME + f"{secure_filename_str}")

    # 데이터를 임포트하고, 포지션 이름을 파일 이름으로부터 추출하기
    data = pd.read_csv(save_path)
    position = save_path.split('_')[-1].strip('.csv')
    
    # 히트맵 만들기
    map = Plot()
    map.set_title("Player Passmap")
    heatmap(data, map)
    heatmap_path = 'SpoitWeb/static/images/heatmap.png'
    map.save_image(heatmap_path)

    recommended_position = position_prediction_YOLO(position, heatmap_path)
    position_explanation = f"<p>기존 포지션: {position}, 추천 드리는 포지션: {recommended_position}</p>"
    coach_recommendation = get_coach_recommendation(recommended_position)
    return [url1, recommended_position, position_explanation, coach_recommendation]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/explanation/')
def explanation():
    return render_template('explanation.html')

@app.route('/upload_detection/', methods = ['GET', 'POST'])
def upload_detection():
    if request.method == 'GET':
         return render_template('upload_detection.html')
    elif request.method == 'POST':
        files = request.files.getlist('files')
        urls = long_running_task_detection(files)
        # 업로드가 완료되었습니다 창과 함께 모든 사진들을 html에 갤러리 모드로 보여주기 및 아이콘 생성: 모든 영상 다운로드, 분석으로 가는 버튼도 지정
        # return render_template('loading.html', task_id = task.id)
        return render_template('uploaded_detection.html', URL_original = urls[0], URL_detected = urls[1])

@app.route('/upload_coach/', methods = ['GET', 'POST'])
def upload_coach():
    if request.method == 'GET':
        return render_template('upload_coach.html')
    elif request.method == 'POST':
        files = request.files.getlist('files')
        rets = long_running_task_coach(files)
        return render_template('uploaded_coach.html', URL_original = rets[0], position_name = rets[1], feature_explanation = Markup(rets[2]), coach_explanation = Markup(rets[3]))

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)

# http://127.0.0.1:8080
