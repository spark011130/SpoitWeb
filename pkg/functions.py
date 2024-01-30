from modules import *

pt_path ='SpoitWeb/static/models/best.pt'
detection_model = YOLO(pt_path)
os.chdir('../')
s3 = boto3.client('s3')
BUCKET_NAME = 'spoits3'
DIRECTORY_NAME = 'user/'
app = Flask(__name__)
warnings.filterwarnings(action='ignore')

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
            if file.lower().endswith(('.jpg', '.png', '.gif', 'jpeg')):
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
  def save_image(self):
    self.fig.savefig(f'SpoitWeb/static/images/heatmap.png')

def heatmap(pass_df, plot):
  pitch, fig, ax = plot.call_plot()
  kde = pitch.kdeplot(
      x = pass_df['x'], y = pass_df['y'], ax = ax,
      fill = True, thresh = 0.05, alpha = .5, levels = 50, cmap = 'viridis'
  )

def alternative_position(conc):
    if conc in ['Center Forward', 'Left Center Forward', 'Right Center Forward', 'Left Wing', 'Right Wing']:
        return 'Forward'
    elif conc in ['Left Attacking Midfield', 'Right Attacking Midfield', 'Center Attacking Midfield', 'Secondary Striker', 'Right Midfield', 'Left Midfield', 'Left Center Midfield', 'Right Center Midfield']:
        return 'Midfield'
    elif conc in ['Left Wing Back', 'Left Back', 'Right Wing Back', 'Right Back']:
        return 'Side Back'
    elif conc in ['Left Center Back', 'Right Center Back', 'Center Back']:
        return 'Center Back'
    elif conc in ['Goalkeeper']:
        return 'Goalkeeper'
    return 'not known'

def position_prediction_euclidean(predicted_pos, data):
    player_x = data['x'].mean()
    player_y = data['y'].mean()

    min_dist = float('inf')
    conc = ''

    for i in range(len(predicted_pos)):
        position_name, x, y = predicted_pos.iloc[i][0], predicted_pos.iloc[i][1], predicted_pos.iloc[i][2]
        dist = ((x-player_x)**2 + (y-player_y)**2)**0.5
        if min_dist > dist:
            min_dist = dist
            conc = position_name
    return alternative_position(conc)

def get_position_explanation(pos):
    with open(f'SpoitWeb/static/coachDB/feature/{pos}.txt', 'r') as file:
        data = file.read().replace('\n', '<br />')
    return data

def get_coach_recommendation(pos):
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

    # 포지션별 평균 위치 데이터 불러오기 및 euclidean distance 계산해서 최고의 포지션 예측하기
    position_path = 'SpoitWeb/static/coachDB/position_xy.csv'
    predicted_pos = pd.read_csv(position_path)
    data = pd.read_csv(save_path)
    predicted_pos_ret = position_prediction_euclidean(predicted_pos, data)
    position_explanation = get_position_explanation(predicted_pos_ret)
    coach_recommendation = get_coach_recommendation(predicted_pos_ret)

    # 히트맵 만들기
    map = Plot()
    map.set_title("Player Passmap (=>)")
    heatmap(data, map)
    map.save_image()

    return [url1, predicted_pos_ret, position_explanation, coach_recommendation]

def long_running_task_goal():
    pass