from pkg.modules import *
from pkg.functions import *

os.chdir('../')
app = Flask(__name__)
warnings.filterwarnings(action='ignore')

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

@app.route('/upload_goal/', methods = ['GET', 'POST'])
def upload_goal():
    if request.method == 'GET':
        return render_template('upload_goal.html')
    elif request.method == 'POST':
        files = request.files.getlist('files')
        rets = long_running_task_goal(files)
        return render_template('uploaded_goal.html')

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 8080)

# http://127.0.0.1:5000
