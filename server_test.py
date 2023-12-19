from flask import Flask
from flask import request

app = Flask(__name__)

# 제목, id, 본문에 나타날 내용은 dictionary에 저장
topics = [
    {'id': 1, 'title':'html', 'body':'html is ...'},
    {'id': 2, 'title':'css', 'body':'css is ...'},
    {'id': 3, 'title':'javascript', 'body':'javascript is ...'}
]

def template(contents, title, content):
    return f''' <!doctype html>
    <html>
        <body>
            <h1><a href = "/">WEB</a></h1>
            <ol>
                {contents}
            </ol>
            <h2>{title}</h2>
            {content}
            <ul>
            <li><a href = "/create/">create</a></li>
            </ul>
        </body>
    </html>
'''

def getContents():
    liTags = ''
    for topic in topics:
        liTags = liTags + f'<li><a href = "/read/{topic["id"]}/">{topic["title"]}</a></li>'
    return liTags

def getTopicData(id):
    title = ''
    body = ''
    for topic in topics:
        if id == topic['id']:
            title = topic['title']
            body = topic['body']
            break
    return title, body

@app.route('/')
def index():
    liTags = getContents()    
    return template(liTags, "Welcome", "Hello, web")

@app.route('/create/', methods = ['GET', 'POST'])
def create(): # GET할시 URL에 변경 (default), POST시 더욱 은밀하게 전송
    print("request method = ", request.method)
    if request.method == 'GET':
        content = '''
            <form action = "/create/" method = "POST"> 
                <p><input type = "text" name = "title" placeholder="title"></p>
                <p><textarea name = "body" placeholder="body"></textarea></p>
                <p><input type="submit" value = "create"></p>
            </form>
        '''
        return template(getContents(), 'create', content)
    elif request.method == 'POST':
        liTags = getContents()
        title = request.form['title']
        body = request.form['body']
        return template(liTags, title, body)
    

@app.route(f'/read/<int:id>/')
def read(id):
    liTags = getContents()
    title, body = getTopicData(id)
    return template(liTags, title, body)

app.run(debug=True) #debug for 편의
