from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from predict import generate_caption

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Tạo thư mục nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    filename = None
    caption = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            caption = generate_caption(file_path)

    return render_template('index.html', filename=filename, caption=caption)

if __name__ == '__main__':
    app.run(debug=True)

# python310 app.py