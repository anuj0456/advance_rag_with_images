from flask import Flask
from controller.views import bp

app = Flask(__name__)
app.register_blueprint(bp)


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
