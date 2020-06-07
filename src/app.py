from flask import Flask, request
from server import Server

app = Flask(__name__)
server = None


@app.route('/start', methods=['POST'])
def start_server():
    global server
    if server:
        server.do_run = False
    tickers = request.json['tickers']
    symbols = request.json['symbols']
    server = Server(dict(zip(tickers, symbols)))
    server.start()
    return app.response_class(status=200)


@app.route('/stop', methods=['DELETE'])
def stop_server():
    global server
    if server:
        server.do_run = False
    server = None
    return app.response_class(status=200)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
