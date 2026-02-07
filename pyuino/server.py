import argparse
import json
import socket
import logging
from .model import YuinoModel
from .converter import YuinoConverter


class YuinoServer:
    def __init__(self, model_path: str, port: int = 30055, buffer: int = 1024):
        self._logger = logging.getLogger('AnyaServer')
        self._s_addr = ("0.0.0.0", port)
        model = YuinoModel.from_pretrained(model_path)
        self._converter = YuinoConverter(model)
        self._buffer = buffer
        self._logger.info("Yuino-Sever START! (port=%d)" % port)

    def __call__(self) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(self._s_addr)
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    message_recv = conn.recv(self._buffer).decode('utf-8')
                    if message_recv:
                        message_resp = self._respond(message_recv)
                        conn.send(message_resp.encode('utf-8'))

    def _respond(self, message: str) -> str:
        ret_msg = {}

        try:
            msg = json.loads(message)
            if msg["op"] == "CONVERT":
                ret_str = self._converter.convert(msg["args"]["pron"])
                ret_msg = json.dumps({"candidates": {"form": [ret_str]}})

        except json.decoder.JSONDecodeError:
            pass

        return ret_msg


def run():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-m', '--model', help='model path', default='AnyaLM')
    arg_parser.add_argument('-p', '--port', help='server port number', default=30055)
    args = arg_parser.parse_args()

    server = YuinoServer(args.model, args.port)
    server()
