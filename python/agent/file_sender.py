import os
import socket
import traceback

from .config import HOST, FILE_PORT, FILE_CHUNK_SIZE
from .protocol import receive_message, send_message


class FileSender:
    def __init__(self, host=HOST, port=FILE_PORT):
        self.host = host
        self.port = port
        self.running = False

    def start(self):
        print(f"[file-sender] Listening on {self.host}:{self.port}")
        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(1.0)

            while self.running:
                try:
                    client, address = server.accept()
                except socket.timeout:
                    continue

                with client:
                    print(f"[file-sender] Connected: {address}")

                    try:
                        request = receive_message(client)
                        file_path = request.get("file_path")

                        if not file_path or not os.path.exists(file_path):
                            send_message(client, {
                                "status": "error",
                                "message": f"File not found: {file_path}"
                            })
                            continue

                        file_size = os.path.getsize(file_path)
                        file_name = os.path.basename(file_path)

                        send_message(client, {
                            "status": "ok",
                            "file_name": file_name,
                            "file_size": file_size
                        })

                        with open(file_path, "rb") as f:
                            while True:
                                chunk = f.read(FILE_CHUNK_SIZE)
                                if not chunk:
                                    break
                                client.sendall(chunk)

                        print(f"[file-sender] Sent: {file_path}")

                    except Exception:
                        traceback.print_exc()