import os
import socket
from typing import Any, Dict
from pathlib import Path

from gh_agent.config import PYTHON_AGENT_HOST, PYTHON_AGENT_PORT, FILE_CHUNK_SIZE, PYTHON_AGENT_FILE_PORT
from gh_agent.protocol import send_message, receive_message 


class PythonAgentClient:
    def __init__(
            self,
            host: str = PYTHON_AGENT_HOST,
            port: int = PYTHON_AGENT_PORT):
        self.host = host
        self.port = port

    def send_command(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, self.port))

            send_message(sock, payload)

            response = receive_message(sock)

        return response
    
    def download_file(self, remote_file_path, local_file_path):
        
        local_file_path = Path(local_file_path)
        local_file_path.parent.mkdir(parents=True, exist_ok=True)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.connect((self.host, PYTHON_AGENT_FILE_PORT))

            send_message(sock, {
                "file_path": remote_file_path
            })

            header = receive_message(sock)

            if header.get("status") != "ok":
                return header

            file_size = int(header["file_size"])
            received = 0

            with open(local_file_path, "wb") as f:
                while received < file_size:
                    chunk = sock.recv(min(FILE_CHUNK_SIZE, file_size - received))
                    if not chunk:
                        break

                    f.write(chunk)
                    received += len(chunk)

            return {
                "status": "ok",
                "local_path": str(local_file_path),
                "file_size": file_size,
                "received": received
            }