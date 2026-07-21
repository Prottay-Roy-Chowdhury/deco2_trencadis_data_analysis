import socket
import traceback
import threading

from agent.config import HOST, COMMAND_PORT, AGENT_NAME, AGENT_VERSION
from agent.protocol import receive_message, send_message, error_response
from agent.command_handler import CommandHandler
from agent.file_sender import FileSender
from agent.design_file_receiver import DesignFileReceiver


class PythonAgent:
    def __init__(self, host=HOST, port=COMMAND_PORT):
        self.host = host
        self.port = port
        self.handler = CommandHandler()
        self.file_sender = FileSender()
        self.design_file_receiver = DesignFileReceiver()
        self.running = False

    def start(self):
        print(f"[agent] {AGENT_NAME} v{AGENT_VERSION}")
        print(f"[agent] Listening on {self.host}:{self.port}")
        print("[agent] Press Ctrl+C to stop.")

        file_thread = threading.Thread(
            target=self.file_sender.start,
            daemon=True
        )
        file_thread.start()

        upload_thread = threading.Thread(
            target=self.design_file_receiver.start,
            daemon=True,
        )
        upload_thread.start()

        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()

            # Important on Windows / VS Code:
            # allows Ctrl+C to be detected instead of blocking forever in accept()
            server.settimeout(1.0)

            try:
                while self.running:
                    try:
                        client, address = server.accept()
                    except socket.timeout:
                        continue

                    with client:
                        print(f"[agent] Connected: {address}")

                        try:
                            message = receive_message(client)
                            print(f"[agent] Received: {message}")

                            response = self.handler.handle(message)

                        except Exception as e:
                            traceback.print_exc()
                            response = error_response(str(e))

                        send_message(client, response)
                        print("[agent] Response sent.")

            except KeyboardInterrupt:
                print("\n[agent] Ctrl+C received.")

            finally:
                self.running = False
                print("[agent] Shutting down...")


def main():
    agent = PythonAgent()
    agent.start()


if __name__ == "__main__":
    main()