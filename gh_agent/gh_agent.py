from email.mime import message
import socket
import traceback
import threading
import time
from datetime import datetime

from gh_agent.config import (
    LOCAL_GH_AGENT_HOST,
    LOCAL_GH_AGENT_PORT,
    GH_AGENT_NAME,
    GH_AGENT_VERSION,
    POLL_INTERVAL_SEC,
    LOG_DIR,
    GH_AGENT_LOG_FILE,
    RECEIVED_ROOT
)

from gh_agent.protocol import (
    receive_message,
    send_message,
    ok_response,
    error_response,
)

from gh_agent.python_agent_client import PythonAgentClient


class GrasshopperAgent:
    def __init__(
            self,
            host: str = LOCAL_GH_AGENT_HOST,
            port: int = LOCAL_GH_AGENT_PORT):
        self.host = host
        self.port = port
        self.python_client = PythonAgentClient()
        self.running = False

        self.lock = threading.Lock()

        self.latest_job_id = None
        self.latest_job_status = "idle"
        self.latest_message = "No job submitted yet."
        self.latest_result = None
        self.latest_error = None
        self.latest_log = []

        LOG_DIR.mkdir(parents=True, exist_ok=True)

    def write_log(self, message):
        timestamp = datetime.now().isoformat(timespec="seconds")
        line = f"[{timestamp}] {message}"

        with self.lock:
            self.latest_log.append(line)
            self.latest_log = self.latest_log[-50:]

        with open(GH_AGENT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        print(f"[gh-agent-log] {line}")

    def set_latest(
            self,
            job_id=None,
            status=None,
            message=None,
            result=None,
            error=None):

        with self.lock:
            if job_id is not None:
                self.latest_job_id = job_id

            if status is not None:
                self.latest_job_status = status

            if message is not None:
                self.latest_message = message

            if result is not None:
                self.latest_result = result

            if error is not None:
                self.latest_error = error

    def get_latest_status(self):
        with self.lock:
            return ok_response(
                latest_job_id=self.latest_job_id,
                latest_job_status=self.latest_job_status,
                latest_message=self.latest_message,
                latest_error=self.latest_error,
                latest_result=self.latest_result,
                recent_log=self.latest_log[-20:],
                log_file=str(GH_AGENT_LOG_FILE),
            )

    def start_polling_job(self, job_id):
        thread = threading.Thread(
            target=self.poll_job_until_done,
            args=(job_id,),
            daemon=True
        )
        thread.start()

    def poll_job_until_done(self, job_id):
        self.write_log(f"Started polling job: {job_id}")

        while self.running:
            try:
                response = self.python_client.send_command({
                    "command": "get_status",
                    "job_id": job_id
                })

                if response.get("status") != "ok":
                    msg = response.get("message", "Failed to get job status.")
                    self.set_latest(
                        job_id=job_id,
                        status="unknown",
                        message=msg,
                        error=msg
                    )
                    self.write_log(msg)
                    time.sleep(POLL_INTERVAL_SEC)
                    continue

                job = response.get("job", {})
                job_status = job.get("status", "unknown")
                job_message = job.get("message", "")
                job_result = job.get("result")
                job_error = job.get("error")

                self.set_latest(
                    job_id=job_id,
                    status=job_status,
                    message=job_message,
                    result=job_result,
                    error=job_error
                )

                self.write_log(
                    f"Job {job_id}: {job_status} - {job_message}"
                )

                if job_status in ("finished", "failed"):
                    self.write_log(f"Job {job_id} completed with status: {job_status}")
                    break

            except Exception as e:
                self.set_latest(
                    job_id=job_id,
                    status="error",
                    message=str(e),
                    error=str(e)
                )
                self.write_log(f"Polling error for job {job_id}: {e}")

            time.sleep(POLL_INTERVAL_SEC)

    def submit_python_job(self, payload):
        response = self.python_client.send_command(payload)

        job_id = response.get("job_id")

        if not job_id:
            return response

        self.set_latest(
            job_id=job_id,
            status="submitted",
            message=response.get("message", "Job submitted."),
            result=None,
            error=None
        )

        self.write_log(f"Submitted job {job_id}: {payload}")

        self.start_polling_job(job_id)

        return ok_response(
            message="Job submitted to Python Agent.",
            job_id=job_id,
            latest_status="submitted"
        )
    
    def start_download_job(self, message):
        download_job_id = f"download-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        self.set_latest(
            job_id=download_job_id,
            status="download_queued",
            message="Download job submitted.",
            result=None,
            error=None
        )

        self.write_log(f"Download job submitted: {message}")

        thread = threading.Thread(
            target=self._run_download_job,
            args=(download_job_id, message),
            daemon=True
        )
        thread.start()

        return ok_response(
            message="Download job submitted to GH Agent.",
            job_id=download_job_id,
            latest_status="download_queued"
        )
    
    def _run_download_job(self, job_id, message):
        try:
            session = message.get("session")
            output_index = int(message.get("output_index", 1))

            requested_types = message.get("file_types", ["pointcloud", "image", "json"])

            if isinstance(requested_types, str):
                requested_types = [requested_types]

            requested_types = [
                str(t).strip().lower()
                for t in requested_types
            ]

            self.set_latest(
                job_id=job_id,
                status="download_running",
                message=f"Listing downloadable files: {requested_types}"
            )

            self.write_log(
                f"Download running: session={session}, output_index={output_index}, file_types={requested_types}"
            )

            listing = self.python_client.send_command({
                "command": "list_downloadable_outputs",
                "session": session,
                "output_index": output_index,
                "file_types": requested_types
            })

            if listing.get("status") != "ok":
                self.set_latest(
                    job_id=job_id,
                    status="download_failed",
                    message=listing.get("message", "Failed to list downloadable outputs."),
                    error=listing
                )
                self.write_log(f"Download failed while listing files: {listing}")
                return

            saved = []
            files = listing.get("files", {})

            total = sum(len(items) for items in files.values())
            count = 0

            for category, items in files.items():
                for item in items:
                    count += 1

                    remote_path = item["path"]
                    name = item["name"]

                    local_path = (
                        RECEIVED_ROOT /
                        listing["session"] /
                        category /
                        name
                    )

                    self.set_latest(
                        job_id=job_id,
                        status="download_running",
                        message=f"Downloading {count}/{total}: {name}"
                    )

                    self.write_log(f"Downloading {name} → {local_path}")

                    result = self.python_client.download_file(
                        remote_path,
                        local_path
                    )

                    saved.append({
                        "category": category,
                        "name": name,
                        "remote_path": remote_path,
                        "local_path": str(local_path),
                        "result": result
                    })

            self.set_latest(
                job_id=job_id,
                status="download_finished",
                message=f"Downloaded {len(saved)} file(s).",
                result=saved,
                error=None
            )

            self.write_log(f"Download finished: {len(saved)} file(s).")

        except Exception as e:
            err = traceback.format_exc()

            self.set_latest(
                job_id=job_id,
                status="download_failed",
                message="Download exception.",
                error=err
            )

            self.write_log(f"Download exception: {e}")
            print(err)

    def handle_local_message(self, message):
        command = message.get("command")

        if not command:
            return error_response("Missing command.")

        if command == "ping":
            return ok_response(
                message="Grasshopper Agent is alive."
            )

        if command == "ping_python":
            return self.python_client.send_command(
                {
                    "command": "ping"
                }
            )

        if command == "get_latest_status":
            return self.get_latest_status()

        if command == "forward":
            payload = message.get("payload")

            if not isinstance(payload, dict):
                return error_response("Missing or invalid payload.")

            return self.submit_python_job(payload)
        
        if command == "downloadable_outputs":
            return self.start_download_job(message)

        return error_response(f"Unknown GH Agent command: {command}")

    def start(self):
        print(f"[gh-agent] {GH_AGENT_NAME} v{GH_AGENT_VERSION}")
        print(f"[gh-agent] Listening locally on {self.host}:{self.port}")
        print(f"[gh-agent] Forwarding to Python Agent at "
              f"{self.python_client.host}:{self.python_client.port}")
        print("[gh-agent] Press Ctrl+C to stop.")

        self.running = True

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
            server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            server.bind((self.host, self.port))
            server.listen()
            server.settimeout(1.0)

            try:
                while self.running:
                    try:
                        client, address = server.accept()
                    except socket.timeout:
                        continue

                    with client:
                        print(f"[gh-agent] Local connection: {address}")

                        try:
                            message = receive_message(client)
                            print(f"[gh-agent] Received local message: {message}")

                            response = self.handle_local_message(message)

                        except Exception as e:
                            traceback.print_exc()
                            response = error_response(str(e))

                        send_message(client, response)
                        print("[gh-agent] Response sent.")

            except KeyboardInterrupt:
                print("\n[gh-agent] Ctrl+C received.")

            finally:
                self.running = False
                print("[gh-agent] Shutting down...")


def main():
    agent = GrasshopperAgent()
    agent.start()


if __name__ == "__main__":
    main()