from pathlib import Path

# Network
HOST = "0.0.0.0"
COMMAND_PORT = 5005
FILE_PORT = 5006

# Message size
HEADER_SIZE = 8
BUFFER_SIZE = 1024 * 1024
FILE_CHUNK_SIZE = 1024 * 1024

# Project paths
PYTHON_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = PYTHON_DIR.parent

# Agent
AGENT_NAME = "Python Agent"
AGENT_VERSION = "0.1.0"