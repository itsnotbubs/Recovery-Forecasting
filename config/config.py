from dotenv import load_dotenv
import os
from pathlib import Path


env_path = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("api_key")
user = os.getenv("username")
password = os.getenv("password")
