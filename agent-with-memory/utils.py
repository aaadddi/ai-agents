import os
from dotenv import load_dotenv
from redis import Redis
import getpass

def set_env_key():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def get_redis_client():
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = Redis.from_url(REDIS_URL)
    redis_client.ping()

    return redis_client