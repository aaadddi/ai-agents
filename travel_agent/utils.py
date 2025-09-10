import os
from dotenv import load_dotenv
from redis import Redis
import getpass
from redisvl.utils.vectorize.text.vertexai import VertexAITextVectorizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.redis import RedisSaver

SYSTEM_USER_ID = "system"

def set_env_key():
    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv('GOOGLE_API_KEY')
    os.environ["GOOGLE_CLOUD_PROJECT"] = os.getenv('GOOGLE_CLOUD_PROJECT')
    os.environ["GOOGLE_CLOUD_LOCATION"] = os.getenv('GOOGLE_CLOUD_LOCATION')
    # os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = os.getenv('GOOGLE_GENAI_USE_VERTEXAI')

    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")

def get_redis_client():
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
    redis_client = Redis.from_url(REDIS_URL)
    if not redis_client.ping(): # testing redis server
        raise "Unable to reach redis"
    return redis_client

def get_redis_saver(redis_client=None):
    if not redis_client:
        redis_client = get_redis_client()    
    redis_saver = RedisSaver(redis_client=redis_client)
    redis_saver.setup()
    return redis_saver

def get_vertex_embed(api_config=None):
    load_dotenv()
    try:
        # Get project configuration from environment variables
        if not api_config:
            project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
            location = os.getenv('GOOGLE_CLOUD_LOCATION', 'us-central1')
        
            if not project_id:
                raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is not set")
        
        # Configure Vertex AI with the correct project ID
            api_config = {
                "project_id": project_id,
                "location": location
            }
        
        vertex_embed = VertexAITextVectorizer(
            model="text-embedding-004",
            api_config=api_config
        )
        return vertex_embed

    except Exception as e:
        raise e

def get_llm(tools=None):

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    if tools:
        llm.bind_tools(tools)
    
    return llm