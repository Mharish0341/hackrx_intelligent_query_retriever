import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENAI_API_KEY = "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"
TEAM_TOKEN = (
    "abb59f46780292d4b07549156e8909080d108a2897f1512cddc6a524d95494d7"
)  # ← keep as-is

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 150
TOP_K_RETRIEVE = 20
