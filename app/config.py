import os
from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TEAM_TOKEN = (
    "abb59f46780292d4b07549156e8909080d108a2897f1512cddc6a524d95494d7"
)  # ← keep as-is

TOP_K_RETRIEVE = 10
