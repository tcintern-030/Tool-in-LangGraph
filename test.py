import os
from dotenv import load_dotenv

load_dotenv()

print("API KEY FOUND:", bool(os.getenv("GEMINI_API_KEY")))