import os
from dotenv import load_dotenv

# Load environment variables from a .env file
# This will look for a .env file in the 'backend' directory
load_dotenv()

# Get the OpenAI API key
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Get the Supabase database URL
DATABASE_URL = os.environ.get("DATABASE_URL")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set.")

if not DATABASE_URL:
    print("Warning: DATABASE_URL environment variable not set.")
