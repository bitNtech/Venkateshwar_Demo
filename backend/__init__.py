"""VAD + ASR pipeline package."""

from dotenv import load_dotenv

# Loaded here, before any module body runs: settings.py reads its defaults
# with os.getenv() at IMPORT time, so every `import backend.*` - run.sh, bare
# uvicorn, a script, a test - must see .env first. Existing process env wins.
load_dotenv()
