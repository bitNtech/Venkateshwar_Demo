"""Manual interactive REPL: start_call() once, then feed typed caller lines
through handle_utterance() turn by turn against whatever LLM_BASE_URL is
actually configured to.

Not a pytest test - same category as smoke_llm.py, which this script reuses
its logging shim and sample metadata from rather than duplicating them. It
imports and uses ConversationManager/LlmClient exactly as they exist today.

Usage:
    python -m backend.scripts.interactive_llm

Type a caller line and press enter to send it; type "exit" or "quit" (or
press Ctrl+C / Ctrl+D) to end the session. Requires LLM_BASE_URL (default
http://localhost:8001/v1) to point at a running OpenAI-compatible
chat-completions server.
"""

from __future__ import annotations

import asyncio
import sys

# Windows consoles default to cp1252, which can't print Tamil script - force
# UTF-8 so this doesn't crash mid-run on the exact platform it's meant for.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from backend.conversation import ConversationManager
from backend.llm import LlmClient
from backend.scripts.smoke_llm import SAMPLE_METADATA, _LoggingLlmClient
from backend.settings import ConversationSettings, LlmSettings

EXIT_WORDS = {"exit", "quit"}


async def main() -> None:
    llm_settings = LlmSettings()
    print(f"LLM_BASE_URL={llm_settings.base_url}  LLM_MODEL={llm_settings.model}")

    llm = LlmClient(llm_settings)
    llm.load()
    logging_llm = _LoggingLlmClient(llm)

    manager = ConversationManager(ConversationSettings())
    manager.load()

    connection_id = "interactive"
    greeting = manager.start_call(connection_id, **SAMPLE_METADATA)
    print(f"\n=== start_call() greeting ===\n{greeting}")
    print("\nType a caller line and press enter. Type 'exit' or 'quit' to end.")

    while True:
        try:
            line = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return

        if not line:
            continue
        if line.lower() in EXIT_WORDS:
            print("Exiting.")
            return

        try:
            reply = await manager.handle_utterance(connection_id, logging_llm, line)
        except Exception as error:
            print(
                f"\nRound trip failed ({error!r}) - is a real OpenAI-compatible server actually "
                f"running at {llm_settings.base_url}? (set LLM_BASE_URL / LLM_MODEL / LLM_API_KEY "
                "to point at one)"
            )
            continue

        print(f"\n=== agent reply ===\n{reply}")


if __name__ == "__main__":
    asyncio.run(main())
