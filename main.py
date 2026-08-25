"""
main.py
Single entry point for the RAG project.

Usage:
    # Ingest a PDF, then ask one question:
    python main.py --pdf data/networking_notes.pdf --question "What's the difference between TCP and UDP?"

    # Just ingest (no question):
    python main.py --pdf data/networking_notes.pdf

    # Already ingested? Just ask questions, one after another:
    python main.py --chat
"""

import argparse
from dotenv import load_dotenv

load_dotenv()  # reads GROQ_API_KEY (and any other vars) from a .env file if present

import os
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found. Check your .env file exists in the project root.")

from src.ingest import ingest_pdf
from src.query import answer_question


def main():
    parser = argparse.ArgumentParser(description="RAG project: ingest PDFs and ask questions.")
    parser.add_argument("--pdf", type=str, help="Path to a PDF to ingest")
    parser.add_argument("--question", type=str, help="A single question to ask")
    parser.add_argument("--chat", action="store_true", help="Start an interactive question loop")
    args = parser.parse_args()

    if args.pdf:
        ingest_pdf(args.pdf)

    if args.question:
        answer = answer_question(args.question)
        print("=" * 60)
        print(answer)
        print("=" * 60)

    if args.chat:
        print("Chat mode. Type 'exit' to quit.\n")
        while True:
            question = input("You: ").strip()
            if question.lower() in ("exit", "quit"):
                break
            if not question:
                continue
            answer = answer_question(question, verbose=False)
            print(f"\nAssistant: {answer}\n")

    if not (args.pdf or args.question or args.chat):
        parser.print_help()


if __name__ == "__main__":
    main()