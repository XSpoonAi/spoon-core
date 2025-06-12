import argparse
import asyncio
import logging
from dotenv import load_dotenv

from cli.commands import SpoonAICLI

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

load_dotenv(override=True)
async def main():
    parser = argparse.ArgumentParser(description="SpoonAI CLI")
    parser.add_argument('--server', action='store_true', help='Start the server')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', default=8000, type=int, help='Server port')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()
    if args.server:
        raise NotImplementedError("Server model is not implemented yet")

    else:
        cli = SpoonAICLI()
        await cli.run()

if __name__ == "__main__":
    asyncio.run(main())
        