"""OpenEnv validator entrypoint.

Provides `main()` and a `__main__` guard for validators that expect a runnable
server module at `server/app.py`.
"""

from app import app


def main() -> None:
	import uvicorn

	uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
