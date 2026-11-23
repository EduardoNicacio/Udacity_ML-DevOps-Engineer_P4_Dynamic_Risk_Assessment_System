"""
WSGI entry point: exposes the Flask API as the ``application`` callable.
"""

from app import app

application = app


if __name__ == "__main__":
    app.run()
