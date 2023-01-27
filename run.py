from app import app
from Monitoring_thread.monitor import Monitor


@Monitor()
def run_app():
    app.run()


if __name__ == "__main__":
    run_app()
