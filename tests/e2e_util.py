import atexit
import json
import os
import signal
import subprocess
import time
from contextlib import contextmanager

from playwright.sync_api import sync_playwright

from prodigy.util import msg

PORT = 12346
BASE_URL = f"http://localhost:{PORT}"


@contextmanager
def prodigy_run_server(command: str, overrides={"port": PORT}):
    """
    Starts Prodigy, allow you to run playwright, turn it off after
    """
    config = json.dumps(overrides)
    cmd = f"python -m prodigy {command}"
    environment = os.environ
    environment["PRODIGY_CONFIG_OVERRIDES"] = f"{config}"
    serv_cmd = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE, env=environment)

    # Make sure we shutdown properly if program exits via CTRL+C
    atexit.register(lambda: os.kill(serv_cmd.pid, signal.SIGKILL))

    # Sleep a bit, just in case.
    time.sleep(4)
    msg.info(f"running: PRODIGY_CONFIG_OVERRIDES='{config}' {cmd} on pid={serv_cmd.pid}")
    yield serv_cmd.pid
    msg.info(f"shutting down {cmd}")
    os.kill(serv_cmd.pid, signal.SIGKILL)


@contextmanager
def prodigy_playwright(command: str, overrides=dict(), headless: bool = True):
    """
    Starts Prodigy, then playwright.
    """
    overrides = {"port": PORT, **overrides}
    with prodigy_run_server(command, overrides):
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()
            page.goto(BASE_URL)
            yield context, page
            context.close()
            browser.close()
