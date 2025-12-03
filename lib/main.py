import threading
import traceback
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter, sleep
import os
import logging
from pathlib import Path  # noqa
import xml.etree.ElementTree as ET  # noqa

from fastapi import FastAPI
from faster_whisper import WhisperModel
from nc_py_api import AsyncNextcloudApp, NextcloudApp
from nc_py_api.ex_app import (
    get_computation_device,
    persistent_storage,
    run_app,
    set_handlers,
    setup_nextcloud_logging,
)

from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider
from ocs import get_file

# ---------Start of configuration values for manual deploy---------
# Uncommenting the following lines may be useful when installing manually.

# xml_path = Path(__file__).resolve().parent / "../appinfo/info.xml"
# os.environ["APP_VERSION"] = ET.parse(xml_path).getroot().find(".//image-tag").text
#
# os.environ["NEXTCLOUD_URL"] = "http://nextcloud.local/index.php"
# os.environ["APP_HOST"] = "0.0.0.0"
# os.environ["APP_PORT"] = "9030"
# os.environ["APP_ID"] = "stt_whisper2"
# os.environ["APP_SECRET"] = "12345"  # noqa
# ---------Enf of configuration values for manual deploy---------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
LOGGER = logging.getLogger(os.environ["APP_ID"])
LOGGER.setLevel(logging.DEBUG)


def load_models():
    models = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for file in os.scandir(dir_path + "/../models/"):
        if os.path.isdir(file.path):
            models[file.name] = create_model_loader(file.path)
    for file in os.scandir(persistent_storage()):
        if os.path.isdir(file.path):
            models[file.name] = create_model_loader(file.path)

    return models

def create_model_loader(file_path):
    device = get_computation_device().lower()
    if device != "cuda":  # other GPUs are currently not supported by Whisper
        device = "cpu"

    return lambda: WhisperModel(file_path, device=device)

ENABLED = Event()

TRIGGER = Event()
WAIT_INTERVAL = 5
WAIT_INTERVAL_WITH_TRIGGER = 5 * 60
models = load_models()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    global ENABLED
    setup_nextcloud_logging("stt_whisper2", logging_level=logging.WARNING)
    set_handlers(
        APP,
        enabled_handler,
        trigger_handler=trigger_handler
    )
    nc = NextcloudApp()
    if nc.enabled_state:
        ENABLED.set()
    start_bg_task()
    yield


APP = FastAPI(lifespan=lifespan)

LAST_MODEL_NAME = None
LAST_MODEL = None

def start_bg_task():
    t = threading.Thread(target=background_thread_task)
    t.start()

def background_thread_task():
    global ENABLED
    global LAST_MODEL_NAME
    global LAST_MODEL

    nc = NextcloudApp()
    while True:
        while not ENABLED.is_set():
            sleep(5)

        try:
            item = nc.providers.task_processing.next_task([f'stt_whisper2:{model_name}' for model_name, _ in models.items()], ['core:audio2text'])
            if not isinstance(item, dict):
                wait_for_task()
                continue
            task = item.get("task")
            if task is None:
                wait_for_task()
                continue
        except Exception as e:
            LOGGER.error(str(e) + "\n" + "".join(traceback.format_exception(e)))
            wait_for_task(10)
            continue
        try:
            LOGGER.info(f"Next task: {task['id']}")
            provider = item.get("provider")
            if provider is None:
                raise ValueError('Next task endpoint did not provide a provider name')
            name = provider.get('name')
            if not isinstance(name, str) or ':' not in name:
                raise ValueError(f"Invalid provider name: {name!r}")
            model_name = name.split(':', 2)[1]
            LOGGER.info( f"model: {model_name}")
            if LAST_MODEL_NAME == model_name:
                model = LAST_MODEL
            else:
                model_load = models.get(model_name)
                if model_load is None:
                    nc.providers.task_processing.report_result(
                        task["id"], None, "Requested model is not available"
                    )
                    continue
                model = model_load()
                LAST_MODEL_NAME = model_name
                LAST_MODEL = model

            LOGGER.info("generating transcription")
            time_start = perf_counter()
            file_name = get_file(nc, task["id"], task.get("input").get('input'))
            segments, info = model.transcribe(file_name)
            transcript = ''
            for segment in segments:
                transcript += segment.text
                percentage = ( segment.start / info.duration ) * 100
                nc.providers.task_processing.set_progress(task['id'], percentage)
            del model
            LOGGER.info(f"transcription generated: {perf_counter() - time_start}s")

            nc.providers.task_processing.report_result(
                task["id"],
                {'output': str(transcript)},
            )
        except Exception as e:  # noqa
            try:
                LOGGER.error(str(e) + "\n" + "".join(traceback.format_exception(e)))
                nc.providers.task_processing.report_result(task["id"], None, str(e))
            except:
                pass
        finally:
            if 'file_name' in locals() and os.path.exists(file_name):
                os.remove(file_name)


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    global ENABLED

    if enabled is True:
        ENABLED.set()
        LOGGER.info("Hello from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.register(TaskProcessingProvider(
                id=f'stt_whisper2:{model_name}',
                name='Nextcloud Local Speech-To-Text Whisper: '+model_name,
                task_type='core:audio2text',
                expected_runtime=120,
            ))
    else:
        ENABLED.clear()
        LOGGER.info("Bye bye from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.unregister(f'stt_whisper2:{model_name}', True)
    return ""


def trigger_handler(providerId: str):
    global TRIGGER
    TRIGGER.set()

# Waits for `interval` seconds or WAIT_INTERVAL
# In case a TRIGGER event comes in, WAIT_INTERVAL is set (increased) to WAIT_INTERVAL_WITH_TRIGGER
def wait_for_task(interval = None):
    global WAIT_INTERVAL
    global WAIT_INTERVAL_WITH_TRIGGER
    if interval is None:
        interval = WAIT_INTERVAL
    if TRIGGER.wait(timeout=interval):
        WAIT_INTERVAL = WAIT_INTERVAL_WITH_TRIGGER
    TRIGGER.clear()

if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
