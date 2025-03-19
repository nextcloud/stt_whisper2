import threading
from contextlib import asynccontextmanager
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
from ocs import ocs

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
ENABLED_FLAG = NextcloudApp().enabled_state


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


models = load_models()

@asynccontextmanager
async def lifespan(_app: FastAPI):
    setup_nextcloud_logging("stt_whisper2", logging_level=logging.WARNING)
    set_handlers(
        APP,
        enabled_handler,
    )
    t = BackgroundProcessTask()
    t.start()
    yield


APP = FastAPI(lifespan=lifespan)

def get_file(nc, task_id, file_id):
    return ocs(nc._session, 'GET',f"/ocs/v2.php/taskprocessing/tasks_provider/{task_id}/file/{file_id}")


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        global ENABLED_FLAG

        nc = NextcloudApp()
        while True:
            if not ENABLED_FLAG:
                sleep(30)
                ENABLED_FLAG = nc.enabled_state
                continue

            try:
                next = nc.providers.task_processing.next_task([f'stt_whisper2:{model_name}' for model_name, _ in models.items()], ['core:audio2text'])
                if not 'task' in next or next is None:
                    sleep(5)
                    continue
                task = next.get('task')
            except Exception as e:
                LOGGER.error(str(e))
                sleep(5)
                continue
            try:
                LOGGER.info(f"Next task: {task['id']}")
                model_name = next.get("provider").get('name').split(':', 2)[1]
                LOGGER.info( f"model: {model_name}")
                model_load = models.get(model_name)
                if model_load is None:
                    NextcloudApp().providers.task_processing.report_result(
                        task["id"], None, "Requested model is not available"
                    )
                    continue
                model = model_load()

                LOGGER.info("generating transcription")
                time_start = perf_counter()
                file_name = get_file(nc, task["id"], task.get("input").get('input'))
                segments, _ = model.transcribe(file_name)
                del model
                LOGGER.info(f"transcription generated: {perf_counter() - time_start}s")

                transcript = ''
                for segment in segments:
                    transcript += segment.text
                NextcloudApp().providers.task_processing.report_result(
                    task["id"],
                    {'output': str(transcript)},
                )
            except Exception as e:  # noqa
                try:
                    LOGGER.error(str(e))
                    nc.providers.task_processing.report_result(task["id"], None, str(e))
                except:
                    pass


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    global ENABLED_FLAG

    ENABLED_FLAG = enabled
    if enabled is True:
        LOGGER.info("Hello from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.register(TaskProcessingProvider(
                id=f'stt_whisper2:{model_name}',
                name='Nextcloud Local Speech-To-Text Whisper: '+model_name,
                task_type='core:audio2text',
                expected_runtime=120,
            ))
    else:
        LOGGER.info("Bye bye from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.unregister(f'stt_whisper2:{model_name}', True)
    return ""


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
