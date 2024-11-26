import threading
from contextlib import asynccontextmanager
from time import perf_counter, sleep
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from fastapi import FastAPI
from faster_whisper import WhisperModel
from nc_py_api import AsyncNextcloudApp, NextcloudApp
from nc_py_api.ex_app import (
    get_computation_device,
    LogLvl,
    persistent_storage,
    run_app,
    set_handlers,
)

from nc_py_api.ex_app.providers.task_processing import TaskProcessingProvider
from ocs import ocs

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
    set_handlers(
        APP,
        enabled_handler,
    )
    t = BackgroundProcessTask()
    t.start()
    yield


def log(nc, level, content):
    logger.log((level+1)*10, content)
    try:
        nc.log(level, content)
    except:
        pass

APP = FastAPI(lifespan=lifespan)

def get_file(nc, task_id, file_id):
    return ocs(nc._session, 'GET',f"/ocs/v2.php/taskprocessing/tasks_provider/{task_id}/file/{file_id}")


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        nc = NextcloudApp()
        while True:
            try:
                next = nc.providers.task_processing.next_task([f'stt_whisper2:{model_name}' for model_name, _ in models.items()], ['core:audio2text'])
                if not 'task' in next or next is None:
                    sleep(5)
                    continue
                task = next.get('task')
            except Exception as e:
                print(str(e))
                log(nc, LogLvl.ERROR, str(e))
                sleep(5)
                continue
            try:
                log(nc, LogLvl.INFO, f"Next task: {task['id']}")
                model_name = next.get("provider").get('name').split(':', 2)[1]
                log(nc, LogLvl.INFO, f"model: {model_name}")
                model_load = models.get(model_name)
                if model_load is None:
                    NextcloudApp().providers.task_processing.report_result(
                        task["id"], None, "Requested model is not available"
                    )
                    continue
                model = model_load()

                log(nc, LogLvl.INFO, "generating transcription")
                time_start = perf_counter()
                file_name = get_file(nc, task["id"], task.get("input").get('input'))
                segments, _ = model.transcribe(file_name)
                del model
                log(nc, LogLvl.INFO, f"transcription generated: {perf_counter() - time_start}s")

                transcript = ''
                for segment in segments:
                    transcript += segment.text
                NextcloudApp().providers.task_processing.report_result(
                    task["id"],
                    {'output': str(transcript)},
                )
            except Exception as e:  # noqa
                print(str(e))
                try:
                    log(nc, LogLvl.ERROR, str(e))
                    nc.providers.task_processing.report_result(task["id"], None, str(e))
                except:
                    pass


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    print(f"enabled={enabled}")
    if enabled is True:
        for model_name, _ in models.items():
            await nc.providers.task_processing.register(TaskProcessingProvider(
                id=f'stt_whisper2:{model_name}',
                name='Nextcloud Local Speech-To-Text Whisper: '+model_name,
                task_type='core:audio2text',
                expected_runtime=120,
            ))
    else:
        for model_name, _ in models.items():
            await nc.providers.task_processing.unregister(f'stt_whisper2:{model_name}', True)
    return ""


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
