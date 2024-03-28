"""Tha main module of the llm2 app
"""

import queue
import threading
import tempfile
import typing
from contextlib import asynccontextmanager
from time import perf_counter
import os

from fastapi import Depends, FastAPI, UploadFile, responses
from nc_py_api import AsyncNextcloudApp, NextcloudApp
from nc_py_api.ex_app import LogLvl, anc_app, run_app, set_handlers, persistent_storage
from faster_whisper import WhisperModel


def load_models():
    models = {}
    dir_path = os.path.dirname(os.path.realpath(__file__))
    for file in os.scandir(dir_path + "/../models/"):
        if os.path.isdir(file.path):
            models[file.name] = lambda: WhisperModel(file.path, device="cpu", compute_type="int8_float16")
    for file in os.scandir(persistent_storage()):
        if os.path.isdir(file.path):
            models[file.name] = lambda: WhisperModel(file.path, device="cpu", compute_type="int8_float16")

    return models


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


APP = FastAPI(lifespan=lifespan)
TASK_LIST: queue.Queue = queue.Queue(maxsize=100)


class BackgroundProcessTask(threading.Thread):
    def run(self, *args, **kwargs):  # pylint: disable=unused-argument
        while True:
            task = TASK_LIST.get(block=True)
            try:
                model_name = task.get("model")
                print(f"model: {model_name}")
                model_load = models.get(model_name)
                if model_load is None:
                    NextcloudApp().providers.speech_to_text.report_result(
                        task["id"], error="Requested model is not available"
                    )
                    continue
                model = model_load()
                print("generating transcription")
                time_start = perf_counter()
                with task.get("file") as tmp:
                    segments, _ = model.transcribe(tmp.name)
                del model
                print(f"transcription generated: {perf_counter() - time_start}s")
                transcript = ''
                for segment in segments:
                    transcript += segment.text
                NextcloudApp().providers.speech_to_text.report_result(
                    task["id"],
                    str(transcript),
                )
            except Exception as e:  # noqa
                print(str(e))
                nc = NextcloudApp()
                nc.log(LogLvl.ERROR, str(e))
                nc.providers.speech_to_text.report_result(task["id"], error=str(e))



@APP.post("/model/{model_name}")
async def tiny_llama(
    _nc: typing.Annotated[AsyncNextcloudApp, Depends(anc_app)],
    data: UploadFile,
    task_id: int,
    model_name=None,
):
    _, file_extension = os.path.splitext(data.filename)
    task_file = tempfile.NamedTemporaryFile(mode="w+b", suffix=f"{file_extension}")
    task_file.write(await data.read())
    try:
        TASK_LIST.put({"file": task_file, "id": task_id, "model": model_name}, block=False)
    except queue.Full:
        return responses.JSONResponse(content={"error": "task queue is full"}, status_code=429)
    return responses.Response()


async def enabled_handler(enabled: bool, nc: AsyncNextcloudApp) -> str:
    print(f"enabled={enabled}")
    if enabled is True:
        for model_name, _ in models.items():
            await nc.providers.speech_to_text.register('stt_whisper2:'+model_name, "Local Whisper Speech To text: " + model_name, '/model/'+model_name)
    else:
        for model_name, _ in models.items():
            await nc.providers.speech_to_text.unregister('stt_whisper2:'+model_name)
    return ""


if __name__ == "__main__":
    run_app("main:APP", log_level="trace")
