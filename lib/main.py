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
from nc_py_api import NextcloudException
import niquests
from niquests import RequestException
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

ENHANCED_SUFFIX = "enhanced"


def provider_id_for(model_name: str, variant: str | None = None) -> str:
    if variant:
        return f"stt_whisper2:{model_name}:{variant}"
    return f"stt_whisper2:{model_name}"


def parse_provider(provider: dict) -> tuple[str, str]:
    provider_id = provider.get("id")
    if not isinstance(provider_id, str) or ":" not in provider_id:
        provider_id = provider.get("name")

    if isinstance(provider_id, str) and provider_id.startswith("stt_whisper2:"):
        parts = provider_id.split(":")
        if len(parts) >= 2 and parts[1]:
            model_name = parts[1]
            if len(parts) >= 3 and parts[2] == ENHANCED_SUFFIX:
                return model_name, ENHANCED_SUFFIX
            return model_name, "normal"

    raise ValueError(f"Invalid provider: {provider!r}")


def schedule_reformulation_and_wait(nc: NextcloudApp, transcript: str) -> str:
    if transcript.strip() == "":
        return transcript
    try:
        data = nc.ocs(
            "POST",
            "/ocs/v1.php/taskprocessing/schedule?format=json",
            headers={"OCS-APIRequest": "true"},
            json={
                "input": {"input": transcript},
                "type": "core:text2text:reformatparagraphs",
                "appId": os.environ["APP_ID"],
            },
        )
    except RequestException as e:
        raise RuntimeError(f"Failed to schedule reformulation task: {e}") from e

    task_id = data.get("task", {}).get("id")

    if not isinstance(task_id, int):
        raise RuntimeError(f"Unexpected schedule response: {data!r}")

    task = {"id": task_id, "status": "STATUS_SCHEDULED", "output": None}
    i = 0
    while (
        task.get("status") != "STATUS_SUCCESSFUL"
        and task.get("status") != "STATUS_FAILED"
        and i < 60 * 6
    ):
        if i < 60 * 3:
            sleep(5)
            i += 1
        else:
            # poll every 10 secs in the second half
            sleep(10)
            i += 2

        try:
            response = nc.ocs("GET", f"/ocs/v1.php/taskprocessing/task/{task_id}")
        except (
            niquests.exceptions.ConnectionError,
            niquests.exceptions.Timeout,
        ) as e:
            LOGGER.warning("Ignored error during task polling", exc_info=e)
            sleep(5)
            i += 1
            continue
        except NextcloudException as e:
            if e.status_code == niquests.codes.too_many_requests:  # pyright: ignore[reportAttributeAccessIssue]
                LOGGER.warning("Rate limited during task polling, waiting 10s before retrying")
                sleep(10)
                i += 2
                continue
            raise RuntimeError("Failed to poll Nextcloud TaskProcessing task") from e

        task = (response or {}).get("task", task)
        LOGGER.debug(f"Task poll ({i * 5}s) response: {task}")

    if task.get("status") == "STATUS_SUCCESSFUL":
        output = (task.get("output") or {}).get("output")
        if isinstance(output, str) and output.strip():
            return output
        raise RuntimeError(f"Reformulation returned empty output: {task!r}")
    if task.get("status") == "STATUS_FAILED":
        raise RuntimeError(f"Reformulation failed: {task!r}")
    raise RuntimeError("Reformulation timed out")


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

        provider_ids = []
        for model_name, _ in models.items():
            provider_ids.append(provider_id_for(model_name))
            provider_ids.append(provider_id_for(model_name, ENHANCED_SUFFIX))

        try:
            item = nc.providers.task_processing.next_task(provider_ids, ["core:audio2text"])
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
            nc.set_user(task["userId"])
            provider = item.get("provider")
            if provider is None:
                raise ValueError('Next task endpoint did not provide a provider name')
            model_name, variant = parse_provider(provider)
            LOGGER.info(f"model: {model_name} variant: {variant}")
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

            enhanced = variant == ENHANCED_SUFFIX
            LOGGER.info("generating transcription")
            time_start = perf_counter()
            file_name = get_file(nc, task["id"], task.get("input").get('input'))
            segments, info = model.transcribe(file_name)
            transcript = ''
            for segment in segments:
                transcript += segment.text
                percentage = ( segment.start / info.duration ) * 100
                if enhanced:
                    percentage /= 2
                nc.providers.task_processing.set_progress(task['id'], percentage)
            del model
            LOGGER.info(f"transcription generated: {perf_counter() - time_start}s")

            if enhanced:
                nc.providers.task_processing.set_progress(task["id"], 50)
                try:
                    LOGGER.info("Creating enhanced version of transcript")
                    transcript = schedule_reformulation_and_wait(nc, transcript)
                    LOGGER.info("Enhanced version of transcript created")
                except Exception as e:
                    LOGGER.error(f"Enhanced transcription failed with error: {str(e)}\n{''.join(traceback.format_exception(e))}. Using raw transcript instead.")
               

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

    major = (await nc.srv_version).get("major")
    supports_enhanced = major >= 34

    if enabled is True:
        ENABLED.set()
        LOGGER.info("Hello from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.register(TaskProcessingProvider(
                id=provider_id_for(model_name),
                name='Nextcloud Local Speech-To-Text Whisper: '+model_name,
                task_type='core:audio2text',
                expected_runtime=120,
            ))
            if supports_enhanced:
                await nc.providers.task_processing.register(TaskProcessingProvider(
                    id=provider_id_for(model_name, ENHANCED_SUFFIX),
                    name='Nextcloud Local Speech-To-Text Whisper: '+model_name+' (enhanced)',
                    task_type='core:audio2text',
                    expected_runtime=240,
                ))
    else:
        ENABLED.clear()
        LOGGER.info("Bye bye from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.unregister(provider_id_for(model_name), True)
            if supports_enhanced:
                await nc.providers.task_processing.unregister(
                    provider_id_for(model_name, ENHANCED_SUFFIX),
                    True,
                )
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
