import threading
import traceback
from contextlib import asynccontextmanager
from threading import Event
from time import perf_counter, sleep
from io import StringIO
import os
from time import gmtime, strftime
from math import floor, modf
import logging
from pathlib import Path  # noqa
import xml.etree.ElementTree as ET  # noqa

import niquests
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

from nc_py_api.ex_app.providers.task_processing import (
    ShapeDescriptor,
    ShapeEnumValue,
    ShapeType,
    TaskProcessingProvider,
)
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

# --- VAD configuration (Silero, bundled with faster-whisper) ---
# Enabled by default; disable by setting STT_WHISPER2_VAD_FILTER=0 in the container env.
# Tunables map 1:1 onto faster_whisper.vad.VadOptions.
VAD_FILTER = os.environ.get("STT_WHISPER2_VAD_FILTER", "1") == "1"
try:
    VAD_PARAMETERS = {
        "threshold": float(os.environ.get("STT_WHISPER2_VAD_THRESHOLD", "0.5")),
        "min_speech_duration_ms": int(os.environ.get("STT_WHISPER2_VAD_MIN_SPEECH_MS", "0")),
        "min_silence_duration_ms": int(os.environ.get("STT_WHISPER2_VAD_MIN_SILENCE_MS", "2000")),
        "speech_pad_ms": int(os.environ.get("STT_WHISPER2_VAD_SPEECH_PAD_MS", "400")),
    }
except:
    raise Exception('Failed to parse VAD settings. All values must be valid numbers')


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



def provider_id_for(model_name: str, enhanced: bool = False, subtitles: bool = False) -> str:
    if subtitles:
        return f"stt_whisper2_subtitles:{model_name}"
    if enhanced:
        return f"stt_whisper2_enhanced:{model_name}"
    return f"stt_whisper2:{model_name}"


def parse_provider(provider: dict) -> tuple[str, bool, bool]:
    provider_id = provider.get("id")
    if not isinstance(provider_id, str) or ":" not in provider_id:
        provider_id = provider.get("name")

    if isinstance(provider_id, str) and provider_id.startswith(
        "stt_whisper2_enhanced:"
    ):
        model_name = provider_id.split(":", 1)[1]
        if model_name:
            return model_name, True, False

    if isinstance(provider_id, str) and provider_id.startswith(
        "stt_whisper2_subtitles:"
    ):
        model_name = provider_id.split(":", 1)[1]
        if model_name:
            return model_name, False, True

    if isinstance(provider_id, str) and provider_id.startswith("stt_whisper2:"):
        model_name = provider_id.split(":", 1)[1]
        if model_name:
            return model_name, False, False

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

def build_transcript_text(segments, info, task_id, nc, enhanced) -> str:
    transcript = ''
    for segment in segments:
        transcript += segment.text
        percentage = ( segment.start / info.duration ) * 100
        if enhanced:
            percentage /= 2
        nc.providers.task_processing.set_progress(task_id, percentage)

    return transcript

def build_transcript_srt(segments, info, task_id, nc) -> str:
    transcript = ''
    i = 0
    for segment in segments:
        i += 1
        start_frac, start_int = modf(segment.start)
        start_ms = floor(start_frac * 1000.0)
        start = strftime('%H:%M:%S', gmtime(start_int))
        end_frac, end_int = modf(segment.end)
        end_ms = floor(end_frac * 1000.0)
        end = strftime('%H:%M:%S', gmtime(end_int))

        transcript += f'{i}\n{start},{start_ms:03d} --> {end},{end_ms:03d}\n{segment.text}\n\n'
        percentage = ( segment.start / info.duration ) * 100
        nc.providers.task_processing.set_progress(task_id, percentage)

    return transcript

def build_transcript_vtt(segments, info, task_id, nc) -> str:
    transcript = 'WEBVTT\n\n'
    for segment in segments:
        start_frac, start_int = modf(segment.start)
        start_ms = floor(start_frac * 1000.0)
        start = strftime('%H:%M:%S', gmtime(start_int))
        end_frac, end_int = modf(segment.end)
        end_ms = floor(end_frac * 1000.0)
        end = strftime('%H:%M:%S', gmtime(end_int))

        transcript += f'{start}.{start_ms:03d} --> {end}.{end_ms:03d}\n{segment.text}\n\n'
        percentage = ( segment.start / info.duration ) * 100
        nc.providers.task_processing.set_progress(task_id, percentage)

    return transcript

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
            provider_ids.append(provider_id_for(model_name, enhanced=True))
            provider_ids.append(provider_id_for(model_name, subtitles=True))

        try:
            item = nc.providers.task_processing.next_task(provider_ids, ["core:audio2text", "core:audio2text:subtitles"])
            if not isinstance(item, dict):
                wait_for_task()
                continue
            task = item.get("task")
            if task is None:
                wait_for_task()
                continue
        except (
                niquests.exceptions.ConnectionError,
                niquests.exceptions.Timeout,
        ) as e:
            LOGGER.info('Temporary error fetching next tasks, will retry:', exc_info=e)
            wait_for_task(5)
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
            model_name, enhanced, subtitles = parse_provider(provider)
            LOGGER.info(f"model: {model_name} enhanced: {enhanced}")
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
            segments, info = model.transcribe(
                file_name,
                vad_filter=VAD_FILTER,
                vad_parameters=VAD_PARAMETERS if VAD_FILTER else None,
            )

            if subtitles:
                if task['input']['format'] == 'vtt':
                    transcript = build_transcript_vtt(segments, info, task['id'], nc)
                else:
                    transcript = build_transcript_srt(segments, info, task['id'], nc)
            else:
                transcript = build_transcript_text(segments, info, task['id'], nc, enhanced)
            del model
            LOGGER.info(f"transcription generated: {perf_counter() - time_start}s")

            if enhanced:
                if task.get("userId") is None:
                    LOGGER.warning("User ID is not set for the task skipping enhanced transcription")
                else:
                    nc.set_user(task["userId"])
                    nc.providers.task_processing.set_progress(task["id"], 50)
                    try:
                        LOGGER.info("Creating enhanced version of transcript")
                        transcript = schedule_reformulation_and_wait(nc, transcript)
                        LOGGER.info("Enhanced version of transcript created")
                    except Exception as e:
                        LOGGER.error(f"Enhanced transcription failed with error: {str(e)}\n{''.join(traceback.format_exception(e))}. Using raw transcript instead.")

            if subtitles:
                file_id = nc.providers.task_processing.upload_result_file(
                    task["id"],
                    StringIO(transcript),
                )
                nc.providers.task_processing.report_result(
                    task["id"],
                    {'output': file_id},
                )
            else:
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
    supports_subtitles = major >= 35

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
                    id=provider_id_for(model_name, enhanced=True),
                    name='Nextcloud Local Speech-To-Text Whisper: '+model_name+' (enhanced)',
                    task_type='core:audio2text',
                    expected_runtime=240,
                ))

            if supports_subtitles:
                optional_input_shape = [
                    ShapeDescriptor(
                        name='format',
                        description='The format of the subtitles file',
                        shape_type=ShapeType.ENUM,
                    ),
                ]
                optional_input_values = {
                    'format': [
                        ShapeEnumValue(name='SubRip Text', value='srt'),
                        ShapeEnumValue(name='WebVTT', value='vtt'),
                    ],
                }
                optional_input_defaults = {
                    'format': 'srt',
                }
                await nc.providers.task_processing.register(TaskProcessingProvider(
                    id=provider_id_for(model_name, subtitles=True),
                    name='Nextcloud Local Speech-To-Text Whisper: '+model_name,
                    task_type='core:audio2text:subtitles',
                    expected_runtime=120,
                    optional_input_shape=optional_input_shape,
                    optional_input_shape_enum_values=optional_input_values,
                    optional_input_shape_defaults=optional_input_defaults,
                ))
    else:
        ENABLED.clear()
        LOGGER.info("Bye bye from %s", nc.app_cfg.app_name)
        for model_name, _ in models.items():
            await nc.providers.task_processing.unregister(provider_id_for(model_name), True)
            if supports_enhanced:
                await nc.providers.task_processing.unregister(
                    provider_id_for(model_name, enhanced=True),
                    True,
                )
            if supports_subtitles:
                await nc.providers.task_processing.unregister(
                    provider_id_for(model_name, subtitles=True),
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
