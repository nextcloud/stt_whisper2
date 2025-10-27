import tempfile
import typing
from json import loads

from niquests import Response, codes, RequestException
from nc_py_api import NextcloudException


def check_error(response: Response, info: str = ""):
	"""Checks HTTP code from Nextcloud, and raises exception in case of error.

	For the OCS and DAV `code` be code returned by HTTP and not the status from ``ocs_meta``.
	"""
	status_code = response.status_code
	if not info:
		info = f"request: {response.request.method} {response.request.url}"
	if 996 <= status_code <= 999:
		if status_code == 996:
			phrase = "Server error"
		elif status_code == 997:
			phrase = "Unauthorised"
		elif status_code == 998:
			phrase = "Not found"
		else:
			phrase = "Unknown error"
		raise NextcloudException(status_code, reason=phrase, info=info)
	if not codes.is_error(status_code):
		return
	raise NextcloudException(status_code, reason=codes(status_code).phrase, info=info)

def ocs(
		ncSession,
		method: str,
		path: str,
		*,
		content: bytes | str | typing.Iterable[bytes] | typing.AsyncIterable[bytes] | None = None,
		json: dict | list | None = None,
		params: dict | None = None,
		files: dict | None = None,
		**kwargs,
):
	ncSession.init_adapter()
	info = f"request: {method} {path}"
	nested_req = kwargs.pop("nested_req", False)
	response: Response = ncSession.adapter.request(
		method, path, content=content, json=json, params=params, files=files, **kwargs
	)
	if response.status_code >= 400:
		print(loads(response.text))
	check_error(response, info)
	if response.status_code == 204:  # NO_CONTENT
		return []
	# Create a temporary file
	with tempfile.NamedTemporaryFile(delete=False, mode='wb') as temp_file:
		temp_file.write(response.content)  # Write the response content to the temp file
		return temp_file.name  # Get the temp file's path