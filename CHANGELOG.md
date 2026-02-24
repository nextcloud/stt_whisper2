# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [2.4.2] - 2026-02-24

### Fixed

- fix(get_file): Don't use delete_on_close with NamedTemporaryFile

## [2.4.1] - 2026-02-24

### Fixed

- fix(get_file): Do not delete temp file after writing and close file before returning path

## [2.4.0] - 2025-12-03

### New

- Ships with Whisper large-v3-turbo model (similar quality as large-v3, but 5-8 times faster)

### Fixed

* fix: Stream audio file to disk
* fix: Make main loop more defensive

## [2.3.0] - 2025-11-12

### New

 - Support Nextcloud 33
 - feat: Add support for taskprocessing trigger event to speed up task pickup
 - Feat: Report task progress to frontend

## [2.2.3] - 2025-10-12

### Fixed

- Pin nc_py_api dependency to v0.20.2 to prevent httpx error

## [2.2.2] - 2025-10-12

### Fixed

- Add missing models back

## [2.2.1] - 2025-07-22

### New

- Add support for nextcloud 32

## [2.2.0] - 2025-07-21

### New

- Add support for HaRP deploy method
- Do not offload model unless a different model is used in the next request

## [2.1.3] - 2025-03-21

### Fixed

- fix: removed the cycle in the cycle
- Fix: Makefile

## [2.1.2] - 2025-03-20

### Fixed

- Improved the registering process of the ExApp in the Nextcloud.
- Reduce requests to Nextcloud when the ExApp is disabled.

## [2.1.1] - 2025-02-25

### Fixed

- fix(logger): only send WARNING+ logs to nextcloud
- fix: Stay compatible with cudnn 8

## [2.1.0] - 2025-01-07

### New

- Support for NC 31
- New License: AGPL-3.0

## [2.0.1] - 2024-11-26

### Fixed

- Fixed NC 30 compatibility, no longer compatible with NC 29

## [2.0.0] - 2024-11-14

### New

- NVIDIA GPU support

### Breaking changes

- Ships with whisper large v3 instead of whisper large v2
- Bump CUDA version to 12.2.x

## [1.1.5] - 2024-08-09

### Fixed

- Fix model loading
- Fix: Not all CPUs support the prev compute type
