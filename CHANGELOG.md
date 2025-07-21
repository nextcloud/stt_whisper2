# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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
