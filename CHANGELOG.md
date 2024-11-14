# Change Log

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

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