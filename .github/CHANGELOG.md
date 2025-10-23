# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Planned features go here

### Changed

- Planned changes go here

### Fixed

- Planned fixes go here

## [2.0.0] - 2025-10-23

### Added

- Complete refactor with TensorFlow 2.x support
- Webcam-based demo for testing without drone
- Comprehensive unit test suite with >90% coverage
- Modern Python packaging (setup.py + pyproject.toml)
- Modular architecture with clean abstractions
  - Abstract camera interface
  - Separate webcam and Tello implementations
  - Collision avoidance as standalone module
- Configuration management system
- Multiple depth map visualization colormaps
- Real-time FPS tracking
- Command-line arguments for all demos
- Safety features for drone operation
  - Commands disabled by default
  - Explicit confirmation for autonomous mode
  - Emergency stop functionality
  - Battery monitoring
- Comprehensive documentation
  - README with installation and usage guides
  - CODE_OF_CONDUCT.md
  - CONTRIBUTING.md
  - SECURITY.md with safety guidelines
  - Detailed docstrings throughout codebase
- Development tools
  - Makefile with common commands
  - GitHub Actions CI/CD pipeline
  - Pre-commit hooks configuration
  - Code formatting with black
  - Linting with flake8 and pylint

### Changed

- **BREAKING**: Migrated from TensorFlow 1.8 to TensorFlow 2.x
- **BREAKING**: Replaced deprecated `tellopy` with `djitellopy`
- **BREAKING**: New project structure with `src/` package
- Improved depth estimation with better model architecture
- Enhanced collision avoidance algorithm
- Better error handling and logging
- Optimized performance for both CPU and GPU
- Updated dependencies to latest stable versions

### Fixed

- Memory leaks in continuous inference
- Thread safety issues with video stream
- Incorrect depth map scaling
- Camera initialization failures
- Various edge cases in collision detection

### Removed

- TensorFlow 1.x support
- Legacy `tellopy` library support
- Deprecated training code (moved to separate branch)
- Old monolithic script structure

### Security

- Added security policy and vulnerability reporting guidelines
- Improved input validation
- Added safety checks for drone commands
- Network security recommendations

## [1.0.0] - 2020-XX-XX

### Added

- Initial release with PyDNet depth estimation
- Basic DJI Tello integration using tellopy
- Simple collision avoidance demonstration
- TensorFlow 1.8 implementation
- Basic README documentation

---

## Version History Summary

- **2.0.0** (2025-10-23): Major refactor with modern stack
- **1.0.0** (2020): Initial release

## Migration Guide

### From 1.0.0 to 2.0.0

**Breaking Changes:**

1. **TensorFlow Version**

   ```bash
   # Old (1.x)
   pip install tensorflow==1.8

   # New (2.x)
   pip install tensorflow>=2.10.0
   ```

2. **Tello Library**

   ```bash
   # Old
   pip install tellopy

   # New
   pip install djitellopy
   ```

3. **Import Paths**

   ```python
   # Old
   from pydnet import pydnet
   from tello_pydnet_interface import TelloCV

   # New
   from src.depth_estimator import DepthEstimator
   from src.tello_source import TelloSource
   ```

4. **Usage**

   ```bash
   # Old
   python3 tello_pydnet_interface.py --checkpoint_dir checkpoint/IROS18/pydnet --resolution 1

   # New
   python tello_demo.py --checkpoint_dir checkpoint/IROS18/pydnet --resolution 1
   ```

**New Features to Try:**

- Test with webcam: `python webcam_demo.py`
- Use configuration system instead of command-line args
- Run comprehensive tests: `make test`
- Check out collision avoidance analysis mode

**Deprecation Notices:**

- Support for Python <3.8 will be dropped in v2.1.0
- TensorFlow <2.10 support will be dropped in v2.1.0

---

[Unreleased]: https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet/compare/v1.0.0...v2.0.0
[1.0.0]: https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet/releases/tag/v1.0.0
