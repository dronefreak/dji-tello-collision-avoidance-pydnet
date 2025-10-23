# DJI Tello Collision Avoidance with PyDNet

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Real-time monocular depth estimation for DJI Tello drone collision avoidance using PyDNet. Based on the paper ["Towards real-time unsupervised monocular depth estimation on CPU"](https://arxiv.org/abs/1806.11430) (IROS 2018).

**Major Update v2.0**: Complete refactor with TensorFlow 2.x, webcam support, comprehensive tests, and modern Python packaging.

## Features

- 🚁 **Tello Drone Integration**: Real-time depth estimation on DJI Tello video stream
- 📷 **Webcam Support**: Test without a drone using any USB/built-in webcam
- 🧠 **PyDNet Depth Estimation**: Fast monocular depth estimation optimized for CPU
- 🎯 **Collision Avoidance**: Autonomous navigation based on depth maps
- 🔧 **Modern Architecture**: Clean, modular codebase with TF2.x
- ✅ **Comprehensive Tests**: Full unit test coverage with mocks
- 📊 **Real-time Visualization**: Depth maps with configurable colormaps
- ⚙️ **Flexible Configuration**: Easy-to-use config system

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster inference

### Basic Installation

```bash
# Clone the repository
git clone https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet.git
cd dji-tello-collision-avoidance-pydnet

# Create virtual environment
python -m venv venv

# Activate it
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt

# Or install as package
pip install -e .
```

## Quick Start

### 1. Webcam Demo (No Drone Required)

Test depth estimation with your webcam:

```bash
python webcam_demo.py --camera_id 0 --colormap plasma
```

**Controls:**

- `q` or `ESC` - Quit
- `p` - Pause/Resume
- `s` - Save screenshot
- `a` - Toggle collision avoidance analysis

### 2. Tello Drone Demo

**Important**: Commands are disabled by default for safety.

```bash
# Connect to Tello WiFi first, then:
python tello_demo.py
```

**Controls:**

- `t` - Takeoff
- `l` - Land
- `e` - Emergency stop
- `q` - Quit

**Enable collision avoidance** (use with caution):

```bash
python tello_demo.py --enable_commands --enable_auto_flight
```

## Model Weights

Download pretrained PyDNet weights:

```bash
# Create checkpoint directory
mkdir -p checkpoint/IROS18/pydnet

# Download weights (link TBD - see original PyDNet repo)
# Place the checkpoint files in: checkpoint/IROS18/pydnet/
```

The code will run without weights (for testing), but results will be random.

## Project Structure

```
.
├── src/
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration management
│   ├── depth_estimator.py       # PyDNet depth estimation
│   ├── utils.py                 # Utility functions
│   ├── camera_interface.py      # Abstract camera interface
│   ├── webcam_source.py         # Webcam implementation
│   ├── tello_source.py          # Tello drone implementation
│   └── collision_avoidance.py   # Navigation logic
├── tests/                       # Unit tests
│   ├── test_depth_estimator.py
│   ├── test_utils.py
│   ├── test_webcam_demo.py
│   ├── test_tello_interface.py
│   └── test_collision_avoidance.py
├── webcam_demo.py               # Webcam demo script
├── tello_demo.py                # Tello demo script
├── requirements.txt             # Dependencies
├── README.md                    # This file
├── CODE_OF_CONDUCT.md          # Community guidelines
├── LICENSE                      # Apache 2.0 License
└── SECURITY.md                  # Security policy
```

## Usage Examples

### Basic Depth Estimation

```python
from src.config import Config
from src.depth_estimator import DepthEstimator
from src.webcam_source import WebcamSource
from src.utils import preprocess_image

# Setup
config = Config()
estimator = DepthEstimator(config)
camera = WebcamSource(config)

# Open camera
camera.open()

# Capture and estimate depth
success, frame = camera.read()
input_img = preprocess_image(frame, (256, 512), normalize=True)
depth_map = estimator.predict(input_img)

# Cleanup
camera.release()
```

### Collision Avoidance

```python
from src.collision_avoidance import CollisionAvoidance

ca = CollisionAvoidance(config)

# Analyze depth map
analysis = ca.analyze_depth(depth_map)
print(f"Suggested action: {analysis['suggested_action']}")
print(f"Safe to move forward: {analysis['is_safe']}")

# Get RC command for drone
left_right, forward_backward, up_down, yaw = ca.get_rc_command(depth_map)
```

### Custom Configuration

```python
from src.config import Config

config = Config(
    input_width=640,
    input_height=320,
    resolution=1,  # 1=High, 2=Quarter, 3=Eighth
    colormap='viridis',
    use_gpu=True,
    camera_id=0,
    min_safe_depth=0.4,
    tello_enable_commands=False  # Safety first!
)
```

## Configuration Options

Key configuration parameters in `src/config.py`:

| Parameter                    | Default  | Description                  |
| ---------------------------- | -------- | ---------------------------- |
| `input_width`                | 512      | Model input width            |
| `input_height`               | 256      | Model input height           |
| `resolution`                 | 1        | Output resolution (1/2/3)    |
| `colormap`                   | 'plasma' | Depth visualization colormap |
| `use_gpu`                    | True     | Enable GPU acceleration      |
| `camera_id`                  | 0        | Webcam device ID             |
| `min_safe_depth`             | 0.3      | Obstacle detection threshold |
| `tello_enable_commands`      | False    | Enable drone commands        |
| `enable_collision_avoidance` | False    | Enable autonomous navigation |

## Testing

Run the test suite:

```bash
# Run all tests
python -m unittest discover tests/

# Run specific test file
python -m unittest tests/test_depth_estimator.py

# Run with verbose output
python -m unittest discover tests/ -v
```

## Performance

Approximate inference times on different hardware:

| Hardware        | Resolution     | FPS    |
| --------------- | -------------- | ------ |
| Intel i7 (CPU)  | High (512x256) | ~15-20 |
| Intel i7 (CPU)  | Quarter        | ~30-40 |
| NVIDIA RTX 3060 | High (512x256) | ~60-80 |
| NVIDIA RTX 3060 | Quarter        | ~120+  |

_Results may vary based on system configuration_

## Safety Warning

⚠️ **IMPORTANT SAFETY NOTICE** ⚠️

- Autonomous flight features are **EXPERIMENTAL**
- Always test in a safe, open environment
- Keep emergency stop ready (`e` key)
- Start with commands **DISABLED**
- Never fly near people, animals, or fragile objects
- Check local drone regulations
- Battery level must be >20% for takeoff
- The developers are not responsible for any damage or injury

## Troubleshooting

### Webcam not detected

```bash
# List available cameras (Linux)
ls /dev/video*

# Try different camera IDs
python webcam_demo.py --camera_id 1
```

### Cannot connect to Tello

- Ensure Tello is powered on (flashing yellow light)
- Connect to Tello's WiFi network (TELLO-XXXXXX)
- Check that no other app is using the Tello
- Try restarting both Tello and computer

### Low FPS

- Use lower resolution: `--resolution 2` or `--resolution 3`
- Disable GPU if causing issues: `--no_gpu`
- Reduce input dimensions: `--width 256 --height 128`

### Import errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

## Contributing

Contributions are welcome! Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) first.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use this work, please cite the original PyDNet paper:

```bibtex
@inproceedings{pydnet18,
  title     = {Towards real-time unsupervised monocular depth estimation on CPU},
  author    = {Poggi, Matteo and
               Aleotti, Filippo and
               Tosi, Fabio and
               Mattoccia, Stefano},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2018}
}
```

## Acknowledgments

- Original [PyDNet](https://github.com/mattpoggi/pydnet) by Matteo Poggi
- [Monodepth](https://github.com/mrharicot/monodepth) framework by Clément Godard
- [DJITelloPy](https://github.com/damiafuentes/DJITelloPy) library

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

- **Author**: dronefreak
- **GitHub**: [@dronefreak](https://github.com/dronefreak)
- **Issues**: [GitHub Issues](https://github.com/dronefreak/dji-tello-collision-avoidance-pydnet/issues)

## Changelog

### v2.0.0 (2025)

- Complete refactor with TensorFlow 2.x
- Added webcam support for testing
- Comprehensive unit tests
- Modern Python packaging
- Improved documentation
- Safety features and command control
- Configurable parameters

### v1.0.0 (Original)

- Initial release with TensorFlow 1.8
- Basic Tello integration
- PyDNet depth estimation

---

**Star ⭐ this repo if you find it useful!**
