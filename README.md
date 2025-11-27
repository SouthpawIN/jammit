# Jammit: Infinite AI Music Radio

A command-line tool for infinite AI music generation and playback, powered by the ACE-Step model. Generate and queue continuous music based on text prompts, all operating silently in the background. Perfect for creating endless ambient soundscapes or themed radio stations.

## Features

*   **Infinite Music Generation:** Continuously generates new music based on your prompt.
*   **Automatic Playback & Queuing:** Generated tracks are automatically played and queued for a seamless listening experience.
*   **Absolutely Silent Operation:** No console output during generation or playback (except for errors that bypass aggressive suppression).
*   **Graceful Shutdown:** Easily stop all associated processes with `Ctrl+C`.
*   **Customizable Parameters:** Adjust generation parameters like duration, inference steps, and guidance scale.
*   **Defaults to Infinite Mode:** Designed as an "infinite radio" out-of-the-box.

## Installation

Follow these steps to set up Jammit on your system.

### 1. Clone the Repository

First, clone this repository to your local machine:

```bash
git clone https://github.com/your-username/jammit.git # Replace with your actual GitHub URL
cd jammit
```

### 2. Install System Dependencies

Jammit uses `aplay` for audio playback. Install it via your system's package manager:

```bash
sudo apt-get update
sudo apt-get install -y alsa-utils
```

### 3. Set Up Python Environment

It's recommended to use a virtual environment for Python projects.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Note: If you encounter issues with `torch` installation, please refer to the official PyTorch website for specific CUDA/CPU installation instructions based on your system.)*

### 4. ACE-Step Model Download

The ACE-Step model will be downloaded automatically the first time you run Jammit if it's not found locally. This might take some time depending on your internet connection.

Alternatively, you can pre-download it manually:
*(Instructions for manual pre-download will depend on the model host, typically Hugging Face. For ACE-Step/ACE-Step-v1-3.5B, you would typically use `huggingface-cli download` or clone the repository.)*
*Self-correction: The model is located in `Models/ACE-Step`. The pipeline itself handles the download using `snapshot_download`. So the current instruction is sufficient. I will just mention the expected download location for transparency.*

The model files will be stored in `~/.cache/ace-step/checkpoints` or `./Models/ACE-Step` relative to your project root, depending on internal logic.

### 5. Make `jammit` Executable and Add to PATH

To run Jammit easily from any directory, make the main script executable and create a symbolic link in your system's PATH:

```bash
chmod +x jammit.py
sudo ln -s $(pwd)/jammit.py /usr/local/bin/jammit
```
*(You may need to log out and back in or run `source ~/.bashrc` / `source ~/.zshrc` for the PATH changes to take effect.)*

## Usage

Jammit is designed to be an infinite radio by default.

### Infinite Music Radio (Default)

To start an infinite music stream based on a prompt:

```bash
jammit "sad lofi rock, ambient and heavy"
```

The script will continuously generate and play music.

### Single Generation

To generate and play a single track:

```bash
jammit "upbeat synthwave" --infinite false
```

### Custom Parameters

You can customize generation parameters:

```bash
jammit "ambient drone, mysterious" --duration 120 --infer-steps 100 --guidance-scale 20.0 --infinite false
```

*   `--duration`: Duration of the song in seconds (default: 213).
*   `--infer-steps`: Number of inference steps (default: 60).
*   `--guidance-scale`: Guidance scale for generation (default: 15.0).
*   `--infinite`: Set to `false` for single generation (default is `true`).

### Stopping Infinite Mode

To stop the infinite music radio, press `Ctrl+C`. The current song will finish playing gracefully, and all background generation processes will terminate.

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is released under the Apache 2.0 License, inheriting from the ACE-Step project. See the `ACE-Step` directory for more details.
