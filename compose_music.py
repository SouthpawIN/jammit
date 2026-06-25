"""
Compose music using ACE-Step v1.5 generation API.
"""
import sys
import os
import warnings
from loguru import logger
from tqdm import tqdm

# Silence everything
logger.remove()
logger.add(sys.stderr, level="CRITICAL")
tqdm.disable = True

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning)

# Add ACE-Step to path (via symlink ACE-Step -> ACE-Step-1.5 in jammit dir)
_ace_step_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'ACE-Step'))
if _ace_step_dir not in sys.path:
    sys.path.insert(0, _ace_step_dir)

import torch
import random
from acestep.handler import AceStepHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

# Global pipeline — initialized once
HANDLER = None
PROJECT_ROOT = _ace_step_dir  # ACE-Step-1.5 root for checkpoint/config resolution


def initialize_handler():
    """Initialize the ACE-Step v1.5 handler (once)."""
    global HANDLER
    if HANDLER is None:
        HANDLER = AceStepHandler()
        HANDLER.initialize_service(
            project_root=PROJECT_ROOT,
            config_path="acestep-v15-turbo",
            device="cuda" if torch.cuda.is_available() else "cpu",
        )


def compose_music(
    prompt="",
    lyrics="",
    instrumental=True,
    n_gen=1,
    duration_seconds=213,
    infer_steps=8,
    guidance_scale=7.0,
    scheduler_type="ode",
    cfg_type=None,          # ignored in v1.5, kept for backwards compat
    seed=None,
    save_path=None,
):
    """
    Generate music using ACE-Step v1.5.

    Args:
        prompt: Description of the music to generate.
        lyrics: Lyrics for the song (use "[instrumental]" for instrumental).
        instrumental: If True and no lyrics are provided, force instrumental tag.
        n_gen: Number of generations to create.
        duration_seconds: Duration in seconds (default 213 = ~3.5 min).
        infer_steps: Inference steps (default 8 for turbo, higher = better quality).
        guidance_scale: How closely to follow prompt (default 7.0).
        scheduler_type: Scheduler/inference method (default "ode").
        seed: Random seed (None = random).
        save_path: Output directory (default ~/jammed).

    Returns:
        list: Paths to the generated .wav files.
    """
    # Default save path
    if save_path is None:
        save_path = os.path.expanduser("~/jammed")

    initialize_handler()

    # Handle instrumental tag
    if instrumental and not lyrics:
        lyrics = "[instrumental]"

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_paths = []

    for i in range(n_gen):
        current_seed = seed + i

        params = GenerationParams(
            caption=prompt,
            lyrics=lyrics,
            instrumental=instrumental,
            duration=duration_seconds,
            inference_steps=infer_steps,
            guidance_scale=guidance_scale,
            infer_method=scheduler_type,
            seed=current_seed,
            thinking=False,  # faster generation, no CoT
        )

        config = GenerationConfig(
            batch_size=1,
            audio_format="wav",
            use_random_seed=False,
            seeds=[current_seed],
        )

        result = generate_music(HANDLER, None, params, config, save_dir=save_path)

        if result.success:
            for audio in result.audios:
                path = audio.get("path", "")
                if path and path.endswith(".wav"):
                    output_paths.append(path)
                    # Log seed for reproducibility
                    logger.info(f"Generated: {path} (seed={current_seed})")

    return output_paths
