"""
Function to compose music using ACE-step's music generation.
"""
import sys
import os
import warnings # Add warnings import
from loguru import logger # Import loguru
from tqdm import tqdm # Import tqdm

# Configure loguru to suppress output
logger.remove()
logger.add(sys.stderr, level="CRITICAL") # Set level to CRITICAL for extreme silence
tqdm.disable = True # Disable tqdm progress bars globally

# --- Suppress Verbose Output ---
# Filter specific warnings that appear early
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=Warning) # Catch all general warnings

# Add the parent directory of ACE-Step to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'ACE-Step')))
import torch
import random
from acestep.pipeline_ace_step import ACEStepPipeline

# Initialize the pipeline globally to avoid reloading the model on every call
PIPELINE = None

def initialize_pipeline(checkpoint_dir=None): # Changed default to None to handle absolute path logic
    """Initializes the ACE-Step pipeline."""
    global PIPELINE
    if PIPELINE is None:
        if checkpoint_dir is None:
            # Default to the 'Models/ACE-Step' directory relative to the project root
            project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
            checkpoint_dir = os.path.join(project_root, 'Models', 'ACE-Step')

        PIPELINE = ACEStepPipeline(
            checkpoint_dir=checkpoint_dir,
            device_id=0,
            dtype="bfloat16" if torch.cuda.is_available() else "float32",
        )
        if not PIPELINE.loaded:
            PIPELINE.load_checkpoint()

def compose_music(
    prompt="",
    lyrics="",
    instrumental=True,
    n_gen=1,
    duration_seconds=213,
    infer_steps=60,
    guidance_scale=15.0,
    scheduler_type="euler",
    cfg_type="apg",
    seed=None,
    save_path=None,
):
    """
    Generate music using ACE-step's music generation model.

    Args:
        prompt (str): Description of the song to be generated.
        lyrics (str): Lyrics for the song. If empty, instrumental music will be generated.
        instrumental (bool): If True and no lyrics are provided, adds a tag to encourage instrumental music.
        n_gen (int): Number of generations to create.
        duration_seconds (int): Duration of the song in seconds. Defaults to 213 (3:33).
        infer_steps (int): Number of inference steps.
        guidance_scale (float): Guidance scale for the generation.
        scheduler_type (str): The scheduler type to use.
        cfg_type (str): The CFG type to use.
        seed (int, optional): Seed for random number generation. If None, a random seed will be used.
        save_path (str, optional): Directory to save the generated files. Defaults to './outputs'.

    Returns:
        list: Paths to the generated audio files.
    """
    initialize_pipeline()

    # Per developer documentation, use "[instrumental]" in the lyrics field to enforce instrumental music.
    if instrumental and not lyrics:
        lyrics = "[instrumental]"

    if seed is None:
        seed = random.randint(0, 2**32 - 1)

    output_paths = []
    for i in range(n_gen):
        current_seed = seed + i
        paths = PIPELINE(
            format="wav",
            audio_duration=duration_seconds,
            prompt=prompt,
            lyrics=lyrics,
            batch_size=1,
            infer_step=infer_steps,
            guidance_scale=guidance_scale,
            scheduler_type=scheduler_type,
            cfg_type=cfg_type,
            manual_seeds=str(current_seed),
            save_path=save_path,
        )

        # The pipeline returns a list of audio paths and a dict of params.
        # We only want the audio paths.
        for item in paths:
            if isinstance(item, str) and item.endswith('.wav'):
                output_paths.append(item)

    return output_paths