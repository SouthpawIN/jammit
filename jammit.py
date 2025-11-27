import argparse
import os
import sys
import time
import subprocess
import threading
from collections import deque
import warnings # Import warnings

warnings.filterwarnings("ignore") # Aggressively ignore all warnings

# Move aggressive sys.stderr redirection to the earliest possible point
original_stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')

# Import the music composition function
from compose_music import compose_music

class JammitCLI:
    def __init__(self):
        self.playback_queue = deque()
        self.current_playing_process = None
        self.playback_thread = None
        self.infinite_mode_prompt = None
        self.generation_lock = threading.Lock() # To prevent multiple simultaneous generations
        self._should_terminate = False # New: Flag to signal termination
        # Store default generation parameters (will be updated by generate_and_play)
        self._duration_seconds = 213
        self._infer_steps = 60
        self._guidance_scale = 15.0

    def _play_audio(self, audio_path):
        if self._should_terminate:
            return

        if self.current_playing_process:
            try:
                self.current_playing_process.terminate()
                self.current_playing_process.wait()
            except Exception:
                pass # Ignore if process already terminated

        try:
            # Play audio silently
            self.current_playing_process = subprocess.Popen(
                ["aplay", audio_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Notify for infinite mode to trigger next generation
            if self.infinite_mode_prompt and not self.generation_lock.locked():
                threading.Thread(target=self._trigger_next_generation).start()

            while self.current_playing_process.poll() is None and not self._should_terminate:
                time.sleep(0.1) # Check frequently
            
            if self.current_playing_process and self.current_playing_process.poll() is None:
                self.current_playing_process.terminate() # Terminate if still running on shutdown
                self.current_playing_process.wait()
            self.current_playing_process = None # Clear process when finished
        except FileNotFoundError:
            pass # Suppress aplay not found error
        except Exception:
            pass # Ignore other playback errors

    def _playback_manager(self):
        while not self._should_terminate:
            if self.playback_queue:
                audio_path = self.playback_queue.popleft()
                self._play_audio(audio_path)
            else:
                time.sleep(1) # Wait if queue is empty

    def _perform_generation(self, prompt):
        # This function will be run in a separate thread to avoid blocking
        if self._should_terminate: # New: Check termination flag
            return

        with self.generation_lock:
            if self.infinite_mode_prompt:
                # Redirect stdout and stderr to /dev/null
                original_stdout = sys.stdout
                original_stderr = sys.stderr
                sys.stdout = open(os.devnull, 'w')
                sys.stderr = open(os.devnull, 'w')
                
                try:
                    generated_paths = compose_music(
                        prompt=prompt,
                        duration_seconds=self._duration_seconds,
                        infer_steps=self._infer_steps,
                        guidance_scale=self._guidance_scale,
                        n_gen=1
                    )
                finally:
                    # Restore stdout and stderr
                    sys.stdout.close()
                    sys.stderr.close()
                    sys.stdout = original_stdout
                    sys.stderr = original_stderr
                
                if generated_paths:
                    self.playback_queue.extend(generated_paths)

    def _trigger_next_generation(self):
        # This function is called when a song starts playing.
        # It should trigger the *next* generation without blocking.
        if self.infinite_mode_prompt:
            # Spawn a new thread to perform the actual generation
            # This ensures _trigger_next_generation itself is non-blocking
            generation_thread = threading.Thread(
                target=self._perform_generation,
                args=(self.infinite_mode_prompt,)
            )
            generation_thread.daemon = True
            generation_thread.start()

    def generate_and_play(self, prompt, infinite_mode=False, duration_seconds=213, infer_steps=60, guidance_scale=15.0):
        self._duration_seconds = duration_seconds
        self._infer_steps = infer_steps
        self._guidance_scale = guidance_scale
        if infinite_mode:
            self.infinite_mode_prompt = prompt
            # Start the playback manager thread
            self.playback_thread = threading.Thread(target=self._playback_manager)
            self.playback_thread.daemon = True
            self.playback_thread.start()
            
            # Trigger the first generation immediately
            self._trigger_next_generation()
            
            # Keep the main thread alive for infinite mode
            # In a real CLI, you might want a more robust way to handle Ctrl+C
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self._should_terminate = True # Signal background threads to terminate
            finally:
                # Ensure current_playing_process is terminated on exit
                if self.current_playing_process and self.current_playing_process.poll() is None:
                    self.current_playing_process.terminate()
                    self.current_playing_process.wait()
                # Optionally join the playback thread to ensure it cleans up
                if self.playback_thread and self.playback_thread.is_alive():
                    self.playback_thread.join(timeout=1) # Give it a moment to exit gracefully
        else:
            # Single generation mode
            # Redirect stdout and stderr to /dev/null
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            sys.stderr = open(os.devnull, 'w')

            try:
                generated_paths = compose_music(prompt=prompt, duration_seconds=duration_seconds,
                                                infer_steps=infer_steps, guidance_scale=guidance_scale, n_gen=1)
            finally:
                # Restore stdout and stderr
                sys.stdout.close()
                sys.stderr.close()
                sys.stdout = original_stdout
                sys.stderr = original_stderr

            if generated_paths:
                audio_path = generated_paths[0]
                self._play_audio(audio_path)

def main():
    # Aggressively redirect sys.stderr to os.devnull for the entire script execution
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        parser = argparse.ArgumentParser(description="Generate music using ACE-Step. Supports infinite generation and silent playback.")
        parser.add_argument("prompt", type=str, help="A description of the music to generate.")
        parser.add_argument("--lyrics", type=str, default="", help="Lyrics for the song. (Currently unused in JammitCLI)")
        parser.add_argument("--instrumental", action="store_true", help="Force instrumental generation. (Currently unused in JammitCLI)")
        parser.add_argument("--n_gen", type=int, default=1, help="Number of generations to create. (Currently fixed to 1 in JammitCLI)")
        parser.add_argument("--duration_seconds", type=int, default=213, help="Duration of the song in seconds.")
        parser.add_argument("--infer_steps", type=int, default=60, help="Number of inference steps.")
        parser.add_argument("--guidance_scale", type=float, default=15.0, help="Guidance scale.")
        parser.add_argument("--infinite", action="store_false", default=True, help="Disable infinite generation mode (default is enabled).")
        parser.add_argument("--output", type=str, default="outputs", help="Output directory for the generated files. (Currently unused in JammitCLI)") # Keep for compatibility, though internal save_path will be used

        args = parser.parse_args()

        cli = JammitCLI()
        cli.generate_and_play(
            prompt=args.prompt,
            infinite_mode=args.infinite,
            duration_seconds=args.duration_seconds,
            infer_steps=args.infer_steps,
            guidance_scale=args.guidance_scale
        )

    finally:
        # Restore original sys.stderr
        sys.stderr.close()
        sys.stderr = original_stderr

if __name__ == "__main__":
    main()