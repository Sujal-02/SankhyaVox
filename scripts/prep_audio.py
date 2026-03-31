import os
import glob
import subprocess
from pathlib import Path

# Paths
DATA_DIR = Path(r"c:\Users\rakes\OneDrive\Desktop\Sujal\SankhyaVox\data")
SOURCE_DIR = DATA_DIR / "SankhyaVox_data"
TARGET_DIR = DATA_DIR / "raw"

# Ensure target directory exists
TARGET_DIR.mkdir(parents=True, exist_ok=True)

def fix_filename(filename):
    """Fix common naming issues based on the dataset."""
    name = str(filename.name)
    # Fix double speaker ID and double _raw
    name = name.replace("S04_S04_", "S04_")
    name = name.replace("_raw_raw", "_raw")
    # Fix chatur to catur
    name = name.replace("chatur", "catur")
    return name

def main():
    print("Starting audio pre-processing...")
    
    # Find all m4a and aac files
    files = []
    for ext in ["**/*.m4a", "**/*.aac"]:
        files.extend(SOURCE_DIR.glob(ext))
        
    if not files:
        print(f"No .m4a or .aac files found in {SOURCE_DIR}")
        return

    print(f"Found {len(files)} files to convert.")
    
    success_count = 0
    error_count = 0
    
    for f in files:
        fixed_name = fix_filename(f)
        # Change extension to .wav
        wav_name = Path(fixed_name).with_suffix(".wav").name
        
        # Extract Speaker ID (e.g., S01 from S01_eka_raw.wav)
        speaker_id = wav_name.split("_")[0]
        
        # Create speaker directory in target
        speaker_dir = TARGET_DIR / speaker_id
        speaker_dir.mkdir(parents=True, exist_ok=True)
        
        target_file = speaker_dir / wav_name
        
        # We enforce 16000 Hz sample rate as per config
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            print("Please install imageio-ffmpeg via: pip install imageio-ffmpeg")
            return

        cmd = [
            ffmpeg_exe,
            "-y",               # Overwrite if exists
            "-i", str(f),       # Input file
            "-ar", "16000",     # 16kHz
            "-ac", "1",         # Mono channel
            "-loglevel", "error", # Keep it relatively quiet, or we can remove for verbosity
            str(target_file)
        ]
        
        print(f"Converting: {f.name} -> {target_file.relative_to(DATA_DIR)}")
        try:
            # Run without capturing stdout so user can see text if needed
            subprocess.run(cmd, check=True)
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"ERROR converting {f.name}: {e}")
            error_count += 1
            
    print(f"\nCompleted! Successfully converted {success_count} files.")
    if error_count > 0:
        print(f"Encountered errors on {error_count} files.")

if __name__ == "__main__":
    main()
