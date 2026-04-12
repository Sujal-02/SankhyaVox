import os
import glob
import shutil

src_dir = r"C:\Users\rakes\OneDrive\Desktop\Sujal\SankhyaVox\data\segments"
dst_dir = r"C:\Users\rakes\OneDrive\Desktop\Sujal\SankhyaVox\data\real_recordings"

VOCAB = [
    "shunya", "eka", "dvi", "tri", "catur", "pancha",
    "shat", "sapta", "ashta", "nava", "dasha", "vimsati", "shata"
]

print("Transferring recordings...")

# Create target directories
for token in VOCAB:
    os.makedirs(os.path.join(dst_dir, token), exist_ok=True)

# Find all .wav files in src_dir (e.g., S01/S01_shunya_01.wav)
wav_files = glob.glob(os.path.join(src_dir, "**", "*.wav"), recursive=True)

count = 0
for wav_path in wav_files:
    filename = os.path.basename(wav_path)
    
    # Identify token from filename
    matched_token = None
    for token in VOCAB:
        if f"_{token}_" in filename or filename.endswith(f"_{token}.wav"):
            matched_token = token
            break
            
    if matched_token:
        dst_path = os.path.join(dst_dir, matched_token, filename)
        # Copy file
        shutil.copy2(wav_path, dst_path)
        count += 1

print(f"✅ Successfully transferred {count} recordings into {dst_dir}")
