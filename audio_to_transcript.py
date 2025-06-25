import os
import subprocess
import tempfile
import shutil
import wave
import contextlib
import webrtcvad
import whisper
import torch
import numpy as np
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# === Configuration ===
DEFAULT_VIDEO_NAME = "sample_video3.mp4"
CLEANED_VIDEO = "output_cleaned_vad.mp4"
CLEANED_AUDIO = "cleaned_audio_vad.wav"
TRANSCRIPT_TXT = "transcript.txt"
AUDIO_SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 1  # Lower = less aggressive, preserves more speech
FRAME_DURATION_MS = 30  # ms
MIN_SPEECH_SEC = 0.3    # Minimum length of a speech segment to keep
MIN_SILENCE_SEC = 0.2   # Minimum silence between segments to consider splitting
MERGE_GAP_SEC = 0.5     # Merge segments if gap is less than this
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

def transcribe_with_timestamps(audio_file):
    """Transcribe audio with start and end timestamps."""
    print(f"🔍 Loading Whisper model...")
    
    # Configure device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load Whisper model
    model = whisper.load_model(WHISPER_MODEL, device=device)
    
    print(f"🔊 Transcribing audio with Whisper ({device.upper()} acceleration)...")
    
    # Run transcription
    result = model.transcribe(audio_file)
    
    # Process segments with start and end timestamps
    transcript_parts = []
    
    for segment in result["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"].strip()
        
        # Format timestamps as [start time - end time]
        start_minutes = int(start_time // 60)
        start_seconds = int(start_time % 60)
        end_minutes = int(end_time // 60)
        end_seconds = int(end_time % 60)
        timestamp = f"[{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}]"
        
        # Add formatted segment to transcript
        transcript_parts.append(f"{timestamp} {text}")
    
    return "\n".join(transcript_parts)

    def main():
        if not check_dependencies():
            return

    # Check for GPU availability and log info
    get_gpu_info()

    input_video = find_video_file()
    exists, msg = check_file_exists(input_video)
    if not exists:
        print(msg)
        print("Available videos:")
        for loc, path, videos in list_available_videos():
            print(f"  {loc}: {videos}")
        return
    print(f"✅ Using video file: {msg}")

    # 1. Extract audio for VAD-based speech detection
    temp_audio = tempfile.mktemp(suffix=".wav")
    extract_audio_smart(input_video, temp_audio)

    # 2. Detect speech segments using VAD
    segments = detect_speech_segments_vad(temp_audio)
    if not segments:
        print("❌ No speech segments detected.")
        return

    # 3. Trim and clean video & audio
    cleaned_video = CLEANED_VIDEO
    cleaned_audio = CLEANED_AUDIO
    cleaned_video, cleaned_audio = extract_and_concat_segments_ffmpeg(input_video, segments, cleaned_video, cleaned_audio)

    print(f"✅ Cleaned video saved as: {cleaned_video}")
    print(f"✅ Cleaned audio saved as: {cleaned_audio}")

    # 4. Transcribe cleaned audio with timestamps
    print("🔍 Transcribing with timestamps...")
    transcript = transcribe_with_timestamps(cleaned_audio)
    
    # Save transcript to file
    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as txt:
        txt.write(transcript)
    
    print(f"✅ Timestamped transcript saved as: {TRANSCRIPT_TXT}")
    print("Sample of transcript:")
    print("\n".join(transcript.split("\n")[:3]) + "\n...")
 
    return cleaned_video, cleaned_audio, TRANSCRIPT_TXT

    if __name__ == "__main__":
        main()