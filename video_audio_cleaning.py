import os
import subprocess
import tempfile
import shutil
import wave
import contextlib
import webrtcvad
import whisper

# === Configuration ===
DEFAULT_VIDEO_NAME = "sample_video4.mp4"
CLEANED_VIDEO = "output_cleaned_vad.mp4"
CLEANED_AUDIO = "cleaned_audio_vad.wav"
TRANSCRIPT_TXT = "transcript.txt"
AUDIO_SAMPLE_RATE = 16000
VAD_AGGRESSIVENESS = 1  # Lower = less aggressive, preserves more speech
FRAME_DURATION_MS = 30  # ms
MIN_SPEECH_SEC = 0.3    # Minimum length of a speech segment to keep
MIN_SILENCE_SEC = 0.2   # Minimum silence between segments to consider splitting
MERGE_GAP_SEC = 0.5     # Merge segments if gap is less than this

def find_video_file():
    current_dir = os.getcwd()
    possible_locations = [
        current_dir,
        "/content",
        "/tmp",
        os.path.expanduser("~"),
        os.path.join(os.path.expanduser("~"), "Downloads"),
        os.path.join(os.path.expanduser("~"), "Videos"),
    ]
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    for location in possible_locations:
        if os.path.exists(location):
            video_path = os.path.join(location, DEFAULT_VIDEO_NAME)
            if os.path.exists(video_path):
                return video_path
    try:
        files = os.listdir(current_dir)
        video_files = [f for f in files if any(f.lower().endswith(ext) for ext in video_extensions)]
        if video_files:
            return os.path.join(current_dir, video_files[0])
    except:
        pass
    try:
        if os.path.exists("/content"):
            files = os.listdir("/content")
            video_files = [f for f in files if any(f.lower().endswith(ext) for ext in video_extensions)]
            if video_files:
                return os.path.join("/content", video_files[0])
    except:
        pass
    return None

def check_file_exists(file_path):
    if not file_path or file_path.startswith('-'):
        return False, "No valid file path provided"
    abs_path = os.path.abspath(file_path)
    if not os.path.exists(abs_path):
        current_dir = os.getcwd()
        return False, f"File not found: {abs_path}\nCurrent directory: {current_dir}\nMake sure the file exists and the path is correct."
    if not os.path.isfile(abs_path):
        return False, f"Path exists but is not a file: {abs_path}"
    if not os.access(abs_path, os.R_OK):
        return False, f"File exists but is not readable: {abs_path}"
    return True, abs_path

def list_available_videos():
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
    found_videos = []
    locations_to_check = [
        ("Current directory", os.getcwd()),
        ("Content directory", "/content"),
        ("Home directory", os.path.expanduser("~")),
        ("Downloads", os.path.join(os.path.expanduser("~"), "Downloads")),
        ("Videos", os.path.join(os.path.expanduser("~"), "Videos")),
    ]
    for location_name, location_path in locations_to_check:
        try:
            if os.path.exists(location_path) and os.path.isdir(location_path):
                files = os.listdir(location_path)
                video_files = [f for f in files if any(f.lower().endswith(ext) for ext in video_extensions)]
                if video_files:
                    found_videos.append((location_name, location_path, video_files))
        except PermissionError:
            continue
        except Exception:
            continue
    return found_videos

def check_dependencies():
    missing = []
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffmpeg")
    try:
        subprocess.run(['ffprobe', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        missing.append("ffprobe")
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("💡 Install FFmpeg: https://ffmpeg.org/download.html")
        return False
    return True

def extract_audio_smart(video_path, output_path):
    cmd = [
        'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', str(AUDIO_SAMPLE_RATE), '-ac', '1',
        output_path
    ]
    subprocess.run(cmd, check=True)

def read_wave(path):
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate == AUDIO_SAMPLE_RATE
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

def vad_collector(sample_rate, frame_duration_ms, vad, audio):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    segments = []
    triggered = False
    start_time = 0
    for i in range(0, len(audio), n):
        frame = audio[i:i+n]
        if len(frame) < n:
            break
        is_speech = vad.is_speech(frame, sample_rate)
        current_time = i / (sample_rate * 2)
        if is_speech and not triggered:
            triggered = True
            start_time = current_time
        elif not is_speech and triggered:
            end_time = current_time
            if end_time - start_time >= MIN_SPEECH_SEC:
                segments.append((start_time, end_time))
            triggered = False
    if triggered:
        end_time = len(audio) / (sample_rate * 2)
        if end_time - start_time >= MIN_SPEECH_SEC:
            segments.append((start_time, end_time))
    return segments

def merge_close_segments(segments, gap_threshold=MERGE_GAP_SEC):
    if not segments:
        return segments
    segments = sorted(segments)
    merged = [segments[0]]
    for current_start, current_end in segments[1:]:
        last_start, last_end = merged[-1]
        if current_start - last_end <= gap_threshold:
            merged[-1] = (last_start, max(last_end, current_end))
        else:
            merged.append((current_start, current_end))
    return merged

def filter_short_segments(segments, min_length=MIN_SPEECH_SEC):
    return [(s, e) for s, e in segments if (e - s) >= min_length]

def detect_speech_segments_vad(audio_file):
    print("🔍 Detecting speech segments using WebRTC VAD (conservative)...")
    audio, sample_rate = read_wave(audio_file)
    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    segments = vad_collector(sample_rate, FRAME_DURATION_MS, vad, audio)
    segments = merge_close_segments(segments, gap_threshold=MERGE_GAP_SEC)
    segments = filter_short_segments(segments, min_length=MIN_SPEECH_SEC)
    print(f"✅ Detected {len(segments)} speech segments.")
    return segments

def extract_and_concat_segments_ffmpeg(input_video, segments, output_video, output_audio):
    temp_dir = tempfile.mkdtemp()
    temp_video_files = []
    temp_audio_files = []

    for idx, (start, end) in enumerate(segments):
        seg_video = os.path.join(temp_dir, f"seg_{idx:03d}.mp4")
        seg_audio = os.path.join(temp_dir, f"seg_{idx:03d}.wav")
        duration = end - start
        # Video segment: GPU decode and encode
        subprocess.run([
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", input_video,
            "-c:v", "h264_nvenc", "-preset", "fast",
            "-c:a", "aac", "-b:a", "192k",
            seg_video
        ], check=True)
        temp_video_files.append(seg_video)
        # Audio segment: GPU decode, audio copy
        subprocess.run([
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-ss", f"{start:.3f}", "-t", f"{duration:.3f}",
            "-i", input_video,
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            seg_audio
        ], check=True)
        temp_audio_files.append(seg_audio)

    # Create concat list files
    video_list_file = os.path.join(temp_dir, "video_list.txt")
    with open(video_list_file, "w") as f:
        for vf in temp_video_files:
            f.write(f"file '{vf}'\n")

    audio_list_file = os.path.join(temp_dir, "audio_list.txt")
    with open(audio_list_file, "w") as f:
        for af in temp_audio_files:
            f.write(f"file '{af}'\n")

    # Concatenate video (GPU encode)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", video_list_file,
        "-c:v", "h264_nvenc", "-preset", "fast",
        "-c:a", "aac", "-b:a", "192k",
        output_video
    ], check=True)

    # Concatenate audio (CPU, as GPU audio encoding is not common)
    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", audio_list_file,
        "-c", "copy", output_audio
    ], check=True)

    shutil.rmtree(temp_dir)
    return output_video, output_audio

def main():
    if not check_dependencies():
        return
    input_video = find_video_file()
    exists, msg = check_file_exists(input_video)
    if not exists:
        print(msg)
        print("Available videos:")
        for loc, path, videos in list_available_videos():
            print(f"  {loc}: {videos}")
        return
    print(f"✅ Using video file: {msg}")

    # 1. Extract audio for VAD-based speech detection (GPU decode)
    temp_audio = tempfile.mktemp(suffix=".wav")
    extract_audio_smart(input_video, temp_audio)

    # 2. Detect speech segments using VAD (conservative, merges short silences)
    segments = detect_speech_segments_vad(temp_audio)
    if not segments:
        print("❌ No speech segments detected.")
        return

    # 3. Trim and clean video & audio (GPU for video)
    cleaned_video = CLEANED_VIDEO
    cleaned_audio = CLEANED_AUDIO
    cleaned_video, cleaned_audio = extract_and_concat_segments_ffmpeg(
        input_video, segments, cleaned_video, cleaned_audio
    )

    print(f"✅ Cleaned video saved as: {cleaned_video}")
    print(f"✅ Cleaned audio saved as: {cleaned_audio}")

    # 4. Transcribe cleaned audio to text using Whisper
    print("🔍 Transcribing cleaned audio with Whisper...")
    model = whisper.load_model("base")  # You can use "small", "medium", "large" for more accuracy
    result = model.transcribe(cleaned_audio)
    with open(TRANSCRIPT_TXT, "w", encoding="utf-8") as txt:
        txt.write(result["text"])
    print(f"✅ Transcript saved as: {TRANSCRIPT_TXT}")

    return cleaned_video, cleaned_audio, TRANSCRIPT_TXT

if __name__ == "__main__":
    main()