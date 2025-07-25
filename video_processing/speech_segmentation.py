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
from datetime import datetime

class VideoProcessor:
    def __init__(self):
        # === Configuration ===
        self.DEFAULT_VIDEO_NAME = "ipcv_vad2.mp4"
        self.AUDIO_SAMPLE_RATE = 16000
        self.VAD_AGGRESSIVENESS = 1  # Lower = less aggressive, preserves more speech
        self.FRAME_DURATION_MS = 30  # ms
        self.MIN_SPEECH_SEC = 0.3    # Minimum length of a speech segment to keep
        self.MIN_SILENCE_SEC = 0.2   # Minimum silence between segments to consider splitting
        self.MERGE_GAP_SEC = 0.5     # Merge segments if gap is less than this
        self.WHISPER_MODEL = "large"  # Options: tiny, base, small, medium, large
        
        # Output file names (with timestamp for uniqueness)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.CLEANED_VIDEO = f"output_cleaned_vad_{timestamp}.mp4"
        self.CLEANED_AUDIO = f"cleaned_audio_vad_{timestamp}.wav"
        self.TRANSCRIPT_TXT = f"transcript_{timestamp}.txt"

    def find_video_file(self):
        """Find video file in common locations."""
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
                video_path = os.path.join(location, self.DEFAULT_VIDEO_NAME)
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

    def check_file_exists(self, file_path):
        """Validate if file exists and is accessible."""
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

    def list_available_videos(self):
        """List all available video files in common locations."""
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

    def check_dependencies(self):
        """Check if required dependencies are available."""
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

    def get_gpu_info(self):
        """Check and display GPU information."""
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"🚀 GPU acceleration available: {gpu_name} ({gpu_count} device(s))")
        else:
            print("💻 Using CPU (no GPU available)")

    def extract_audio_smart(self, video_path, output_path):
        """Extract audio from video using FFmpeg with GPU acceleration."""
        cmd = [
            'ffmpeg', '-y', '-hwaccel', 'cuda', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', str(self.AUDIO_SAMPLE_RATE), '-ac', '1',
            output_path
        ]
        subprocess.run(cmd, check=True)

    def read_wave(self, path):
        """Read WAV file and return PCM data."""
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            assert num_channels == 1
            sample_width = wf.getsampwidth()
            assert sample_width == 2
            sample_rate = wf.getframerate()
            assert sample_rate == self.AUDIO_SAMPLE_RATE
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def vad_collector(self, sample_rate, frame_duration_ms, vad, audio):
        """Collect speech segments using VAD."""
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
                if end_time - start_time >= self.MIN_SPEECH_SEC:
                    segments.append((start_time, end_time))
                triggered = False
        
        if triggered:
            end_time = len(audio) / (sample_rate * 2)
            if end_time - start_time >= self.MIN_SPEECH_SEC:
                segments.append((start_time, end_time))
        
        return segments

    def merge_close_segments(self, segments, gap_threshold=None):
        """Merge segments that are close together."""
        if gap_threshold is None:
            gap_threshold = self.MERGE_GAP_SEC
            
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

    def filter_short_segments(self, segments, min_length=None):
        """Filter out segments that are too short."""
        if min_length is None:
            min_length = self.MIN_SPEECH_SEC
        return [(s, e) for s, e in segments if (e - s) >= min_length]

    def detect_speech_segments_vad(self, audio_file):
        """Detect speech segments using WebRTC VAD."""
        print("🔍 Detecting speech segments using WebRTC VAD (conservative)...")
        audio, sample_rate = self.read_wave(audio_file)
        vad = webrtcvad.Vad(self.VAD_AGGRESSIVENESS)
        
        segments = self.vad_collector(sample_rate, self.FRAME_DURATION_MS, vad, audio)
        segments = self.merge_close_segments(segments, gap_threshold=self.MERGE_GAP_SEC)
        segments = self.filter_short_segments(segments, min_length=self.MIN_SPEECH_SEC)
        
        print(f"✅ Detected {len(segments)} speech segments.")
        return segments

    def extract_and_concat_segments_ffmpeg(self, input_video, segments, output_video, output_audio):
        """Extract and concatenate video/audio segments using FFmpeg."""
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

    def transcribe_with_timestamps(self, audio_file):
        """Transcribe audio with start and end timestamps in RAG-compatible format."""
        print(f"🔍 Loading Whisper model...")
        
        # Configure device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Whisper model
        model = whisper.load_model(self.WHISPER_MODEL, device=device)
        
        print(f"🔊 Transcribing audio with Whisper ({device.upper()} acceleration)...")
        
        # Run transcription
        result = model.transcribe(audio_file)
        
        # Process segments with start and end timestamps in RAG format
        transcript_parts = []
        
        for segment in result["segments"]:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"].strip()
            
            # Format timestamps as [MM:SS - MM:SS] for RAG compatibility
            start_minutes = int(start_time // 60)
            start_seconds = int(start_time % 60)
            end_minutes = int(end_time // 60)
            end_seconds = int(end_time % 60)
            
            # Use MM:SS format for consistency with RAG analyzer
            timestamp = f"[{start_minutes:02d}:{start_seconds:02d} - {end_minutes:02d}:{end_seconds:02d}]"
            
            # Add formatted segment to transcript
            transcript_parts.append(f"{timestamp} {text}")
        
        return "\n".join(transcript_parts)

    def process_video(self, video_path=None):
        """
        Main processing method - returns the path to the generated transcript file.
        Modified to ensure RAG-compatible output format.
        """
        print(f"🎬 VIDEO PROCESSOR STARTED")
        print(f"📅 Date: 2025-07-23 13:23:29")
        print(f"👤 User: niranjani1401")
        print("="*50)
        
        if not self.check_dependencies():
            return None

        # Check for GPU availability and log info
        self.get_gpu_info()

        # Use provided video path or find one
        if video_path is None:
            input_video = self.find_video_file()
        else:
            input_video = video_path
            
        exists, msg = self.check_file_exists(input_video)
        if not exists:
            print(msg)
            print("Available videos:")
            for loc, path, videos in self.list_available_videos():
                print(f"  {loc}: {videos}")
            return None
            
        print(f"✅ Using video file: {msg}")

        try:
            # 1. Extract audio for VAD-based speech detection
            temp_audio = tempfile.mktemp(suffix=".wav")
            self.extract_audio_smart(input_video, temp_audio)

            # 2. Detect speech segments using VAD
            segments = self.detect_speech_segments_vad(temp_audio)
            if not segments:
                print("❌ No speech segments detected.")
                return None

            # 3. Trim and clean video & audio
            cleaned_video, cleaned_audio = self.extract_and_concat_segments_ffmpeg(
                input_video, segments, self.CLEANED_VIDEO, self.CLEANED_AUDIO
            )

            print(f"✅ Cleaned video saved as: {cleaned_video}")
            print(f"✅ Cleaned audio saved as: {cleaned_audio}")

            # 4. Transcribe cleaned audio with RAG-compatible timestamps
            print("🔍 Transcribing with RAG-compatible timestamps...")
            transcript = self.transcribe_with_timestamps(cleaned_audio)
            
            # Save transcript to file
            with open(self.TRANSCRIPT_TXT, "w", encoding="utf-8") as txt:
                txt.write(transcript)
            
            print(f"✅ RAG-compatible transcript saved as: {self.TRANSCRIPT_TXT}")
            print("📝 Sample of transcript (RAG format):")
            sample_lines = transcript.split("\n")[:3]
            for line in sample_lines:
                print(f"   {line}")
            print("   ...")
            
            # Clean up temporary file
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
            
            print(f"🎉 VIDEO PROCESSING COMPLETED SUCCESSFULLY!")
            print(f"📄 Transcript ready for RAG analysis: {self.TRANSCRIPT_TXT}")
            print(f"🔗 Format optimized for seamless RAG integration")
            
            return self.TRANSCRIPT_TXT
            
        except Exception as e:
            print(f"❌ Error during video processing: {e}")
            return None

    def get_output_files(self):
        """Get the paths of generated output files."""
        return {
            'transcript': self.TRANSCRIPT_TXT,
            'cleaned_video': self.CLEANED_VIDEO,
            'cleaned_audio': self.CLEANED_AUDIO
        }

# Standalone execution (for backward compatibility)
def main():
    """Standalone execution function."""
    processor = VideoProcessor()
    result = processor.process_video()
    
    if result:
        print(f"\n✅ Standalone processing completed!")
        print(f"📄 RAG-ready transcript: {result}")
    else:
        print(f"\n❌ Standalone processing failed!")

if __name__ == "__main__":
    main()