import os

class Config:
    # Video Processing Settings
    WHISPER_MODEL = "large"
    AUDIO_SAMPLE_RATE = 16000
    VAD_AGGRESSIVENESS = 1
    
    # RAG Analysis Settings
    LLM_MODEL = "llama3.1:8b"
    LLM_BASE_URL = "http://localhost:11434"
    
    # Output Directories
    OUTPUT_DIR = "output"
    VIDEO_OUTPUT_DIR = "output/videos"
    AUDIO_OUTPUT_DIR = "output/audio"
    TRANSCRIPT_OUTPUT_DIR = "output/transcripts"
    TOPICS_OUTPUT_DIR = "output/topics"
    HTML_OUTPUT_DIR = "output/html_players"
    
    def __init__(self):
        self.create_output_dirs()
    
    def create_output_dirs(self):
        """Create output directories if they don't exist"""
        dirs = [
            self.OUTPUT_DIR,
            self.VIDEO_OUTPUT_DIR, 
            self.AUDIO_OUTPUT_DIR,
            self.TRANSCRIPT_OUTPUT_DIR,
            self.TOPICS_OUTPUT_DIR,
            self.HTML_OUTPUT_DIR
        ]
        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)
            print(f"📁 Directory ready: {dir_path}")