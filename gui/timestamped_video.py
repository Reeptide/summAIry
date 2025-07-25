import os
import re
import json
import shutil
from datetime import datetime

class TimestampedVideoPlayer:
    def __init__(self):
        self.topics = []
        self.video_path = ""
        self.output_dir = "timestamped_video_output"
        self.pdf_path = ""
        self.pdf_full_path = ""

    # ... [parse_timestamp, seconds_to_timestamp, parse_topic_file_rag_format, parse_topic_file unchanged] ...

    def create_timestamped_player(self, video_path, topic_file_path, pdf_path=None, output_name="timestamped_video", video_title=None):
        print(f"🎬 CREATING RAG-ENHANCED TIMESTAMPED VIDEO PLAYER")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 User: niranjani1401")
        print("="*70)

        # Parse topics
        print(f"🧠 Attempting to parse RAG-generated topics...")
        if not self.parse_topic_file_rag_format(topic_file_path):
            print(f"⚠️ RAG format failed, trying original format...")
            if not self.parse_topic_file(topic_file_path):
                print(f"❌ Failed to parse topics in both formats")
                return None

        # Use output_cleaned_vad.mp4 as the video file, copied to output dir
        if os.path.exists(video_path):
            self.video_path = os.path.basename(video_path)
            os.makedirs(self.output_dir, exist_ok=True)
            new_video_path = os.path.join(self.output_dir, self.video_path)
            shutil.copy2(video_path, new_video_path)
            print(f"📹 Using cleaned VAD video: {self.video_path}")
        else:
            print(f"❌ Cleaned VAD video not found: {video_path}")
            return None

        # Copy PDF if provided
        self.pdf_full_path = ""
        if pdf_path and os.path.exists(pdf_path):
            self.pdf_path = os.path.basename(pdf_path)
            pdf_output_path = os.path.join(self.output_dir, self.pdf_path)
            shutil.copy2(pdf_path, pdf_output_path)
            self.pdf_full_path = os.path.abspath(pdf_path)
            print(f"📄 Added PDF for download: {self.pdf_path}")
            print(f"📄 Full PDF Path: {self.pdf_full_path}")
        else:
            self.pdf_path = ""
            print("⚠️ No PDF found for download button.")

        if not video_title:
            video_title = f"RAG-Enhanced Lecture - {len(self.topics)} Topics"

        html_content = self.generate_html(video_title)

        html_filename = f"{output_name}.html"
        html_path = os.path.join(self.output_dir, html_filename)

        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)

        info_data = {
            'video_file': self.video_path,
            'topic_file': topic_file_path,
            'pdf_file': self.pdf_path,
            'pdf_full_path': self.pdf_full_path,
            'generated_pdfs_path': os.path.abspath(os.path.dirname(pdf_path)) if pdf_path else "",
            'total_topics': len(self.topics),
            'total_duration_seconds': max(topic['end_seconds'] for topic in self.topics),
            'generated_date': datetime.now().isoformat(),
            'user': 'niranjani1401',
            'rag_enhanced': True,
            'ai_generated_topics': True,
            'topics': self.topics
        }

        info_path = os.path.join(self.output_dir, f"{output_name}_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2)

        print(f"✅ RAG-ENHANCED VIDEO PLAYER CREATED!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🌐 HTML file: {html_path}")
        print(f"📋 Info file: {info_path}")
        print(f"📹 Video file: {self.video_path}")
        print(f"📄 PDF file: {self.pdf_path}")
        print(f"📄 PDF full path: {self.pdf_full_path}")
        if self.pdf_full_path:
            print(f"📁 Generated PDFs directory: {os.path.abspath(os.path.dirname(pdf_path))}")
        print(f"🧠 RAG topics processed: {len(self.topics)}")

        return {
            'html_path': html_path,
            'video_path': self.video_path,
            'pdf_path': self.pdf_path,
            'pdf_full_path': self.pdf_full_path,
            'generated_pdfs_path': os.path.abspath(os.path.dirname(pdf_path)) if pdf_path else "",
            'info_path': info_path,
            'output_dir': self.output_dir,
            'topics_count': len(self.topics),
            'rag_enhanced': True
        }

    # ... [generate_html and other methods unchanged] ...

# For standalone testing
if __name__ == "__main__":
    player = TimestampedVideoPlayer()
    video_path = os.path.join("video_analyser", "output_cleaned_vad.mp4")
    topic_file = os.path.join("rag_analysis", "topics.txt")
    pdf_path = input("Enter PDF file path (e.g., generated_pdfs/Morphology_PDF_Transcript.pdf): ").strip()
    if os.path.exists(video_path) and os.path.exists(topic_file):
        result = player.create_timestamped_player(video_path, topic_file, pdf_path=pdf_path)
        if result:
            print(f"\n🌐 Open this file in your browser:")
            print(f"   file://{os.path.abspath(result['html_path'])}")
            print(f"📁 Generated PDFs directory: {result['generated_pdfs_path']}")
        else:
            print("❌ Video or topic file not found!")
    else:
        print("❌ Video or topic file not found!")