#!/usr/bin/env python3
"""
Complete Video Topic Analyzer with GUI, RAG Notes, and PDF Generation
Video → Transcript → Topics → RAG Notes → PDF → Interactive HTML Player
"""

import os
import sys
import shutil
from datetime import datetime
import subprocess

# Add project directories to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'video_processing'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_analysis'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'gui'))

from speech_segmentation import VideoProcessor
from rag_transcript_analysis import RAGOptimizedTranscriptAnalyzer
from timestamped_video import TimestampedVideoPlayer
from config.settings import Config

class CompleteVideoTopicAnalyzer:
    def __init__(self):
        self.config = Config()
        self.video_processor = VideoProcessor()
        self.rag_analyzer = RAGOptimizedTranscriptAnalyzer(
            model=self.config.LLM_MODEL,
            base_url=self.config.LLM_BASE_URL
        )
        self.gui_player = TimestampedVideoPlayer()
    
    def process_video_complete_pipeline(self, video_path: str, video_title: str = None):
        """Complete pipeline: Video → Transcript → Topics → RAG Notes → PDF → Interactive Player"""
        print(f"🚀 COMPLETE VIDEO TOPIC ANALYZER PIPELINE")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 User: niranjani1401")
        print("="*70)
        
        if not video_title:
            video_title = f"Lecture Analysis - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            # Step 1: Process video and generate transcript
            print(f"\n📹 STEP 1: Video Processing & Transcription")
            print("-" * 50)
            transcript_file = self.video_processor.process_video(video_path)
            if not transcript_file:
                print("❌ Video processing failed")
                return None
            
            # Move transcript to organized location (always as transcript.txt in rag_analysis)
            transcript_dest = os.path.join("rag_analysis", "transcript.txt")
            shutil.copy2(transcript_file, transcript_dest)
            print(f"✅ Transcript copied to {transcript_dest}")

            # Step 1B: Move/rename cleaned VAD video file
            vad_dir = os.path.dirname(video_path)
            vad_files = [f for f in os.listdir(vad_dir) if f.startswith("output_cleaned_vad") and f.endswith(".mp4")]
            if vad_files:
                vad_src = os.path.join(vad_dir, vad_files[0])
                vad_dest = os.path.join("video_analyser", "output_cleaned_vad.mp4")
                shutil.copy2(vad_src, vad_dest)
                print(f"✅ Cleaned VAD video copied to {vad_dest}")
            else:
                print("❌ output_cleaned_vad.mp4 not found in video_analyser directory")
                return None

            # Step 2: Analyze transcript for RAG topics
            print(f"\n🧠 STEP 2: RAG Topic Analysis")
            print("-" * 50)
            topics_result = self.rag_analyzer.analyze_transcript_for_rag(transcript_dest)
            if not topics_result:
                print("❌ Topic analysis failed")
                return None
            topics_dest = os.path.join("rag_analysis", "topics.txt")
            shutil.copy2(topics_result['saved_file'], topics_dest)
            print(f"✅ Topics file copied to {topics_dest}")

            # Step 3: Run OptimizedHybridRAG to generate notes
            print(f"\n📚 STEP 3: RAG Notes Generation")
            print("-" * 50)
            rag_py = os.path.join("rag_analysis", "optimized_hybrid_rag.py")
            # Assume ipcvtb.txt is present in rag_analysis
            subprocess.run([sys.executable, rag_py], check=True)
            print(f"✅ OptimizedHybridRAG notes generated")

            # Step 4: Run PDF lecture notes generator
            print(f"\n📝 STEP 4: PDF Generation")
            print("-" * 50)
            pdf_py = os.path.join("rag_analysis", "pdf_lecture_notes.py")
            subprocess.run([sys.executable, pdf_py], check=True)
            generated_pdfs_dir = "generated_pdfs"
            pdf_files = [f for f in os.listdir(generated_pdfs_dir) if f.endswith(".pdf")]
            pdf_path = os.path.join(generated_pdfs_dir, pdf_files[0]) if pdf_files else ""
            if pdf_path:
                print(f"✅ PDF generated at {pdf_path}")
            else:
                print("❌ PDF generation failed")
                return None

            # Step 5: Create Interactive HTML Player
            print(f"\n🌐 STEP 5: Creating Interactive HTML Player")
            print("-" * 50)
            self.gui_player.output_dir = self.config.HTML_OUTPUT_DIR
            # Always use output_cleaned_vad.mp4 from video_analyser
            video_for_gui = os.path.join("video_analyser", "output_cleaned_vad.mp4")
            player_result = self.gui_player.create_timestamped_player(
                video_path=video_for_gui,
                topic_file_path=topics_dest,
                pdf_path=pdf_path,
                output_name=f"lecture_player_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                video_title=video_title
            )
            if not player_result:
                print("❌ HTML player creation failed")
                return None

            print(f"\n🎉 COMPLETE SUCCESS!")
            print("="*70)
            print(f"📄 Transcript: {transcript_dest}")
            print(f"📋 Topics: {topics_dest}")
            print(f"📚 PDF: {pdf_path}")
            print(f"🌐 HTML Player: {player_result['html_path']}")
            print(f"📁 All files in: {self.config.OUTPUT_DIR}")

            topics_count = len(self.gui_player.topics)
            print(f"\n📊 PIPELINE SUMMARY:")
            print(f"   🎬 Video processed: ✅")
            print(f"   📝 Transcript generated: ✅")
            print(f"   🧠 Topics extracted: {topics_count} topics")
            print(f"   📚 RAG notes generated: ✅")
            print(f"   📝 PDF notes generated: ✅")
            print(f"   🌐 Interactive player: ✅")

            return {
                'transcript_file': transcript_dest,
                'topics_file': topics_dest,
                'topics_result': topics_result,
                'player_result': player_result,
                'pdf_path': pdf_path,
                'video_file': video_for_gui,
                'topics_count': topics_count
            }

        except Exception as e:
            print(f"❌ Pipeline failed with error: {e}")
            return None

def main():
    """Main execution function"""
    print("🎬 COMPLETE VIDEO TOPIC ANALYZER")
    print("✨ Features: Video Processing + RAG Analysis + PDF + Interactive HTML Player")
    print("🎯 Purpose: End-to-end lecture analysis with visual navigation")
    print(f"👤 User: niranjani1401")
    print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    analyzer = CompleteVideoTopicAnalyzer()

    # Get video input
    video_path = input("\n📹 Enter video file path: ").strip()
    if not video_path:
        print("❌ No video path provided")
        return

    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return

    # Get optional video title
    video_title = input("📝 Enter video title (optional): ").strip()
    if not video_title:
        video_title = None

    print(f"\n🚀 Starting complete analysis pipeline...")
    print(f"📄 Input: {video_path}")
    print(f"🎯 Processing: Video → Transcript → Topics → RAG → PDF → Interactive Player")
    print("-" * 70)

    # Run complete pipeline
    result = analyzer.process_video_complete_pipeline(video_path, video_title)

    if result:
        print(f"\n🏆 ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"🌐 Open the HTML file to interact with your timestamped video:")
        print(f"   {result['player_result']['html_path']}")
        print(f"🎬 Total topics created: {result['topics_count']}")

        open_file = input(f"\n🌐 Open HTML player in browser? (y/N): ").strip().lower()
        if open_file in ['y', 'yes']:
            import webbrowser
            webbrowser.open(f"file://{os.path.abspath(result['player_result']['html_path'])}")

    else:
        print(f"\n❌ Analysis pipeline failed")

if __name__ == "__main__":
    main()