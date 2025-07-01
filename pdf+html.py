import os
import re
import json
import shutil
import textwrap
from datetime import datetime
from typing import List, Dict

# PDF libraries
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, darkblue, lightblue, whitesmoke, beige, grey, darkgrey, lightgrey
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER

# Semantic similarity for matching transcript to topics
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    has_sentence_transformers = True
except ImportError:
    has_sentence_transformers = False

class LecturePDFSummaryGenerator:
    """
    Enhanced PDF generator using semantic similarity to extract relevant transcript content for each topic.
    """
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        if has_sentence_transformers:
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception as e:
                print(f"⚠️ Could not load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None

    def _setup_custom_styles(self):
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=darkblue,
            fontName='Helvetica-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=blue,
            fontName='Helvetica'
        ))
        self.styles.add(ParagraphStyle(
            name='TopicHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            spaceBefore=20,
            textColor=darkblue,
            fontName='Helvetica-Bold',
            leftIndent=0.2*inch
        ))
        self.styles.add(ParagraphStyle(
            name='TopicContent',
            parent=self.styles['Normal'],
            fontSize=11,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leftIndent=0.4*inch,
            rightIndent=0.2*inch,
            fontName='Helvetica'
        ))
        self.styles.add(ParagraphStyle(
            name='Timestamp',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=15,
            textColor=grey,
            leftIndent=0.4*inch,
            fontName='Helvetica-Oblique'
        ))
        self.styles.add(ParagraphStyle(
            name='SummaryInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            textColor=darkgrey,
            fontName='Helvetica'
        ))

    def _time_to_seconds(self, time_str: str) -> int:
        parts = time_str.split(':')
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0

    def parse_topics_from_file(self, topics_file: str) -> List[Dict]:
        with open(topics_file, 'r', encoding='utf-8') as f:
            content = f.read()
        topics = []
        lines = content.strip().split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            topic_name = None
            start_time = None
            end_time = None
            # Format 1: *Topic Name**: [MM:SS - MM:SS]
            match1 = re.match(r'\*(.+?)\*\*:\s*\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', line)
            if match1:
                topic_name = match1.group(1).strip()
                start_time = match1.group(2)
                end_time = match1.group(3)
            elif ' : [' in line and line.endswith(']'):
                parts = line.split(' : [')
                if len(parts) == 2:
                    topic_name = parts[0].strip()
                    timestamp_part = parts[1].rstrip(']')
                    if ' - ' in timestamp_part:
                        start_time, end_time = timestamp_part.split(' - ')
                        start_time = start_time.strip()
                        end_time = end_time.strip()
            elif '(' in line and line.endswith(')'):
                match3 = re.match(r'(.+?)\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)', line)
                if match3:
                    topic_name = match3.group(1).strip()
                    start_time = match3.group(2)
                    end_time = match3.group(3)
            else:
                timestamp_match = re.search(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', line)
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    topic_name = line[:timestamp_match.start()].strip()
                    topic_name = re.sub(r'[*:]+$', '', topic_name).strip()
            if topic_name and start_time and end_time:
                topic_name = topic_name.strip('*').strip()
                start_seconds = self._time_to_seconds(start_time)
                end_seconds = self._time_to_seconds(end_time)
                if end_seconds > start_seconds:
                    topics.append({
                        'name': topic_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'duration': end_seconds - start_seconds
                    })
        topics.sort(key=lambda x: x['start_seconds'])
        return topics

    def parse_timestamped_transcript(self, transcript_file: str) -> List[Dict]:
        with open(transcript_file, 'r', encoding='utf-8') as f:
            content = f.read()
        segments = []
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            timestamp_match = re.match(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]\s*(.*)', line)
            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                text_content = timestamp_match.group(3).strip()
                if text_content:
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'content': text_content,
                        'start_seconds': self._time_to_seconds(start_time),
                        'end_seconds': self._time_to_seconds(end_time)
                    })
        return segments

    def extract_content_for_topics(self, topics: List[Dict], transcript_segments: List[Dict]) -> List[Dict]:
        enhanced_topics = []
        for topic in topics:
            topic_start = topic['start_seconds']
            topic_end = topic['end_seconds']
            relevant_segments = []
            for segment in transcript_segments:
                if (segment['start_seconds'] < topic_end and segment['end_seconds'] > topic_start):
                    overlap_start = max(segment['start_seconds'], topic_start)
                    overlap_end = min(segment['end_seconds'], topic_end)
                    overlap_duration = overlap_end - overlap_start
                    if overlap_duration > 0:
                        segment_duration = segment['end_seconds'] - segment['start_seconds']
                        overlap_percentage = overlap_duration / segment_duration if segment_duration > 0 else 0
                        relevant_segments.append({
                            'content': segment['content'],
                            'overlap_percentage': overlap_percentage,
                            'segment_start': segment['start_time'],
                            'segment_end': segment['end_time']
                        })
            explanatory_content = self._find_best_explanatory_content(
                topic['name'],
                relevant_segments
            )
            enhanced_topic = topic.copy()
            enhanced_topic['explanatory_content'] = explanatory_content
            enhanced_topic['source_segments_count'] = len(relevant_segments)
            enhanced_topics.append(enhanced_topic)
        return enhanced_topics

    def _find_best_explanatory_content(self, topic_name: str, relevant_segments: List[Dict]) -> List[str]:
        if not relevant_segments:
            return [f"Content for '{topic_name}' - please refer to the video at the specified timestamp."]
        all_content = []
        for segment in relevant_segments:
            sentences = re.split(r'(?<=[.?!])\s+', segment['content'])
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:
                    all_content.append({
                        'text': sentence,
                        'overlap_weight': segment['overlap_percentage']
                    })
        if not all_content:
            return [f"Discussion about {topic_name} - detailed content available in the video."]
        # Use semantic similarity if available
        if self.sentence_model and len(all_content) > 3:
            try:
                topic_embedding = self.sentence_model.encode([topic_name])
                sentence_texts = [item['text'] for item in all_content]
                sentence_embeddings = self.sentence_model.encode(sentence_texts)
                similarities = cosine_similarity(topic_embedding, sentence_embeddings)[0]
                for i, item in enumerate(all_content):
                    item['relevance_score'] = similarities[i] * (1 + item['overlap_weight'])
                all_content.sort(key=lambda x: x['relevance_score'], reverse=True)
            except Exception:
                all_content.sort(key=lambda x: x['overlap_weight'], reverse=True)
        else:
            all_content.sort(key=lambda x: x['overlap_weight'], reverse=True)
        selected_content = []
        total_length = 0
        max_total_length = 400
        for item in all_content:
            sentence = item['text'].strip()
            if sentence and len(sentence) > 15:
                if total_length + len(sentence) <= max_total_length and len(selected_content) < 3:
                    selected_content.append(sentence)
                    total_length += len(sentence)
                elif len(selected_content) == 0:
                    if len(sentence) > max_total_length:
                        sentence = sentence[:max_total_length-3] + "..."
                    selected_content.append(sentence)
                    break
        if not selected_content:
            selected_content = [f"Detailed discussion about {topic_name} available in the video."]
        return selected_content

    def generate_pdf_summary(self, topics_file: str, transcript_file: str, output_path: str, video_title: str = "Lecture Summary") -> str:
        topics = self.parse_topics_from_file(topics_file)
        if not topics:
            return None
        transcript_segments = self.parse_timestamped_transcript(transcript_file)
        enhanced_topics = self.extract_content_for_topics(topics, transcript_segments)
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        story = []
        # Title page
        story.extend(self._build_title_page(video_title, enhanced_topics))
        # Summary overview
        story.extend(self._build_summary_overview(enhanced_topics))
        # Topic details
        story.extend(self._build_topic_details(enhanced_topics))
        # Footer info
        story.extend(self._build_footer_info(transcript_file))
        try:
            doc.build(story)
            return output_path
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return None

    def _build_title_page(self, video_title: str, topics: List[Dict]) -> List:
        content = []
        content.append(Paragraph(video_title, self.styles['CustomTitle']))
        content.append(Spacer(1, 0.5*inch))
        content.append(Paragraph("📚 Comprehensive Topic Summary", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 0.3*inch))
        total_duration = max(topic['end_seconds'] for topic in topics) if topics else 0
        total_duration_str = f"{total_duration // 60}:{total_duration % 60:02d}"
        stats_data = [
            ['📊 Summary Statistics', ''],
            ['Total Topics:', str(len(topics))],
            ['Total Duration:', total_duration_str],
            ['Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Source:', 'AI-Powered Topic Segmentation'],
        ]
        stats_table = Table(stats_data, colWidths=[2.5*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 0), (-1, 0), lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), darkblue),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        content.append(stats_table)
        content.append(Spacer(1, 0.5*inch))
        description = """
        This document provides a comprehensive summary of the video lecture content, 
        organized by topics with relevant explanations extracted directly from the 
        transcript. Each topic includes the timestamp range and key explanatory content 
        to help you navigate and understand the lecture material effectively.
        """
        content.append(Paragraph(description, self.styles['Normal']))
        content.append(PageBreak())
        return content

    def _build_summary_overview(self, topics: List[Dict]) -> List:
        content = []
        content.append(Paragraph("📋 Topics Overview", self.styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))
        table_data = [['#', 'Topic', 'Duration', 'Time Range']]
        for i, topic in enumerate(topics, 1):
            duration = topic['end_seconds'] - topic['start_seconds']
            duration_str = f"{duration // 60}:{duration % 60:02d}"
            time_range = f"{topic['start_time']} - {topic['end_time']}"
            topic_name = topic['name']
            if len(topic_name) > 45:
                topic_name = textwrap.fill(topic_name, width=45)
            table_data.append([
                str(i),
                topic_name,
                duration_str,
                time_range
            ])
        overview_table = Table(table_data, colWidths=[0.5*inch, 3.5*inch, 0.8*inch, 1.2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), beige),
            ('GRID', (0, 0), (-1, -1), 1, black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        content.append(overview_table)
        content.append(PageBreak())
        return content

    def _build_topic_details(self, topics: List[Dict]) -> List:
        content = []
        content.append(Paragraph("📖 Detailed Topic Content", self.styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))
        for i, topic in enumerate(topics, 1):
            topic_header = f"{i}. {topic['name']}"
            content.append(Paragraph(topic_header, self.styles['TopicHeader']))
            timestamp_info = f"⏰ {topic['start_time']} - {topic['end_time']} " \
                            f"(Duration: {(topic['end_seconds'] - topic['start_seconds']) // 60}:" \
                            f"{(topic['end_seconds'] - topic['start_seconds']) % 60:02d})"
            content.append(Paragraph(timestamp_info, self.styles['Timestamp']))
            explanatory_content = topic.get('explanatory_content', [])
            if explanatory_content:
                content.append(Paragraph("📝 Key Points:", self.styles['Normal']))
                for explanation in explanatory_content:
                    explanation = explanation.strip()
                    if explanation:
                        bullet_text = f"• {explanation}"
                        content.append(Paragraph(bullet_text, self.styles['TopicContent']))
            else:
                content.append(Paragraph(
                    "• Detailed content available in the video at the specified timestamp.",
                    self.styles['TopicContent']
                ))
            source_info = f"📊 Extracted from {topic.get('source_segments_count', 0)} transcript segments"
            content.append(Paragraph(source_info, self.styles['SummaryInfo']))
            content.append(Spacer(1, 0.3*inch))
            if i % 4 == 0 and i < len(topics):
                content.append(PageBreak())
        return content

    def _build_footer_info(self, transcript_file: str) -> List:
        content = []
        content.append(PageBreak())
        content.append(Paragraph("ℹ️ Generation Information", self.styles['Heading2']))
        content.append(Spacer(1, 0.1*inch))
        details = [
            f"📄 Source Transcript: {os.path.basename(transcript_file)}",
            f"🤖 Generated by: AI-Powered Video Lecture Summarizer",
            f"📅 Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔧 Content Extraction: Semantic similarity matching with transcript overlap analysis",
            f"📋 Format: Topic name, timestamp range, and explanatory content from lecture",
        ]
        for detail in details:
            content.append(Paragraph(detail, self.styles['SummaryInfo']))
            content.append(Spacer(1, 0.1*inch))
        disclaimer = """
        <b>Note:</b> This summary is automatically generated from the video lecture transcript. 
        The explanatory content is extracted directly from the speaker's words during the 
        corresponding time segments. For complete understanding, please refer to the original 
        video content at the specified timestamps.
        """
        content.append(Spacer(1, 0.2*inch))
        content.append(Paragraph(disclaimer, self.styles['Normal']))
        return content

class SideBySideVideoPlayerWithPDF:
    def __init__(self):
        self.topics = []
        self.video_path = ""
        self.output_dir = "timestamped_video_output"
        self.pdf_generator = LecturePDFSummaryGenerator()

    @staticmethod
    def parse_timestamp(time_str):
        parts = time_str.split(':')
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        return 0

    @staticmethod
    def seconds_to_timestamp(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"

    def parse_topic_file(self, topic_file_path):
        try:
            with open(topic_file_path, 'r', encoding='utf-8') as f:
                lines = [line.strip() for line in f if line.strip()]
        except Exception as e:
            print(f"❌ Error reading topic file: {e}")
            return False
        topics = []
        for line in lines:
            topic_name = None
            timestamp_part = None
            if '**: [' in line and line.endswith(']'):
                parts = line.split('**: [')
                if len(parts) == 2:
                    topic_name = parts[0].lstrip('*').strip()
                    timestamp_part = parts[1].rstrip(']')
            elif ' : [' in line and line.endswith(']'):
                parts = line.split(' : [')
                if len(parts) == 2:
                    topic_name = parts[0].strip()
                    timestamp_part = parts[1].rstrip(']')
            elif '(' in line and line.endswith(')'):
                match3 = re.match(r'(.+?)\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)', line)
                if match3:
                    topic_name = match3.group(1).strip()
                    start_time = match3.group(2)
                    end_time = match3.group(3)
                    start_sec = self.parse_timestamp(start_time)
                    end_sec = self.parse_timestamp(end_time)
                    topics.append({
                        'name': topic_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_sec,
                        'end_seconds': end_sec,
                        'duration': end_sec - start_sec
                    })
                    continue
            else:
                timestamp_match = re.search(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', line)
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    topic_name = line[:timestamp_match.start()].strip()
                    topic_name = re.sub(r'[*:]+$', '', topic_name).strip()
                    start_sec = self.parse_timestamp(start_time)
                    end_sec = self.parse_timestamp(end_time)
                    topics.append({
                        'name': topic_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_sec,
                        'end_seconds': end_sec,
                        'duration': end_sec - start_sec
                    })
                    continue
            if topic_name and timestamp_part and ' - ' in timestamp_part:
                start_str, end_str = [s.strip() for s in timestamp_part.split(' - ')]
                start_sec = self.parse_timestamp(start_str)
                end_sec = self.parse_timestamp(end_str)
                topics.append({
                    'name': topic_name,
                    'start_time': start_str,
                    'end_time': end_str,
                    'start_seconds': start_sec,
                    'end_seconds': end_sec,
                    'duration': end_sec - start_sec
                })
        if not topics:
            print("❌ No topics found in the file")
            return False
        topics.sort(key=lambda x: x['start_seconds'])
        self.topics = topics
        return True

    def copy_video_file(self, video_path):
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        os.makedirs(self.output_dir, exist_ok=True)
        _, ext = os.path.splitext(video_path)
        new_video_path = os.path.join(self.output_dir, f"video{ext}")
        try:
            shutil.copy2(video_path, new_video_path)
        except Exception as e:
            print(f"❌ Error copying video: {e}")
            return None
        self.video_path = f"video{ext}"
        return new_video_path

    def generate_pdf_summary(self, topic_file_path, transcript_file_path, video_title):
        if not transcript_file_path or not os.path.exists(transcript_file_path):
            return None
        safe_title = re.sub(r'[^\w\s-]', '', video_title)
        safe_title = re.sub(r'[-\s]+', '_', safe_title)
        pdf_filename = f"{safe_title}_summary.pdf"
        pdf_path = os.path.join(self.output_dir, pdf_filename)
        result = self.pdf_generator.generate_pdf_summary(
            topics_file=topic_file_path,
            transcript_file=transcript_file_path,
            output_path=pdf_path,
            video_title=video_title
        )
        return pdf_filename if result else None

    def generate_css(self):
        return """
body {
    margin: 0; padding: 0;
    font-family: 'Roboto', Arial, sans-serif;
    background: #101010;
    color: #fff;
    height: 100vh; width: 100vw;
    overflow: hidden;
}
#main-container {
    display: flex;
    flex-direction: row;
    height: 100vh;
    width: 100vw;
}
#player-section {
    flex: 0 0 80vw;
    background: #000;
    display: flex;
    flex-direction: column;
    height: 100vh;
    position: relative;
}
#contents-section {
    flex: 0 0 20vw;
    background: #181818;
    border-left: 2px solid #292929;
    overflow-y: auto;
    height: 100vh;
    display: flex;
    flex-direction: column;
}
#player-video-box {
    flex: 1 1 auto;
    display: flex;
    align-items: center;
    justify-content: center;
    background: #000;
    min-height: 0;
    height: 91vh;
    max-height: 91vh;
}
#player-video-box video {
    width: 100%;
    height: 100%;
    max-height: 91vh;
    object-fit: contain;
    background: #000;
}
#player-timeline-bar {
    width: 96%;
    height: 8px;
    background: #333;
    border-radius: 4px;
    margin: 8px auto;
    position: relative;
    cursor: pointer;
    box-shadow: 0 1px 4px rgba(0,0,0,0.2);
}
#player-timeline-bar .timeline-bar {
    position: absolute;
    left: 0; top: 0; height: 100%; width: 100%;
    pointer-events: none;
    border-radius: 4px;
}
#player-timeline-bar .timeline-progress {
    position: absolute;
    left: 0; top: 0; height: 100%;
    background: #448aff;
    border-radius: 4px;
    transition: width 0.1s ease;
    z-index: 2;
}
#player-timeline-bar .timeline-segments {
    position: absolute;
    left: 0; top: 0; height: 100%; width: 100%;
    z-index: 1;
    pointer-events: none;
}
#player-timeline-bar .timeline-segment {
    position: absolute;
    top: 0; height: 100%;
    background: rgba(80, 200, 255, 0.2);
    z-index: 3;
    pointer-events: auto;
    cursor: pointer;
    transition: background 0.2s ease;
    border-left: 1px solid rgba(255,255,255,0.1);
}
#player-timeline-bar .timeline-segment:first-child {
    border-left: none;
}
#player-timeline-bar .timeline-segment.active {
    background: rgba(80, 200, 255, 0.5);
}
#player-timeline-bar .timeline-segment:hover {
    background: rgba(180, 220, 255, 0.4);
}
#timeline-hover-box {
    position: absolute;
    display: none;
    background: rgba(20,20,20,0.95);
    color: #fff;
    padding: 8px 12px;
    font-size: 12px;
    border-radius: 6px;
    pointer-events: none;
    z-index: 100;
    min-width: 140px;
    max-width: 300px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    white-space: nowrap;
    border: 1px solid rgba(255,255,255,0.1);
}
#timeline-hover-box .topic-name {
    font-weight: 600;
    margin-bottom: 2px;
    color: #fff;
}
#timeline-hover-box .topic-time {
    color: #aaf;
    font-size: 11px;
}
#player-info {
    padding: 12px 16px;
    background: #1a1a1a;
    border-top: 1px solid #242424;
    flex-shrink: 0;
}
#pdf-download {
    margin-top: 8px;
    text-align: left;
}
#pdf-download a {
    display: inline-block;
    background: linear-gradient(90deg,#3498db,#2c3e50);
    color: #fff;
    text-decoration: none;
    padding: 4px 10px;
    border-radius: 3px;
    font-weight: 500;
    font-size: 0.75rem;
    transition: background 0.2s;
}
#pdf-download a:hover {
    background: linear-gradient(90deg,#2c3e50,#3498db);
}
#video-title {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 8px;
}
#video-meta {
    display: flex;
    gap: 10px;
    color: #bbbbbb;
    font-size: 12px;
    margin-bottom: 8px;
}
.meta-item {
    background: #232323;
    border-radius: 8px;
    padding: 2px 8px;
    display: flex;
    align-items: center;
    gap: 3px;
}
#current-topic {
    background: #181818;
    border-left: 4px solid #3498db;
    border-radius: 6px;
    margin-top: 8px;
    padding: 10px 12px;
}
#current-topic h4 {
    margin: 0 0 5px 0;
    color: #60a5fa;
    font-size: 0.9rem;
}
#current-topic .topic-label {
    font-weight: 600;
    font-size: 0.95rem;
    color: #fff;
}
#contents-header {
    padding: 16px 12px 8px 12px;
    border-bottom: 1px solid #222;
}
#contents-header h2 {
    font-size: 1.1rem;
    font-weight: 700;
    margin: 0 0 3px 0;
    display: flex;
    align-items: center;
    gap: 6px;
    color: #fff;
}
#contents-header .subtitle {
    font-size: 0.85rem;
    color: #bbbbbb;
}
#toc-list {
    padding: 12px;
    flex: 1 1 auto;
    overflow-y: auto;
}
.topic-item {
    background: #222;
    border-radius: 6px;
    margin-bottom: 10px;
    padding: 10px 8px 8px 10px;
    cursor: pointer;
    border: 2px solid transparent;
    transition: background 0.16s, border 0.16s, transform 0.16s;
    position: relative;
}
.topic-item.active {
    background: linear-gradient(90deg,#3498db, #173a51 80%);
    color: #fff;
    border: 2px solid #3498db;
    transform: translateX(2px) scale(1.01);
}
.topic-item .topic-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
}
.topic-item .topic-number {
    background: #333;
    color: #5ddcff;
    font-size: 0.75em;
    border-radius: 4px;
    padding: 3px 6px;
    font-weight: 700;
    margin-right: 6px;
    min-width: 24px;
    text-align: center;
}
.topic-item.active .topic-number {
    background: #fff;
    color: #3498db;
}
.topic-item .topic-name {
    font-size: 0.9rem;
    font-weight: 600;
    color: #fff;
    flex: 1;
    margin-right: 4px;
    line-height: 1.2;
}
.topic-item .topic-time {
    font-family: monospace;
    font-size: 0.8em;
    color: #bbbbbb;
    margin-top: 4px;
}
.topic-item .topic-duration {
    background: #333;
    color: #fff;
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 0.7em;
    margin-left: 5px;
}
@media (max-width: 1000px) {
    #main-container {
        flex-direction: column;
    }
    #player-section, #contents-section {
        width: 100vw;
        flex: none;
    }
    #player-section { height: 60vh; }
    #contents-section { height: 40vh; }
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-thumb { background: #242424; border-radius: 3px; }
"""

    def generate_javascript(self):
        topics_json = json.dumps(self.topics)
        return f"""
(function() {{
    const topics = {topics_json};
    const video = document.getElementById('the-video');
    
    // Timeline bar logic
    const timelineBar = document.getElementById('player-timeline-bar');
    const progressBar = document.getElementById('timeline-progress');
    const segmentsContainer = document.getElementById('timeline-segments');
    const hoverBox = document.getElementById('timeline-hover-box');
    let totalDuration = 0;

    function getTotalDuration() {{
        if (video.duration && !isNaN(video.duration) && video.duration > 0) {{
            return video.duration;
        }}
        if (topics.length > 0) {{
            return Math.max(...topics.map(t => t.end_seconds));
        }}
        return 3600;
    }}

    function createTimelineSegments() {{
        segmentsContainer.innerHTML = '';
        totalDuration = getTotalDuration();
        topics.forEach((topic, idx) => {{
            const startPercent = (topic.start_seconds / totalDuration) * 100;
            const endPercent = (topic.end_seconds / totalDuration) * 100;
            const widthPercent = endPercent - startPercent;
            const seg = document.createElement('div');
            seg.className = 'timeline-segment';
            seg.style.left = startPercent + '%';
            seg.style.width = widthPercent + '%';
            seg.dataset.topicIndex = idx;
            seg.addEventListener('mouseenter', (e) => {{
                showHoverBox(e, idx);
            }});
            seg.addEventListener('mousemove', (e) => {{
                showHoverBox(e, idx);
            }});
            seg.addEventListener('mouseleave', () => {{
                hideHoverBox();
            }});
            seg.addEventListener('click', (e) => {{
                e.stopPropagation();
                jumpToTopic(idx);
            }});
            segmentsContainer.appendChild(seg);
        }});
    }}

    function updateTimelineProgress() {{
        if (!progressBar) return;
        const duration = getTotalDuration();
        if (duration > 0) {{
            const percent = (video.currentTime / duration) * 100;
            progressBar.style.width = Math.min(100, percent) + '%';
        }}
        const curr = video.currentTime;
        segmentsContainer.querySelectorAll('.timeline-segment').forEach((seg, idx) => {{
            const t = topics[idx];
            seg.classList.toggle('active', curr >= t.start_seconds && curr < t.end_seconds);
        }});
    }}

    function showHoverBox(e, idx) {{
        if (!hoverBox) return;
        const topic = topics[idx];
        hoverBox.innerHTML = `<div class="topic-name">${{topic.name}}</div><div class="topic-time">${{topic.start_time}} - ${{topic.end_time}}</div>`;
        hoverBox.style.display = 'block';
        
        const rect = timelineBar.getBoundingClientRect();
        const boxRect = hoverBox.getBoundingClientRect();
        let x = e.clientX - rect.left;
        
        // Center the box above the mouse position
        let leftVal = x - (boxRect.width / 2);
        
        // Keep the box within the timeline bounds
        if (leftVal < 0) leftVal = 5;
        if (leftVal + boxRect.width > rect.width) leftVal = rect.width - boxRect.width - 5;
        
        hoverBox.style.left = leftVal + 'px';
        hoverBox.style.top = '-45px';
    }}

    function hideHoverBox() {{
        if (hoverBox) hoverBox.style.display = 'none';
    }}

    timelineBar.addEventListener('click', function(e) {{
        const rect = timelineBar.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = x / rect.width;
        const duration = getTotalDuration();
        video.currentTime = Math.max(0, Math.min(duration, percent * duration));
    }});

    timelineBar.addEventListener('mousemove', function(e) {{
        const rect = timelineBar.getBoundingClientRect();
        const x = e.clientX - rect.left;
        const percent = x / rect.width;
        const duration = getTotalDuration();
        const time = percent * duration;
        
        // Find the topic for the current hover position
        let idx = -1;
        for (let i = 0; i < topics.length; i++) {{
            if (time >= topics[i].start_seconds && time <= topics[i].end_seconds) {{
                idx = i;
                break;
            }}
        }}
        
        if (idx >= 0) {{
            showHoverBox(e, idx);
        }} else {{
            hideHoverBox();
        }}
    }});
    
    timelineBar.addEventListener('mouseleave', hideHoverBox);

    video.addEventListener('timeupdate', updateTimelineProgress);
    video.addEventListener('loadedmetadata', () => {{
        totalDuration = getTotalDuration();
        createTimelineSegments();
        updateTimelineProgress();
    }});
    window.addEventListener('DOMContentLoaded', () => {{
        createTimelineSegments();
        updateTimelineProgress();
    }});

    function jumpToTopic(idx) {{
        if (idx >= 0 && idx < topics.length) {{
            video.currentTime = topics[idx].start_seconds;
            video.play();
        }}
    }}

    // Existing player logic
    const tocList = document.getElementById('toc-list');
    const currentTopicDiv = document.getElementById('current-topic');
    let currentTopicIdx = -1;
    function secondsToTimestamp(seconds) {{
        seconds = Math.floor(seconds);
        const h = Math.floor(seconds / 3600);
        const m = Math.floor((seconds % 3600) / 60);
        const s = seconds % 60;
        if (h > 0) return h+":" + String(m).padStart(2,'0') + ":" + String(s).padStart(2,'0');
        return String(m).padStart(2,'0') + ":" + String(s).padStart(2,'0');
    }}
    function highlightTOC(idx) {{
        const items = tocList.querySelectorAll('.topic-item');
        items.forEach((item,i) => {{
            if (i === idx) item.classList.add('active');
            else item.classList.remove('active');
        }});
        if (idx >= 0) {{
            items[idx].scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }}
    }}
    function setCurrentTopic(idx) {{
        currentTopicIdx = idx;
        if (idx < 0) {{
            currentTopicDiv.innerHTML = '<h4>Ready</h4><div class="topic-label">Click any topic to jump</div>';
            highlightTOC(-1);
        }} else {{
            const t = topics[idx];
            currentTopicDiv.innerHTML = `<h4>Now Playing</h4>
                <div class="topic-label">${{t.name}}</div>
                <div class="topic-time">${{t.start_time}} - ${{t.end_time}} &nbsp;|&nbsp; <span class="topic-duration">${{secondsToTimestamp(t.duration)}}</span></div>
                <div style="font-size:0.9em;color:#99b;">Section ${{idx+1}} of ${{topics.length}}</div>`;
            highlightTOC(idx);
        }}
    }}
    function findTopicIdx(time) {{
        for (let i=0;i<topics.length;i++) {{
            if (time >= topics[i].start_seconds && time < topics[i].end_seconds)
                return i;
        }}
        return -1;
    }}
    tocList.querySelectorAll('.topic-item').forEach((item, idx) => {{
        item.addEventListener('click', () => {{
            video.currentTime = topics[idx].start_seconds;
            video.play();
            setCurrentTopic(idx);
        }});
    }});
    video.addEventListener('timeupdate', () => {{
        const idx = findTopicIdx(video.currentTime);
        if (idx !== currentTopicIdx)
            setCurrentTopic(idx);
    }});
    video.addEventListener('loadedmetadata', () => {{
        setCurrentTopic(findTopicIdx(video.currentTime));
    }});
    setCurrentTopic(-1);

    document.addEventListener('keydown', (e) => {{
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
        if (e.key === " " || e.key === "Spacebar") {{
            e.preventDefault();
            if (video.paused) video.play(); else video.pause();
        }}
        if (e.key === "ArrowRight") {{
            video.currentTime = Math.min(video.duration, video.currentTime + 10);
        }}
        if (e.key === "ArrowLeft") {{
            video.currentTime = Math.max(0, video.currentTime - 10);
        }}
        if (e.key === "ArrowUp") {{
            e.preventDefault();
            if (currentTopicIdx > 0)
                video.currentTime = topics[currentTopicIdx-1].start_seconds;
        }}
        if (e.key === "ArrowDown") {{
            e.preventDefault();
            if (currentTopicIdx < topics.length-1 && currentTopicIdx >= 0)
                video.currentTime = topics[currentTopicIdx+1].start_seconds;
        }}
        if (e.key === "f" || e.key === "F") {{
            if (video.requestFullscreen) video.requestFullscreen();
        }}
    }});
}})();
"""

    def generate_html(self, video_title="Lecture Video", pdf_filename=None):
        toc_items = []
        for i, t in enumerate(self.topics):
            toc_items.append(f"""
<div class="topic-item" tabindex="0">
    <div class="topic-header">
        <span class="topic-number">{i+1:02d}</span>
        <span class="topic-name">{t['name']}</span>
    </div>
    <div class="topic-time">{t['start_time']} - {t['end_time']}
        <span class="topic-duration">{self.seconds_to_timestamp(t['duration'])}</span>
    </div>
</div>
""")
        toc_html = "\n".join(toc_items)
        n_topics = len(self.topics)
        total_duration = self.seconds_to_timestamp(max((t['end_seconds'] for t in self.topics), default=0))
        pdf_section = (f"""
<div id="pdf-download">
    <a href="{pdf_filename}" download="{pdf_filename}" target="_blank">📄 PDF Summary</a>
</div>
""" if pdf_filename else "")
        
        timeline_html = f"""
<div id="player-timeline-bar">
    <div class="timeline-bar"></div>
    <div class="timeline-progress" id="timeline-progress" style="width:0%"></div>
    <div class="timeline-segments" id="timeline-segments"></div>
    <div id="timeline-hover-box"></div>
</div>
"""
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <title>{video_title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://fonts.googleapis.com/css?family=Roboto:400,500,700&display=swap" rel="stylesheet">
    <style>
{self.generate_css()}
    </style>
</head>
<body>
<div id="main-container">
    <!-- Video Player Section (Left, 80%) -->
    <div id="player-section">
        <div id="player-video-box">
            <video id="the-video" controls preload="metadata">
                <source src="{self.video_path}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        {timeline_html}
        <div id="player-info">
            <div id="video-title">{video_title}</div>
            <div id="video-meta">
                <div class="meta-item">📚 {n_topics} Topics</div>
                <div class="meta-item">⏱️ {total_duration}</div>
                <div class="meta-item">📅 {datetime.now().strftime('%b %d, %Y')}</div>
                <div class="meta-item">🎬 Interactive</div>
            </div>
            <div id="current-topic"></div>
            {pdf_section}
        </div>
    </div>
    <!-- Table of Contents Section (Right, 20%) -->
    <div id="contents-section">
        <div id="contents-header">
            <h2>📋 Contents</h2>
            <div class="subtitle">Click to jump to section</div>
        </div>
        <div id="toc-list">
            {toc_html}
        </div>
    </div>
</div>
<script>
{self.generate_javascript()}
</script>
</body>
</html>
"""
        return html

    def create_player_with_pdf(self, video_path, topic_file_path, transcript_file_path=None,
                              output_name="video_player_side_by_side", video_title=None):
        if not self.parse_topic_file(topic_file_path):
            return None
        copied_video = self.copy_video_file(video_path)
        if not copied_video:
            return None
        if not video_title:
            video_title = "Lecture Video"
        pdf_filename = None
        if transcript_file_path and os.path.exists(transcript_file_path):
            pdf_filename = self.generate_pdf_summary(topic_file_path, transcript_file_path, video_title)
        html_content = self.generate_html(video_title, pdf_filename)
        html_filename = f"{output_name}.html"
        html_path = os.path.join(self.output_dir, html_filename)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        return {
            'html_path': html_path,
            'video_path': copied_video,
            'pdf_path': os.path.join(self.output_dir, pdf_filename) if pdf_filename else None,
            'output_dir': self.output_dir,
            'topics_count': len(self.topics)
        }

def main():
    print("🟦 SIDE-BY-SIDE VIDEO PLAYER (80/20 layout)")
    video_path = input("Enter video file path: ").strip() or "output_cleaned_vad_mod.mp4"
    topic_file = input("Enter topic segmentation file path: ").strip()
    if not topic_file:
        import glob
        files = glob.glob("topic_segmentation_*.txt")
        topic_file = files[0] if files else None
    transcript_file = input("Enter transcript file path (optional, for PDF generation): ").strip()
    if transcript_file and not os.path.exists(transcript_file):
        transcript_file = None
    video_title = input("Enter video title (optional): ").strip() or "Lecture Video"
    output_name = input("Enter output name (default: video_player_side_by_side): ").strip() or "video_player_side_by_side"
    if not (os.path.exists(video_path) and os.path.exists(topic_file)):
        print("❌ Required files not found.")
        return
    player = SideBySideVideoPlayerWithPDF()
    result = player.create_player_with_pdf(
        video_path=video_path,
        topic_file_path=topic_file,
        transcript_file_path=transcript_file,
        output_name=output_name,
        video_title=video_title
    )
    if result:
        print(f"\n✅ Player created: {os.path.abspath(result['html_path'])}")
        if result['pdf_path']:
            print(f"📄 PDF: {os.path.abspath(result['pdf_path'])}")

if __name__ == "__main__":
    main()