import os
import re
import json
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import Color, black, blue, darkblue
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.platypus import PageBreak, Image
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
from reportlab.pdfgen import canvas
from reportlab.lib import colors
import textwrap

class LecturePDFSummaryGenerator:
    """
    Generate PDF summaries with content extracted from actual lecture transcript
    Enhanced to handle multiple topic formats
    """
    
    def __init__(self):
        # Initialize sentence transformer for content matching
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded for content extraction")
        except Exception as e:
            print(f"⚠️ Could not load sentence transformer: {e}")
            self.sentence_model = None
        
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom PDF styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            spaceAfter=30,
            alignment=TA_CENTER,
            textColor=darkblue,
            fontName='Helvetica-Bold'
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading1'],
            fontSize=16,
            spaceAfter=20,
            alignment=TA_CENTER,
            textColor=blue,
            fontName='Helvetica'
        ))
        
        # Topic header style
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
        
        # Content style
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
        
        # Timestamp style
        self.styles.add(ParagraphStyle(
            name='Timestamp',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=15,
            textColor=colors.grey,
            leftIndent=0.4*inch,
            fontName='Helvetica-Oblique'
        ))
        
        # Summary info style
        self.styles.add(ParagraphStyle(
            name='SummaryInfo',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8,
            textColor=colors.darkgrey,
            fontName='Helvetica'
        ))
    
    def parse_topics_from_file(self, topics_file: str) -> List[Dict]:
        """
        Parse topics from file supporting multiple formats:
        1. *Topic Name**: [MM:SS - MM:SS]
        2. Topic Name : [MM:SS - MM:SS]
        3. Topic Name (MM:SS - MM:SS)
        """
        print(f"📄 Parsing topics from file: {topics_file}")
        
        try:
            with open(topics_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading topics file: {e}")
            return []
        
        topics = []
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            topic_name = None
            start_time = None
            end_time = None
            
            # Format 1: *Topic Name**: [MM:SS - MM:SS] (Your current format)
            match1 = re.match(r'\*(.+?)\*\*:\s*\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', line)
            if match1:
                topic_name = match1.group(1).strip()
                start_time = match1.group(2)
                end_time = match1.group(3)
                print(f"  ✅ Format 1 - Line {line_num}: {topic_name} ({start_time} - {end_time})")
            
            # Format 2: Topic Name : [MM:SS - MM:SS]
            elif ' : [' in line and line.endswith(']'):
                parts = line.split(' : [')
                if len(parts) == 2:
                    topic_name = parts[0].strip()
                    timestamp_part = parts[1].rstrip(']')
                    if ' - ' in timestamp_part:
                        start_time, end_time = timestamp_part.split(' - ')
                        start_time = start_time.strip()
                        end_time = end_time.strip()
                        print(f"  ✅ Format 2 - Line {line_num}: {topic_name} ({start_time} - {end_time})")
            
            # Format 3: Topic Name (MM:SS - MM:SS)
            elif '(' in line and line.endswith(')'):
                match3 = re.match(r'(.+?)\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)', line)
                if match3:
                    topic_name = match3.group(1).strip()
                    start_time = match3.group(2)
                    end_time = match3.group(3)
                    print(f"  ✅ Format 3 - Line {line_num}: {topic_name} ({start_time} - {end_time})")
            
            # Format 4: Handle timestamps with brackets anywhere in the line
            else:
                # Look for timestamp pattern anywhere in the line
                timestamp_match = re.search(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', line)
                if timestamp_match:
                    start_time = timestamp_match.group(1)
                    end_time = timestamp_match.group(2)
                    # Extract topic name (everything before the timestamp)
                    topic_name = line[:timestamp_match.start()].strip()
                    # Clean up topic name
                    topic_name = re.sub(r'[*:]+$', '', topic_name).strip()
                    print(f"  ✅ Format 4 - Line {line_num}: {topic_name} ({start_time} - {end_time})")
            
            # If we successfully parsed the topic
            if topic_name and start_time and end_time:
                # Clean topic name
                topic_name = topic_name.strip('*').strip()
                
                # Convert times to seconds
                start_seconds = self._time_to_seconds(start_time)
                end_seconds = self._time_to_seconds(end_time)
                
                # Validate that end time is after start time
                if end_seconds > start_seconds:
                    topics.append({
                        'name': topic_name,
                        'start_time': start_time,
                        'end_time': end_time,
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'duration': end_seconds - start_seconds
                    })
                else:
                    print(f"  ⚠️ Line {line_num}: Invalid time range {start_time} - {end_time}")
            else:
                print(f"  ❌ Line {line_num}: Could not parse - {line}")
        
        # Sort topics by start time
        topics.sort(key=lambda x: x['start_seconds'])
        
        print(f"✅ Successfully parsed {len(topics)} topics")
        return topics
    
    def parse_timestamped_transcript(self, transcript_file: str) -> List[Dict]:
        """Parse timestamped transcript into segments"""
        print(f"📄 Parsing timestamped transcript: {transcript_file}")
        
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading transcript: {e}")
            return []
        
        segments = []
        lines = content.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match pattern: [MM:SS - MM:SS] content
            timestamp_match = re.match(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]\s*(.*)', line)
            if timestamp_match:
                start_time = timestamp_match.group(1)
                end_time = timestamp_match.group(2)
                text_content = timestamp_match.group(3).strip()
                
                if text_content:  # Only include segments with content
                    segments.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'content': text_content,
                        'start_seconds': self._time_to_seconds(start_time),
                        'end_seconds': self._time_to_seconds(end_time)
                    })
        
        print(f"✅ Parsed {len(segments)} transcript segments")
        return segments
    
    def _time_to_seconds(self, time_str: str) -> int:
        """Convert MM:SS or HH:MM:SS to seconds"""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            return int(parts[0]) * 60 + int(parts[1])
        elif len(parts) == 3:  # HH:MM:SS
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        return 0
    
    def extract_content_for_topics(self, topics: List[Dict], transcript_segments: List[Dict]) -> List[Dict]:
        """Extract the most relevant content from transcript for each topic"""
        print(f"🔍 Extracting content for {len(topics)} topics from transcript...")
        
        enhanced_topics = []
        
        for topic in topics:
            topic_start = topic['start_seconds']
            topic_end = topic['end_seconds']
            
            # Find transcript segments that overlap with this topic
            relevant_segments = []
            for segment in transcript_segments:
                # Check for overlap
                if (segment['start_seconds'] < topic_end and segment['end_seconds'] > topic_start):
                    overlap_start = max(segment['start_seconds'], topic_start)
                    overlap_end = min(segment['end_seconds'], topic_end)
                    overlap_duration = overlap_end - overlap_start
                    
                    if overlap_duration > 0:
                        # Calculate overlap percentage
                        segment_duration = segment['end_seconds'] - segment['start_seconds']
                        overlap_percentage = overlap_duration / segment_duration if segment_duration > 0 else 0
                        
                        relevant_segments.append({
                            'content': segment['content'],
                            'overlap_percentage': overlap_percentage,
                            'segment_start': segment['start_time'],
                            'segment_end': segment['end_time']
                        })
            
            # Extract best explanatory content
            explanatory_content = self._find_best_explanatory_content(
                topic['name'], 
                relevant_segments
            )
            
            enhanced_topic = topic.copy()
            enhanced_topic['explanatory_content'] = explanatory_content
            enhanced_topic['source_segments_count'] = len(relevant_segments)
            
            enhanced_topics.append(enhanced_topic)
            print(f"  📝 {topic['name']}: Found {len(relevant_segments)} relevant segments")
        
        print(f"✅ Content extraction completed")
        return enhanced_topics
    
    def _find_best_explanatory_content(self, topic_name: str, relevant_segments: List[Dict]) -> List[str]:
        """Find the best 2-3 explanatory sentences for a topic"""
        if not relevant_segments:
            return [f"Content for '{topic_name}' - please refer to the video at the specified timestamp."]
        
        # Combine all content from relevant segments
        all_content = []
        for segment in relevant_segments:
            # Split content into sentences
            sentences = re.split(r'[.!?]+', segment['content'])
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Filter out very short fragments
                    all_content.append({
                        'text': sentence,
                        'overlap_weight': segment['overlap_percentage']
                    })
        
        if not all_content:
            return [f"Discussion about {topic_name} - detailed content available in the video."]
        
        # Use semantic similarity to find most relevant content
        if self.sentence_model and len(all_content) > 3:
            try:
                # Get embeddings for topic name and all sentences
                topic_embedding = self.sentence_model.encode([topic_name])
                sentence_texts = [item['text'] for item in all_content]
                sentence_embeddings = self.sentence_model.encode(sentence_texts)
                
                # Calculate similarity scores
                similarities = cosine_similarity(topic_embedding, sentence_embeddings)[0]
                
                # Combine semantic similarity with overlap weight
                for i, item in enumerate(all_content):
                    item['relevance_score'] = similarities[i] * (1 + item['overlap_weight'])
                
                # Sort by relevance and select top sentences
                all_content.sort(key=lambda x: x['relevance_score'], reverse=True)
                
            except Exception as e:
                print(f"⚠️ Semantic matching failed: {e}, using overlap-based selection")
                # Fallback to overlap-based selection
                all_content.sort(key=lambda x: x['overlap_weight'], reverse=True)
        else:
            # Simple selection based on overlap weight
            all_content.sort(key=lambda x: x['overlap_weight'], reverse=True)
        
        # Select best 2-3 sentences
        selected_content = []
        total_length = 0
        max_total_length = 400  # Character limit for explanatory content
        
        for item in all_content:
            sentence = item['text'].strip()
            if sentence and len(sentence) > 15:  # Minimum sentence length
                if total_length + len(sentence) <= max_total_length and len(selected_content) < 3:
                    selected_content.append(sentence)
                    total_length += len(sentence)
                elif len(selected_content) == 0:  # Always include at least one sentence
                    # Truncate if necessary
                    if len(sentence) > max_total_length:
                        sentence = sentence[:max_total_length-3] + "..."
                    selected_content.append(sentence)
                    break
        
        # Ensure we have at least some content
        if not selected_content:
            selected_content = [f"Detailed discussion about {topic_name} available in the video."]
        
        return selected_content
    
    def generate_pdf_summary(self, topics_file: str, transcript_file: str, 
                           output_path: str, video_title: str = "Lecture Summary") -> str:
        """Generate comprehensive PDF summary"""
        print(f"📄 Generating PDF summary: {output_path}")
        
        # Parse topics from file
        topics = self.parse_topics_from_file(topics_file)
        if not topics:
            print("❌ No topics found, cannot generate PDF")
            return None
        
        # Parse transcript and extract content for topics
        transcript_segments = self.parse_timestamped_transcript(transcript_file)
        enhanced_topics = self.extract_content_for_topics(topics, transcript_segments)
        
        # Create PDF document
        doc = SimpleDocTemplate(
            output_path,
            pagesize=A4,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=1*inch
        )
        
        # Build content
        story = []
        
        # Title page
        story.extend(self._build_title_page(video_title, enhanced_topics))
        
        # Summary overview
        story.extend(self._build_summary_overview(enhanced_topics))
        
        # Topic details
        story.extend(self._build_topic_details(enhanced_topics))
        
        # Footer info
        story.extend(self._build_footer_info(transcript_file))
        
        # Generate PDF
        try:
            doc.build(story)
            print(f"✅ PDF summary generated successfully: {output_path}")
            return output_path
        except Exception as e:
            print(f"❌ Error generating PDF: {e}")
            return None
    
    def _build_title_page(self, video_title: str, topics: List[Dict]) -> List:
        """Build PDF title page"""
        content = []
        
        # Main title
        content.append(Paragraph(video_title, self.styles['CustomTitle']))
        content.append(Spacer(1, 0.5*inch))
        
        # Subtitle
        content.append(Paragraph("📚 Comprehensive Topic Summary", self.styles['CustomSubtitle']))
        content.append(Spacer(1, 0.3*inch))
        
        # Summary statistics
        total_duration = max(topic['end_seconds'] for topic in topics) if topics else 0
        total_duration_str = f"{total_duration // 60}:{total_duration % 60:02d}"
        
        stats_data = [
            ['📊 Summary Statistics', ''],
            ['Total Topics:', str(len(topics))],
            ['Total Duration:', total_duration_str],
            ['Generated:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Source:', 'AI-Powered Topic Segmentation'],
            ['User:', 'Atulk47']
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 2*inch])
        stats_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.darkblue),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('GRID', (0, 0), (-1, -1), 1, colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        content.append(stats_table)
        content.append(Spacer(1, 0.5*inch))
        
        # Description
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
        """Build summary overview section"""
        content = []
        
        content.append(Paragraph("📋 Topics Overview", self.styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))
        
        # Create topics table
        table_data = [['#', 'Topic', 'Duration', 'Time Range']]
        
        for i, topic in enumerate(topics, 1):
            duration = topic['end_seconds'] - topic['start_seconds']
            duration_str = f"{duration // 60}:{duration % 60:02d}"
            time_range = f"{topic['start_time']} - {topic['end_time']}"
            
            # Wrap long topic names
            topic_name = topic['name']
            if len(topic_name) > 45:
                topic_name = textwrap.fill(topic_name, width=45)
            
            table_data.append([
                str(i),
                topic_name,
                duration_str,
                time_range
            ])
        
        # Create and style table
        overview_table = Table(table_data, colWidths=[0.5*inch, 3.5*inch, 0.8*inch, 1.2*inch])
        overview_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.darkblue),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('FONTSIZE', (0, 1), (-1, -1), 9),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ]))
        
        content.append(overview_table)
        content.append(PageBreak())
        
        return content
    
    def _build_topic_details(self, topics: List[Dict]) -> List:
        """Build detailed topic content section"""
        content = []
        
        content.append(Paragraph("📖 Detailed Topic Content", self.styles['Heading1']))
        content.append(Spacer(1, 0.2*inch))
        
        for i, topic in enumerate(topics, 1):
            # Topic header
            topic_header = f"{i}. {topic['name']}"
            content.append(Paragraph(topic_header, self.styles['TopicHeader']))
            
            # Timestamp info
            timestamp_info = f"⏰ {topic['start_time']} - {topic['end_time']} " \
                            f"(Duration: {(topic['end_seconds'] - topic['start_seconds']) // 60}:" \
                            f"{(topic['end_seconds'] - topic['start_seconds']) % 60:02d})"
            content.append(Paragraph(timestamp_info, self.styles['Timestamp']))
            
            # Explanatory content
            explanatory_content = topic.get('explanatory_content', [])
            if explanatory_content:
                content.append(Paragraph("📝 Key Points:", self.styles['Normal']))
                for j, explanation in enumerate(explanatory_content, 1):
                    # Clean up the explanation text
                    explanation = explanation.strip()
                    if explanation:
                        # Format as a bullet point
                        bullet_text = f"• {explanation}"
                        content.append(Paragraph(bullet_text, self.styles['TopicContent']))
            else:
                content.append(Paragraph(
                    "• Detailed content available in the video at the specified timestamp.",
                    self.styles['TopicContent']
                ))
            
            # Source info
            source_info = f"📊 Extracted from {topic.get('source_segments_count', 0)} transcript segments"
            content.append(Paragraph(source_info, self.styles['SummaryInfo']))
            
            content.append(Spacer(1, 0.3*inch))
            
            # Add page break every 4-5 topics to avoid crowding
            if i % 4 == 0 and i < len(topics):
                content.append(PageBreak())
        
        return content
    
    def _build_footer_info(self, transcript_file: str) -> List:
        """Build footer information"""
        content = []
        
        content.append(PageBreak())
        content.append(Paragraph("ℹ️ Generation Information", self.styles['Heading2']))
        content.append(Spacer(1, 0.1*inch))
        
        # Generation details
        details = [
            f"📄 Source Transcript: {os.path.basename(transcript_file)}",
            f"🤖 Generated by: AI-Powered Video Lecture Summarizer",
            f"👤 User: Atulk47",
            f"📅 Generation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"🔧 Content Extraction: Semantic similarity matching with transcript overlap analysis",
            f"📋 Format: Topic name, timestamp range, and explanatory content from lecture",
        ]
        
        for detail in details:
            content.append(Paragraph(detail, self.styles['SummaryInfo']))
            content.append(Spacer(1, 0.1*inch))
        
        # Disclaimer
        disclaimer = """
        <b>Note:</b> This summary is automatically generated from the video lecture transcript. 
        The explanatory content is extracted directly from the speaker's words during the 
        corresponding time segments. For complete understanding, please refer to the original 
        video content at the specified timestamps.
        """
        content.append(Spacer(1, 0.2*inch))
        content.append(Paragraph(disclaimer, self.styles['Normal']))
        
        return content


def main():
    """Test the enhanced PDF summary generator"""
    generator = LecturePDFSummaryGenerator()
    
    # Example usage
    transcript_file = input("Enter transcript file path: ").strip()
    topics_file = input("Enter topics file path: ").strip()
    
    if not os.path.exists(transcript_file):
        print(f"❌ Transcript file not found: {transcript_file}")
        return
    
    if not os.path.exists(topics_file):
        print(f"❌ Topics file not found: {topics_file}")
        return
    
    video_title = input("Enter video title (default: Digital Image Processing Lecture): ").strip()
    if not video_title:
        video_title = "Digital Image Processing Lecture"
    
    output_path = input("Enter output PDF path (default: lecture_summary.pdf): ").strip()
    if not output_path:
        output_path = "lecture_summary.pdf"
    
    print(f"\n🚀 Generating PDF summary...")
    print(f"📄 Topics: {topics_file}")
    print(f"📄 Transcript: {transcript_file}")
    print(f"🎯 Title: {video_title}")
    print("-" * 60)
    
    # Generate PDF
    result = generator.generate_pdf_summary(
        topics_file=topics_file,
        transcript_file=transcript_file,
        output_path=output_path,
        video_title=video_title
    )
    
    if result:
        print(f"✅ PDF generated successfully: {result}")
        print(f"📁 File location: {os.path.abspath(result)}")
    else:
        print("❌ Failed to generate PDF")


if __name__ == "__main__":
    # Check dependencies
    try:
        import reportlab
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("💡 Install with: pip install reportlab sentence-transformers")
        exit(1)
    
    main()