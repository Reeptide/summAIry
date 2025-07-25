#!/usr/bin/env python3
"""
PDF Maker for Lecture Notes
Creates aesthetic, notebook-style PDFs with YAML configuration support
"""

import os
import glob
import yaml
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle, Image
from reportlab.lib.colors import Color, black, white, navy, darkblue, lightgrey, grey
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.pdfgen import canvas
from reportlab.platypus.frames import Frame
from reportlab.platypus.doctemplate import PageTemplate, BaseDocTemplate
import re
from datetime import datetime

class NotebookPageTemplate(PageTemplate):
    """Custom page template with notebook-style background"""

    def __init__(self, id, frames, **kwargs):
        PageTemplate.__init__(self, id, frames, **kwargs)

    def beforeDrawPage(self, canvas, doc):
        """Draw notebook-style background"""
        canvas.saveState()
        width, height = letter

        # Cream notebook background
        canvas.setFillColor(Color(0.98, 0.97, 0.94, 1))
        canvas.rect(0, 0, width, height, fill=1, stroke=0)

        # Red margin line
        canvas.setStrokeColor(Color(0.8, 0.2, 0.2, 0.6))
        canvas.setLineWidth(1.5)
        canvas.line(80, 0, 80, height)

        # Horizontal ruling
        canvas.setStrokeColor(Color(0.7, 0.8, 0.9, 0.3))
        canvas.setLineWidth(0.5)
        line_spacing = 24
        y_start = 50
        while y_start < height - 50:
            canvas.line(80, y_start, width - 50, y_start)
            y_start += line_spacing

        # Hole punches
        canvas.setFillColor(Color(0.9, 0.9, 0.9, 1))
        punch_y_positions = [height - 150, height - 300, height - 450, height - 600]
        for y_pos in punch_y_positions:
            if y_pos > 50:
                canvas.circle(25, y_pos, 8, fill=1, stroke=1)
                canvas.setFillColor(white)
                canvas.circle(25, y_pos, 6, fill=1, stroke=0)
                canvas.setFillColor(Color(0.9, 0.9, 0.9, 1))

        canvas.restoreState()

class PDFMaker:
    def __init__(self, config_path="config/config.yaml"):
        self.config = self.load_config(config_path)
        self.setup_paths()
        self.setup_styles()

    def load_config(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print(f"✅ Loaded configuration from {config_path}")
            return config
        except FileNotFoundError:
            print(f"❌ Config file {config_path} not found!")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML config: {e}")
            raise

    def setup_paths(self):
        base_dir = self.config.get('base_directory', os.getcwd())
        self.pdf_transcript_dir = os.path.join(base_dir, self.config['directories']['pdf_transcript'])
        self.transcript_only_dir = os.path.join(base_dir, self.config['directories']['transcript_only'])
        self.course_name = self.config['course']['name']
        self.course_abbrev = self.config['course']['abbreviation']
        self.notes_pattern = self.config['file_patterns']['notes_pattern']
        self.notes_regex = self.config['file_patterns']['notes_regex']
        self.filename_prefix = self.config['output']['filename_prefix']
        self.topics_file = self.config.get('topics_file', 'topics.txt')

    def setup_styles(self):
        self.styles = getSampleStyleSheet()
        self.styles.add(ParagraphStyle(
            name='NotebookTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=navy,
            spaceAfter=25,
            spaceBefore=10,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            wordWrap='CJK',
            leading=28
        ))
        self.styles.add(ParagraphStyle(
            name='TopicPageTitle',
            parent=self.styles['Title'],
            fontSize=20,
            textColor=navy,
            spaceAfter=20,
            spaceBefore=15,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold',
            wordWrap='CJK',
            leading=24
        ))
        self.styles.add(ParagraphStyle(
            name='NotebookSectionHeader',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=navy,
            spaceBefore=20,
            spaceAfter=10,
            fontName='Helvetica-Bold',
            alignment=TA_LEFT,
            leftIndent=0,
            borderWidth=0,
            borderPadding=0,
            backColor=None
        ))
        self.styles.add(ParagraphStyle(
            name='NotebookSubHeader',
            parent=self.styles['Heading2'],
            fontSize=13,
            textColor=Color(0.2, 0.3, 0.6, 1),
            spaceBefore=15,
            spaceAfter=6,
            fontName='Helvetica-Bold',
            leftIndent=5
        ))
        self.styles.add(ParagraphStyle(
            name='NotebookBody',
            parent=self.styles['Normal'],
            fontSize=11,
            textColor=Color(0.15, 0.15, 0.15, 1),
            spaceBefore=6,
            spaceAfter=8,
            alignment=TA_JUSTIFY,
            leftIndent=10,
            rightIndent=10,
            fontName='Helvetica',
            leading=14
        ))
        self.styles.add(ParagraphStyle(
            name='NotebookCode',
            parent=self.styles['Code'],
            fontSize=10,
            textColor=Color(0.3, 0.5, 0.3, 1),
            spaceBefore=8,
            spaceAfter=8,
            leftIndent=20,
            rightIndent=20,
            fontName='Courier',
            backColor=Color(0.97, 0.99, 0.97, 1),
            borderWidth=0,
            borderColor=None,
            borderPadding=6
        ))
        self.styles.add(ParagraphStyle(
            name='TOCTitle',
            parent=self.styles['Title'],
            fontSize=28,
            textColor=navy,
            spaceAfter=30,
            alignment=TA_CENTER,
            fontName='Helvetica-Bold'
        ))
        self.styles.add(ParagraphStyle(
            name='TOCEntry',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=Color(0.2, 0.2, 0.2, 1),
            spaceBefore=4,
            spaceAfter=4,
            leftIndent=0,
            fontName='Helvetica',
            leading=16
        ))

    def format_text_with_markdown(self, text):
        text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
        text = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', text)
        text = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', text)
        return text

    def extract_start_time_for_sorting(self, topic_with_time):
        timing_match = re.search(r'\[(\d{2}):(\d{2})-\d{2}:\d{2}\]$', topic_with_time)
        if timing_match:
            minutes = int(timing_match.group(1))
            seconds = int(timing_match.group(2))
            return minutes * 60 + seconds
        return 0

    def load_topics_from_file(self):
        if not os.path.exists(self.topics_file):
            print(f"⚠️  Topics file {self.topics_file} not found. Will auto-discover from files.")
            return None
        topics = []
        try:
            with open(self.topics_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        topics.append(line)
            print(f"✅ Loaded {len(topics)} topics from {self.topics_file}")
            return topics
        except Exception as e:
            print(f"❌ Error reading topics file {self.topics_file}: {e}")
            return None

    def discover_topics_from_files(self):
        topics = []
        directories = [self.pdf_transcript_dir, self.transcript_only_dir]
        for directory in directories:
            if os.path.exists(directory):
                pattern = os.path.join(directory, self.notes_pattern)
                files = glob.glob(pattern)
                for file_path in files:
                    filename = os.path.basename(file_path)
                    match = re.match(self.notes_regex, filename)
                    if match:
                        topic_name = match.group(1)
                        topic_display = topic_name.replace('_', ' ')
                        timing_match = re.search(r'(\d{4})-(\d{4})\.md$', filename)
                        if timing_match:
                            start_time = timing_match.group(1)
                            end_time = timing_match.group(2)
                            start_formatted = f"{start_time[:2]}:{start_time[2:]}"
                            end_formatted = f"{end_time[:2]}:{end_time[2:]}"
                            topic_with_time = f"{topic_display} [{start_formatted}-{end_formatted}]"
                            if topic_with_time not in topics:
                                topics.append(topic_with_time)
                break
        topics.sort(key=self.extract_start_time_for_sorting)
        return topics

    def get_topics(self):
        topics = self.load_topics_from_file()
        if not topics:
            print("🔍 Auto-discovering topics from files...")
            topics = self.discover_topics_from_files()
        return topics

    def find_file_for_topic(self, topic, directory):
        core_topic = re.sub(r'\s*\[\d{2}:\d{2}-\d{2}:\d{2}\]$', '', topic)
        topic_filename = core_topic.replace(' ', '_')
        pattern = os.path.join(directory, f"notes_*_{topic_filename}_*.md")
        matching_files = glob.glob(pattern)
        if matching_files:
            return matching_files[0]
        return None

    def add_logo_if_exists(self):
        logo_path = "logo.png"
        if os.path.exists(logo_path):
            try:
                logo = Image(logo_path)
                logo._restrictSize(2*inch, 1.5*inch)
                return logo
            except Exception as e:
                print(f"⚠️  Error loading logo: {e}")
                return None
        else:
            print("ℹ️  No logo.png found, skipping logo")
            return None

    def create_table_of_contents(self, topics):
        toc_elements = []
        logo = self.add_logo_if_exists()
        if logo:
            logo_table = Table([[logo]], colWidths=[6*inch])
            logo_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            toc_elements.append(logo_table)
            toc_elements.append(Spacer(1, 20))
        main_title = f"📚 {self.course_abbrev} Lecture Notes"
        toc_elements.append(Paragraph(main_title, self.styles['TOCTitle']))
        toc_elements.append(Spacer(1, 20))
        course_info = f"""
        <b>{self.course_name}</b><br/>
        <i>Generated on {datetime.now().strftime('%B %d, %Y')}</i><br/>
        {len(topics)} Topics Covered
        """
        toc_elements.append(Paragraph(course_info, self.styles['Normal']))
        toc_elements.append(Spacer(1, 40))
        toc_data = []
        header_style = ParagraphStyle(
            'TOCHeader',
            parent=self.styles['Normal'],
            fontSize=14,
            textColor=navy,
            fontName='Helvetica-Bold',
            alignment=TA_CENTER
        )
        toc_data.append([
            Paragraph("Topic", header_style),
            Paragraph("Duration", header_style)
        ])
        for i, topic in enumerate(topics, 1):
            timing_match = re.search(r'\[(\d{2}:\d{2}-\d{2}:\d{2})\]$', topic)
            timing = timing_match.group(1) if timing_match else "N/A"
            topic_name = re.sub(r'\s*\[\d{2}:\d{2}-\d{2}:\d{2}\]$', '', topic)
            topic_cell = Paragraph(f"{i}. {topic_name}", self.styles['TOCEntry'])
            timing_cell = Paragraph(timing, self.styles['TOCEntry'])
            toc_data.append([topic_cell, timing_cell])
        toc_table = Table(toc_data, colWidths=[4.5*inch, 1.5*inch])
        toc_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), Color(0.95, 0.95, 1, 1)),
            ('TEXTCOLOR', (0, 0), (-1, 0), navy),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('TOPPADDING', (0, 0), (-1, 0), 12),
            ('ALIGN', (0, 1), (0, -1), 'LEFT'),
            ('ALIGN', (1, 1), (1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 11),
            ('LEFTPADDING', (0, 1), (-1, -1), 8),
            ('RIGHTPADDING', (0, 1), (-1, -1), 8),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, Color(0.98, 0.98, 0.98, 1)]),
            ('GRID', (0, 0), (-1, -1), 0.5, Color(0.8, 0.8, 0.8, 1)),
            ('LINEBELOW', (0, 0), (-1, 0), 2, navy),
        ]))
        toc_elements.append(toc_table)
        toc_elements.append(PageBreak())
        return toc_elements

    def parse_markdown_content(self, content):
        elements = []
        lines = content.split('\n')
        current_section = ""
        current_content = []
        for line in lines:
            line = line.strip()
            if line.startswith('**Model:**') or line.startswith('**Mode:**') or line.startswith('**Chunks Retrieved:**'):
                continue
            if line.startswith('# '):
                if current_content:
                    elements.append((current_section, '\n'.join(current_content)))
                current_section = 'title'
                current_content = [line[2:]]
            elif line.startswith('## '):
                if current_content:
                    elements.append((current_section, '\n'.join(current_content)))
                current_section = line[3:]
                current_content = []
            elif line.startswith('### '):
                current_content.append(f"SUBHEADING:{line[4:]}")
            elif line.startswith('**Topic:**'):
                topic_match = re.search(r'\*\*Topic:\*\* (.+)', line)
                if topic_match:
                    elements.append(('topic', topic_match.group(1)))
            elif line:
                current_content.append(line)
        if current_content:
            elements.append((current_section, '\n'.join(current_content)))
        return elements

    def format_title_with_better_breaks(self, title):
        if len(title) <= 50:
            return title
        words = title.split()
        lines = []
        current_line = []
        current_length = 0
        for word in words:
            if current_length + len(word) + 1 > 35 and current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_length = len(word)
            else:
                current_line.append(word)
                current_length += len(word) + 1
        if current_line:
            lines.append(' '.join(current_line))
        return '<br/>'.join(lines)

    def process_content_to_elements(self, elements):
        pdf_elements = []
        main_title = None
        topic_info = None
        for section, content in elements:
            if section == 'title':
                main_title = content
            elif section == 'topic':
                topic_info = content
        if topic_info:
            formatted_title = self.format_title_with_better_breaks(topic_info)
            pdf_elements.append(Paragraph(formatted_title, self.styles['TopicPageTitle']))
            pdf_elements.append(Spacer(1, 25))
        elif main_title:
            formatted_title = self.format_title_with_better_breaks(main_title)
            pdf_elements.append(Paragraph(formatted_title, self.styles['TopicPageTitle']))
            pdf_elements.append(Spacer(1, 25))
        for section, content in elements:
            if section in ['title', 'topic'] or not section or not content:
                continue
            section_clean = section.strip()
            if section_clean:
                pdf_elements.append(Paragraph(f"● {section_clean}", self.styles['NotebookSectionHeader']))
                pdf_elements.append(Spacer(1, 8))
            content_lines = content.split('\n')
            current_paragraph = []
            for line in content_lines:
                line = line.strip()
                if not line:
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        formatted_text = self.format_text_with_markdown(text)
                        pdf_elements.append(Paragraph(formatted_text, self.styles['NotebookBody']))
                        current_paragraph = []
                    pdf_elements.append(Spacer(1, 4))
                    continue
                if line.startswith('SUBHEADING:'):
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        formatted_text = self.format_text_with_markdown(text)
                        pdf_elements.append(Paragraph(formatted_text, self.styles['NotebookBody']))
                        current_paragraph = []
                    subheading = line.replace('SUBHEADING:', '')
                    pdf_elements.append(Paragraph(f"▸ {subheading}", self.styles['NotebookSubHeader']))
                    continue
                if line.startswith('* ') or line.startswith('- '):
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        formatted_text = self.format_text_with_markdown(text)
                        pdf_elements.append(Paragraph(formatted_text, self.styles['NotebookBody']))
                        current_paragraph = []
                    bullet_text = line[2:]
                    formatted_bullet = self.format_text_with_markdown(bullet_text)
                    pdf_elements.append(Paragraph(f"• {formatted_bullet}", self.styles['NotebookBody']))
                    continue
                if '=' in line and ('(' in line or 'f(' in line):
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        formatted_text = self.format_text_with_markdown(text)
                        pdf_elements.append(Paragraph(formatted_text, self.styles['NotebookBody']))
                        current_paragraph = []
                    pdf_elements.append(Paragraph(line, self.styles['NotebookCode']))
                    continue
                if line.startswith('*Disclaimer') or 'AI-generated' in line:
                    if current_paragraph:
                        text = ' '.join(current_paragraph)
                        formatted_text = self.format_text_with_markdown(text)
                        pdf_elements.append(Paragraph(formatted_text, self.styles['NotebookBody']))
                        current_paragraph = []
                    disclaimer_style = ParagraphStyle(
                        'Disclaimer',
                        parent=self.styles['Normal'],
                        fontSize=9,
                        textColor=Color(0.6, 0.6, 0.6, 1),
                        spaceBefore=15,
                        spaceAfter=10,
                        alignment=TA_CENTER,
                        fontName='Helvetica-Oblique'
                    )
                    pdf_elements.append(Paragraph(f"<i>{line}</i>", disclaimer_style))
                    continue
                current_paragraph.append(line)
            if current_paragraph:
                text = ' '.join(current_paragraph)
                formatted_text = self.format_text_with_markdown(text)
                pdf_elements.append(Paragraph(formatted_text, self.styles['NotebookBody']))
            pdf_elements.append(Spacer(1, 18))
        return pdf_elements

    def create_pdf(self, output_path, topics, note_type):
        doc = BaseDocTemplate(str(output_path), pagesize=letter)
        frame = Frame(
            90, 50, letter[0] - 140, letter[1] - 100,
            leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
        )
        template = NotebookPageTemplate('notebook', [frame])
        doc.addPageTemplates([template])
        story = []
        story.extend(self.create_table_of_contents(topics))
        directory = self.pdf_transcript_dir if note_type == 'pdf_transcript' else self.transcript_only_dir
        for topic in topics:
            file_path = self.find_file_for_topic(topic, directory)
            if file_path and os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    elements = self.parse_markdown_content(content)
                    pdf_elements = self.process_content_to_elements(elements)
                    story.extend(pdf_elements)
                    story.append(PageBreak())
                    print(f"✅ Added: {topic}")
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {e}")
                    story.append(Paragraph(f"📝 {topic}", self.styles['TopicPageTitle']))
                    story.append(Paragraph(f"Error loading content: {str(e)}", self.styles['NotebookBody']))
                    story.append(PageBreak())
            else:
                print(f"⚠️  No file found for topic: {topic}")
                story.append(Paragraph(f"📝 {topic}", self.styles['TopicPageTitle']))
                story.append(Paragraph("Content not available", self.styles['NotebookBody']))
                story.append(PageBreak())
        try:
            doc.build(story)
            return True
        except Exception as e:
            print(f"❌ Error building PDF: {e}")
            return False

    def run(self):
        print("🚀 Starting PDF Generation...")
        print("=" * 50)
        print(f"📚 Course: {self.course_name} ({self.course_abbrev})")
        print(f"📁 PDF Transcript Dir: {self.pdf_transcript_dir}")
        print(f"📁 Transcript Only Dir: {self.transcript_only_dir}")
        print(f"📄 Topics File: {self.topics_file}")

        topics = self.get_topics()
        print(f"📚 Found {len(topics)} topics (sorted chronologically):")
        for i, topic in enumerate(topics, 1):
            print(f"  {i}. {topic}")

        if not topics:
            print("❌ No topics found! Check configuration and files.")
            return

        output_dir = Path('generated_pdfs')
        output_dir.mkdir(exist_ok=True)
        success_count = 0

        pdf_transcript_path = output_dir / f'{self.filename_prefix}_PDF_Transcript.pdf'
        print(f"\n📄 Creating PDF+Transcript version...")
        if self.create_pdf(pdf_transcript_path, topics, 'pdf_transcript'):
            print(f"✅ Successfully created: {pdf_transcript_path}")
            success_count += 1
        else:
            print(f"❌ Failed to create: {pdf_transcript_path}")

        transcript_only_path = output_dir / f'{self.filename_prefix}_Transcript_Only.pdf'
        print(f"\n📄 Creating Transcript-Only version...")
        if self.create_pdf(transcript_only_path, topics, 'transcript_only'):
            print(f"✅ Successfully created: {transcript_only_path}")
            success_count += 1
        else:
            print(f"❌ Failed to create: {transcript_only_path}")

        print("\n" + "=" * 50)
        print(f"🎉 PDF Generation Complete!")
        print(f"📊 Success: {success_count}/2 PDFs created")
        print(f"📁 Output directory: {output_dir.absolute()}")

        if success_count == 2:
            print("🌟 All PDFs generated successfully!")
        elif success_count == 1:
            print("⚠️  One PDF generated successfully, check errors above")
        else:
            print("❌ PDF generation failed, check errors above")

if __name__ == "__main__":
    try:
        from reportlab.lib.pagesizes import letter
        import yaml
    except ImportError:
        print("Installing required packages...")
        os.system("pip install reportlab PyYAML")
        from reportlab.lib.pagesizes import letter
        import yaml

    pdf_maker = PDFMaker()
    pdf_maker.run()