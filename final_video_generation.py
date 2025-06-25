import os
import re
import json
import base64
from datetime import datetime
import tempfile
import shutil

class CompactSideBySideTimestampedVideoPlayer:
    def __init__(self):
        self.topics = []
        self.video_path = ""
        self.output_dir = "timestamped_video_output"
        
    def parse_timestamp(self, time_str):
        """Convert MM:SS or HH:MM:SS to seconds"""
        parts = time_str.split(':')
        if len(parts) == 2:  # MM:SS
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        elif len(parts) == 3:  # HH:MM:SS
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        return 0
    
    def seconds_to_timestamp(self, seconds):
        """Convert seconds back to MM:SS format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"
    
    def parse_topic_file(self, topic_file_path):
        """Parse the topic segmentation file"""
        print(f"📄 Parsing topic file: {topic_file_path}")
        
        try:
            with open(topic_file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
        except Exception as e:
            print(f"❌ Error reading topic file: {e}")
            return False
        
        lines = content.split('\n')
        topics = []
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            topic_name = None
            timestamp_part = None
            
            # Try format 1: "*Topic Name**: [timestamp]"
            if '**: [' in line and line.endswith(']'):
                parts = line.split('**: [')
                if len(parts) == 2:
                    topic_name = parts[0].strip()
                    if topic_name.startswith('*'):
                        topic_name = topic_name[1:].strip()
                    timestamp_part = parts[1].rstrip(']')
            
            # Try format 2: "Topic Name : [timestamp]"
            elif ' : [' in line and line.endswith(']'):
                parts = line.split(' : [')
                if len(parts) == 2:
                    topic_name = parts[0].strip()
                    timestamp_part = parts[1].rstrip(']')
            
            if topic_name and timestamp_part:
                if ' - ' in timestamp_part:
                    start_time_str, end_time_str = timestamp_part.split(' - ')
                    start_seconds = self.parse_timestamp(start_time_str.strip())
                    end_seconds = self.parse_timestamp(end_time_str.strip())
                    
                    topics.append({
                        'name': topic_name,
                        'start_time': start_time_str.strip(),
                        'end_time': end_time_str.strip(),
                        'start_seconds': start_seconds,
                        'end_seconds': end_seconds,
                        'duration': end_seconds - start_seconds
                    })
                    
                    print(f"  ✅ Topic {len(topics)}: {topic_name} ({start_time_str.strip()} - {end_time_str.strip()})")
        
        if not topics:
            print("❌ No topics found in the file")
            return False
        
        topics.sort(key=lambda x: x['start_seconds'])
        self.topics = topics
        
        print(f"✅ Successfully parsed {len(topics)} topics")
        return True
    
    def copy_video_file(self, video_path):
        """Copy video file to output directory"""
        if not os.path.exists(video_path):
            print(f"❌ Video file not found: {video_path}")
            return None
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        _, ext = os.path.splitext(video_path)
        new_video_path = os.path.join(self.output_dir, f"video{ext}")
        
        print(f"📹 Copying video to output directory...")
        try:
            shutil.copy2(video_path, new_video_path)
            print(f"✅ Video copied successfully")
        except Exception as e:
            print(f"❌ Error copying video: {e}")
            return None
        
        self.video_path = f"video{ext}"
        return new_video_path
    
    def generate_css(self):
        """Generate CSS styles for compact side-by-side layout"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Roboto', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #0f0f0f;
            color: #ffffff;
            line-height: 1.6;
            overflow-x: hidden;
            height: 100vh;
        }

        .main-container {
            display: flex;
            height: 100vh;
            width: 100%;
            background: #0f0f0f;
        }

        /* Video Section - Left Side - MORE SPACE */
        .video-section {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 16px;
            background: #0f0f0f;
            min-width: 0;
            overflow-y: auto;
        }

        .video-container {
            position: relative;
            width: 100%;
            background: #000;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.6);
            margin-bottom: 12px;
        }

        .video-wrapper {
            position: relative;
            width: 100%;
            aspect-ratio: 16/9;
            background: #000;
        }

        video {
            width: 100%;
            height: 100%;
            object-fit: contain;
            cursor: pointer;
            background: #000;
        }

        .video-loading {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #fff;
            font-size: 18px;
            z-index: 5;
        }

        .video-error {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #ff4444;
            font-size: 16px;
            text-align: center;
            z-index: 5;
            padding: 20px;
            background: rgba(0, 0, 0, 0.8);
            border-radius: 8px;
        }

        .custom-controls {
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: linear-gradient(transparent, rgba(0,0,0,0.9));
            padding: 20px;
            z-index: 10;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .video-container:hover .custom-controls {
            opacity: 1;
        }

        .progress-container {
            position: relative;
            height: 8px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 4px;
            margin-bottom: 16px;
            cursor: pointer;
            overflow: visible;
        }

        .progress-bar {
            height: 100%;
            background: #ff0000;
            border-radius: 4px;
            transition: width 0.1s ease;
            position: relative;
            z-index: 3;
        }

        .progress-handle {
            position: absolute;
            right: -8px;
            top: 50%;
            transform: translateY(-50%);
            width: 16px;
            height: 16px;
            background: #ff0000;
            border-radius: 50%;
            opacity: 0;
            transition: opacity 0.3s ease;
            z-index: 4;
        }

        .progress-container:hover .progress-handle {
            opacity: 1;
        }

        .timeline-segments {
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            z-index: 5;
        }

        .timeline-segment {
            position: absolute;
            height: 100%;
            border-right: 2px solid rgba(255, 255, 255, 0.4);
            cursor: pointer;
            transition: all 0.2s ease;
            z-index: 6;
        }

        .timeline-segment:last-child {
            border-right: none;
        }

        .timeline-segment:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .timeline-segment.active {
            background: rgba(255, 0, 0, 0.4);
            border-right-color: rgba(255, 255, 255, 0.8);
        }

        .segment-tooltip {
            position: absolute;
            bottom: 150%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 8px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.2s ease;
            z-index: 9999;
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.8);
            max-width: 250px;
            white-space: normal;
            text-align: center;
            line-height: 1.4;
        }

        .timeline-segment:hover .segment-tooltip {
            opacity: 1;
        }

        .segment-tooltip::after {
            content: '';
            position: absolute;
            top: 100%;
            left: 50%;
            transform: translateX(-50%);
            border: 6px solid transparent;
            border-top-color: rgba(0, 0, 0, 0.9);
        }

        .controls-row {
            display: flex;
            align-items: center;
            gap: 16px;
        }

        .play-btn {
            background: none;
            border: none;
            color: white;
            font-size: 28px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 52px;
            height: 52px;
            border-radius: 50%;
            transition: all 0.3s ease;
        }

        .play-btn:hover {
            background: rgba(255, 255, 255, 0.15);
            transform: scale(1.05);
        }

        .time-display {
            color: white;
            font-size: 15px;
            font-family: 'Roboto Mono', monospace;
            min-width: 120px;
            font-weight: 500;
        }

        .volume-control {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .volume-btn {
            background: none;
            border: none;
            color: white;
            font-size: 22px;
            cursor: pointer;
            padding: 10px;
            border-radius: 6px;
            transition: all 0.3s ease;
        }

        .volume-btn:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        .fullscreen-btn {
            background: none;
            border: none;
            color: white;
            font-size: 22px;
            cursor: pointer;
            padding: 10px;
            border-radius: 6px;
            transition: all 0.3s ease;
            margin-left: auto;
        }

        .fullscreen-btn:hover {
            background: rgba(255, 255, 255, 0.15);
        }

        /* COMPACT Video Info Section */
        .video-info {
            background: #1a1a1a;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            flex-shrink: 0;
        }

        .video-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 8px;
            color: #ffffff;
            line-height: 1.2;
        }

        .video-meta {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #aaaaaa;
            font-size: 12px;
            margin-bottom: 10px;
            flex-wrap: wrap;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 4px;
            background: rgba(255, 255, 255, 0.05);
            padding: 4px 8px;
            border-radius: 12px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            font-size: 11px;
        }

        /* COMPACT Currently Playing Section */
        .current-topic-display {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            padding: 10px 12px;
            border-radius: 6px;
            border-left: 3px solid #ff0000;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }

        .current-topic-display h3 {
            color: #ff0000;
            font-size: 0.85rem;
            margin-bottom: 4px;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .current-topic-display p {
            color: #cccccc;
            margin: 2px 0;
            font-size: 0.8rem;
        }

        .current-topic-title {
            font-weight: 600;
            font-size: 0.85rem;
            color: #ffffff;
        }

        /* SMALLER Table of Contents Section - Right Side */
        .toc-section {
            width: 320px;
            min-width: 320px;
            max-width: 320px;
            background: #181818;
            border-left: 2px solid #303030;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
            flex-shrink: 0;
        }

        .toc-header {
            padding: 16px 16px;
            border-bottom: 1px solid #303030;
            background: #1a1a1a;
            flex-shrink: 0;
        }

        .toc-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: #ffffff;
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }

        .toc-subtitle {
            color: #aaaaaa;
            font-size: 0.8rem;
            font-weight: 400;
        }

        .topics-container {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }

        .topic-item {
            background: #1f1f1f;
            margin-bottom: 8px;
            border-radius: 8px;
            transition: all 0.3s ease;
            cursor: pointer;
            border: 1px solid transparent;
            overflow: hidden;
        }

        .topic-item:hover {
            background: #2a2a2a;
            border-color: #404040;
            transform: translateX(3px);
        }

        .topic-item.active {
            background: linear-gradient(135deg, #ff0000, #cc0000);
            border-color: #ff0000;
            transform: translateX(4px);
            box-shadow: 0 3px 15px rgba(255, 0, 0, 0.3);
        }

        .topic-content {
            padding: 12px;
        }

        .topic-header {
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            gap: 8px;
            margin-bottom: 6px;
        }

        .topic-number {
            background: rgba(255, 255, 255, 0.1);
            color: #aaaaaa;
            font-size: 10px;
            padding: 3px 6px;
            border-radius: 10px;
            font-weight: 600;
            min-width: 24px;
            text-align: center;
            flex-shrink: 0;
        }

        .topic-item.active .topic-number {
            background: rgba(255, 255, 255, 0.2);
            color: #ffffff;
        }

        .topic-name {
            font-weight: 600;
            color: #ffffff;
            font-size: 0.85rem;
            line-height: 1.3;
            flex: 1;
            word-wrap: break-word;
        }

        .topic-item.active .topic-name {
            color: #ffffff;
        }

        .topic-time-info {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.75rem;
            color: #aaaaaa;
            font-family: 'Roboto Mono', monospace;
        }

        .topic-item.active .topic-time-info {
            color: #ffcccc;
        }

        .topic-duration {
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 5px;
            border-radius: 8px;
            font-size: 0.7rem;
        }

        .topic-item.active .topic-duration {
            background: rgba(255, 255, 255, 0.2);
        }

        /* COMPACT Shortcuts Panel */
        .shortcuts-panel {
            padding: 12px;
            background: #1a1a1a;
            border-top: 1px solid #303030;
            flex-shrink: 0;
        }

        .shortcuts-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #ffffff;
            margin-bottom: 8px;
            text-align: center;
        }

        .shortcuts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 4px;
            font-size: 0.7rem;
            color: #aaaaaa;
        }

        .shortcut-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .shortcut-key {
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 5px;
            border-radius: 3px;
            font-family: monospace;
            font-size: 0.65rem;
        }

        /* Loading spinner */
        .loading-spinner {
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid #ff0000;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 16px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        /* Responsive Design - ONLY for very small screens */
        @media (max-width: 800px) {
            .main-container {
                flex-direction: column;
                height: auto;
            }
            
            .toc-section {
                width: 100%;
                min-width: 100%;
                max-width: 100%;
                height: 350px;
                border-left: none;
                border-top: 2px solid #303030;
            }

            .video-section {
                height: auto;
            }
        }

        @media (max-width: 600px) {
            .video-section {
                padding: 10px;
            }
            
            .video-title {
                font-size: 1.2rem;
            }
            
            .toc-section {
                height: 300px;
            }
        }

        /* Scrollbar Styling */
        .topics-container::-webkit-scrollbar {
            width: 5px;
        }

        .topics-container::-webkit-scrollbar-track {
            background: #1a1a1a;
        }

        .topics-container::-webkit-scrollbar-thumb {
            background: #404040;
            border-radius: 3px;
        }

        .topics-container::-webkit-scrollbar-thumb:hover {
            background: #505050;
        }

        /* Focus States */
        .topic-item:focus,
        .play-btn:focus,
        .volume-btn:focus,
        .fullscreen-btn:focus {
            outline: 2px solid #ff0000;
            outline-offset: 2px;
        }
        """
    
    def generate_javascript(self):
        """Generate JavaScript - keeping all original functionality"""
        topics_json = json.dumps(self.topics)
        
        return f"""
        class CompactSideBySideVideoPlayer {{
            constructor() {{
                this.video = document.getElementById('mainVideo');
                this.topics = {topics_json};
                this.currentTopicIndex = -1;
                this.isPlaying = false;
                this.isDragging = false;
                this.isVideoReady = false;
                
                this.progressBar = document.querySelector('.progress-bar');
                this.progressContainer = document.querySelector('.progress-container');
                this.currentTopicDisplay = document.getElementById('currentTopicDisplay');
                this.playBtn = document.getElementById('playBtn');
                this.timeDisplay = document.getElementById('timeDisplay');
                
                this.init();
            }}
            
            init() {{
                this.showLoadingState();
                this.attachEventListeners();
                this.updateDisplay();
                
                console.log('🎬 Compact Side-by-Side Video Player initialized');
                console.log(`📊 Loaded ${{this.topics.length}} topics`);
                
                if (this.video.readyState >= 2) {{
                    this.onVideoReady();
                }}
            }}
            
            showLoadingState() {{
                const videoWrapper = document.querySelector('.video-wrapper');
                const existingLoader = videoWrapper.querySelector('.video-loading');
                
                if (!existingLoader) {{
                    const loader = document.createElement('div');
                    loader.className = 'video-loading';
                    loader.innerHTML = `
                        <div class="loading-spinner"></div>
                        <div>Loading video...</div>
                    `;
                    videoWrapper.appendChild(loader);
                }}
            }}
            
            hideLoadingState() {{
                const loader = document.querySelector('.video-loading');
                if (loader) {{
                    loader.remove();
                }}
            }}
            
            showErrorState(message) {{
                const videoWrapper = document.querySelector('.video-wrapper');
                const existingError = videoWrapper.querySelector('.video-error');
                
                if (!existingError) {{
                    const error = document.createElement('div');
                    error.className = 'video-error';
                    error.innerHTML = `
                        <div>❌ Video Error</div>
                        <div>${{message}}</div>
                        <div><small>Please check the video file path and format</small></div>
                    `;
                    videoWrapper.appendChild(error);
                }}
            }}
            
            onVideoReady() {{
                this.isVideoReady = true;
                this.hideLoadingState();
                this.createTimelineSegments();
                this.updateDisplay();
                console.log('✅ Video loaded successfully');
            }}
            
            createTimelineSegments() {{
                if (!this.isVideoReady) {{
                    console.warn('Video not ready, postponing segment creation');
                    return;
                }}
                
                const segmentsContainer = document.querySelector('.timeline-segments');
                segmentsContainer.innerHTML = '';
                
                const totalDuration = this.getTotalDuration();
                
                if (totalDuration <= 0) {{
                    console.warn('Invalid video duration, using estimated duration');
                    return;
                }}
                
                this.topics.forEach((topic, index) => {{
                    const segment = document.createElement('div');
                    segment.className = 'timeline-segment';
                    segment.dataset.topicIndex = index;
                    
                    const startPercent = (topic.start_seconds / totalDuration) * 100;
                    const endPercent = (topic.end_seconds / totalDuration) * 100;
                    const widthPercent = endPercent - startPercent;
                    
                    segment.style.left = startPercent + '%';
                    segment.style.width = widthPercent + '%';
                    
                    const tooltip = document.createElement('div');
                    tooltip.className = 'segment-tooltip';
                    
                    let displayName = topic.name;
                    if (displayName.length > 35) {{
                        displayName = displayName.substring(0, 32) + '...';
                    }}
                    
                    tooltip.innerHTML = `
                        <strong>${{displayName}}</strong><br>
                        <small>${{topic.start_time}} - ${{topic.end_time}} (${{this.formatTime(topic.duration)}})</small>
                    `;
                    segment.appendChild(tooltip);
                    
                    segment.addEventListener('click', (e) => {{
                        e.stopPropagation();
                        this.jumpToTopic(index);
                    }});
                    
                    segmentsContainer.appendChild(segment);
                }});
                
                console.log(`📊 Created ${{this.topics.length}} timeline segments`);
            }}
            
            getTotalDuration() {{
                if (this.video.duration && !isNaN(this.video.duration) && this.video.duration > 0) {{
                    return this.video.duration;
                }}
                if (this.topics.length > 0) {{
                    return Math.max(...this.topics.map(t => t.end_seconds));
                }}
                return 3600;
            }}
            
            attachEventListeners() {{
                this.video.addEventListener('loadstart', () => {{
                    console.log('📺 Video loading started');
                    this.showLoadingState();
                }});
                
                this.video.addEventListener('loadeddata', () => {{
                    this.onVideoReady();
                }});
                
                this.video.addEventListener('loadedmetadata', () => {{
                    this.onVideoReady();
                }});
                
                this.video.addEventListener('canplay', () => {{
                    this.onVideoReady();
                }});
                
                this.video.addEventListener('error', (e) => {{
                    console.error('❌ Video error:', e);
                    const error = this.video.error;
                    let message = 'Unknown error occurred';
                    
                    if (error) {{
                        switch(error.code) {{
                            case error.MEDIA_ERR_ABORTED:
                                message = 'Video loading was aborted';
                                break;
                            case error.MEDIA_ERR_NETWORK:
                                message = 'Network error while loading video';
                                break;
                            case error.MEDIA_ERR_DECODE:
                                message = 'Video decoding error';
                                break;
                            case error.MEDIA_ERR_SRC_NOT_SUPPORTED:
                                message = 'Video format not supported';
                                break;
                        }}
                    }}
                    
                    this.hideLoadingState();
                    this.showErrorState(message);
                }});
                
                this.video.addEventListener('timeupdate', () => {{
                    if (this.isVideoReady) {{
                        this.updateProgress();
                        this.updateCurrentTopic();
                    }}
                }});
                
                this.video.addEventListener('play', () => {{
                    this.isPlaying = true;
                    this.updatePlayButton();
                }});
                
                this.video.addEventListener('pause', () => {{
                    this.isPlaying = false;
                    this.updatePlayButton();
                }});
                
                this.video.addEventListener('ended', () => {{
                    this.isPlaying = false;
                    this.updatePlayButton();
                }});
                
                this.playBtn.addEventListener('click', () => {{
                    this.togglePlayPause();
                }});
                
                this.progressContainer.addEventListener('click', (e) => {{
                    if (!this.isDragging && this.isVideoReady) {{
                        this.seekToPosition(e);
                    }}
                }});
                
                document.querySelectorAll('.topic-item').forEach((item, index) => {{
                    item.addEventListener('click', () => {{
                        this.jumpToTopic(index);
                    }});
                }});
                
                document.addEventListener('keydown', (e) => {{
                    this.handleKeyboard(e);
                }});
                
                this.video.addEventListener('click', () => {{
                    this.togglePlayPause();
                }});
            }}
            
            updateProgress() {{
                const duration = this.getTotalDuration();
                if (duration > 0) {{
                    const progress = (this.video.currentTime / duration) * 100;
                    this.progressBar.style.width = Math.min(progress, 100) + '%';
                }}
                
                this.updateTimeDisplay();
                this.updateActiveSegment();
            }}
            
            updateTimeDisplay() {{
                const current = this.formatTime(this.video.currentTime || 0);
                const total = this.formatTime(this.getTotalDuration());
                this.timeDisplay.textContent = `${{current}} / ${{total}}`;
            }}
            
            updateActiveSegment() {{
                const currentTime = this.video.currentTime || 0;
                document.querySelectorAll('.timeline-segment').forEach((segment, index) => {{
                    const topic = this.topics[index];
                    const isActive = currentTime >= topic.start_seconds && currentTime <= topic.end_seconds;
                    segment.classList.toggle('active', isActive);
                }});
            }}
            
            updateCurrentTopic() {{
                const currentTime = this.video.currentTime || 0;
                let newTopicIndex = -1;
                
                for (let i = 0; i < this.topics.length; i++) {{
                    const topic = this.topics[i];
                    if (currentTime >= topic.start_seconds && currentTime <= topic.end_seconds) {{
                        newTopicIndex = i;
                        break;
                    }}
                }}
                
                if (newTopicIndex !== this.currentTopicIndex) {{
                    this.currentTopicIndex = newTopicIndex;
                    this.highlightCurrentTopic();
                    this.updateCurrentTopicDisplay();
                }}
            }}
            
            highlightCurrentTopic() {{
                document.querySelectorAll('.topic-item').forEach((item, index) => {{
                    item.classList.toggle('active', index === this.currentTopicIndex);
                }});
                
                if (this.currentTopicIndex >= 0) {{
                    const activeItem = document.querySelector(`[data-topic-index="${{this.currentTopicIndex}}"]`);
                    if (activeItem && activeItem.classList.contains('topic-item')) {{
                        activeItem.scrollIntoView({{ 
                            behavior: 'smooth', 
                            block: 'center' 
                        }});
                    }}
                }}
            }}
            
            updateCurrentTopicDisplay() {{
                const display = this.currentTopicDisplay;
                
                if (this.currentTopicIndex >= 0) {{
                    const topic = this.topics[this.currentTopicIndex];
                    display.innerHTML = `
                        <h3>🎯 Currently Playing</h3>
                        <p class="current-topic-title">${{topic.name}}</p>
                        <p>⏱️ ${{topic.start_time}} - ${{topic.end_time}} (${{this.formatTime(topic.duration)}})</p>
                        <p>📍 Topic ${{this.currentTopicIndex + 1}} of ${{this.topics.length}}</p>
                    `;
                }} else {{
                    display.innerHTML = `
                        <h3>📺 Ready</h3>
                        <p class="current-topic-title">Click any topic to jump</p>
                        <p>⚡ Use keyboard shortcuts</p>
                    `;
                }}
            }}
            
            updatePlayButton() {{
                this.playBtn.innerHTML = this.isPlaying ? '⏸️' : '▶️';
            }}
            
            jumpToTopic(topicIndex) {{
                if (topicIndex >= 0 && topicIndex < this.topics.length) {{
                    const topic = this.topics[topicIndex];
                    this.video.currentTime = topic.start_seconds;
                    
                    if (this.video.paused) {{
                        this.video.play().catch(e => {{
                            console.log('Could not auto-play video:', e);
                        }});
                    }}
                    
                    console.log(`🎯 Jumped to: ${{topic.name}} at ${{topic.start_time}}`);
                }}
            }}
            
            togglePlayPause() {{
                if (this.video.paused) {{
                    this.video.play().catch(e => {{
                        console.log('Could not play video:', e);
                    }});
                }} else {{
                    this.video.pause();
                }}
            }}
            
            seekToPosition(e) {{
                const rect = this.progressContainer.getBoundingClientRect();
                const pos = (e.clientX - rect.left) / rect.width;
                const newTime = pos * this.getTotalDuration();
                this.video.currentTime = Math.max(0, Math.min(newTime, this.getTotalDuration()));
            }}
            
            handleKeyboard(e) {{
                if (e.target.tagName.toLowerCase() === 'input') return;
                
                switch(e.key) {{
                    case ' ':
                        e.preventDefault();
                        this.togglePlayPause();
                        break;
                    case 'ArrowRight':
                        e.preventDefault();
                        this.video.currentTime += 10;
                        break;
                    case 'ArrowLeft':
                        e.preventDefault();
                        this.video.currentTime -= 10;
                        break;
                    case 'ArrowUp':
                        e.preventDefault();
                        this.jumpToPreviousTopic();
                        break;
                    case 'ArrowDown':
                        e.preventDefault();
                        this.jumpToNextTopic();
                        break;
                    case 'f':
                    case 'F':
                        e.preventDefault();
                        this.toggleFullscreen();
                        break;
                    default:
                        if (e.key >= '0' && e.key <= '9') {{
                            e.preventDefault();
                            const percent = parseInt(e.key) * 10;
                            this.video.currentTime = (this.getTotalDuration() * percent) / 100;
                        }}
                        break;
                }}
            }}
            
            jumpToPreviousTopic() {{
                if (this.currentTopicIndex > 0) {{
                    this.jumpToTopic(this.currentTopicIndex - 1);
                }} else if (this.topics.length > 0) {{
                    this.jumpToTopic(0);
                }}
            }}
            
            jumpToNextTopic() {{
                if (this.currentTopicIndex < this.topics.length - 1) {{
                    this.jumpToTopic(this.currentTopicIndex + 1);
                }} else if (this.topics.length > 0) {{
                    this.jumpToTopic(this.topics.length - 1);
                }}
            }}
            
            toggleFullscreen() {{
                if (!document.fullscreenElement) {{
                    this.video.requestFullscreen().catch(err => {{
                        console.log('Error entering fullscreen:', err);
                    }});
                }} else {{
                    document.exitFullscreen();
                }}
            }}
            
            formatTime(seconds) {{
                if (!seconds || isNaN(seconds)) return '00:00';
                
                const hrs = Math.floor(seconds / 3600);
                const mins = Math.floor((seconds % 3600) / 60);
                const secs = Math.floor(seconds % 60);
                
                if (hrs > 0) {{
                    return `${{hrs}}:${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
                }}
                return `${{mins.toString().padStart(2, '0')}}:${{secs.toString().padStart(2, '0')}}`;
            }}
            
            updateDisplay() {{
                this.updateTimeDisplay();
                this.updatePlayButton();
                this.updateCurrentTopicDisplay();
            }}
        }}
        
        document.addEventListener('DOMContentLoaded', () => {{
            new CompactSideBySideVideoPlayer();
        }});
        """
    
    def generate_html(self, video_title="Lecture Video"):
        """Generate HTML with COMPACT side-by-side layout"""
        
        # Generate topics HTML
        topics_html = ""
        for i, topic in enumerate(self.topics):
            duration_str = self.seconds_to_timestamp(topic['duration'])
            
            topics_html += f"""
            <div class="topic-item" data-topic-index="{i}" tabindex="0">
                <div class="topic-content">
                    <div class="topic-header">
                        <span class="topic-number">{i+1:02d}</span>
                        <div class="topic-name">{topic['name']}</div>
                    </div>
                    <div class="topic-time-info">
                        <span>{topic['start_time']} - {topic['end_time']}</span>
                        <span class="topic-duration">{duration_str}</span>
                    </div>
                </div>
            </div>
            """
        
        # Calculate total duration
        total_duration = max(topic['end_seconds'] for topic in self.topics) if self.topics else 0
        duration_display = self.seconds_to_timestamp(total_duration)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{video_title} - Compact Player</title>
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;600;700&family=Roboto+Mono:wght@400;500&display=swap" rel="stylesheet">
            <style>
                {self.generate_css()}
            </style>
        </head>
        <body>
            <div class="main-container">
                <!-- Left Side: Video Section - MORE SPACE -->
                <div class="video-section">
                    <div class="video-container">
                        <div class="video-wrapper">
                            <video id="mainVideo" preload="metadata" controls>
                                <source src="{self.video_path}" type="video/mp4">
                                <source src="{self.video_path}" type="video/webm">
                                Your browser does not support the video tag.
                            </video>
                        </div>
                        
                        <div class="custom-controls">
                            <div class="progress-container">
                                <div class="timeline-segments"></div>
                                <div class="progress-bar">
                                    <div class="progress-handle"></div>
                                </div>
                            </div>
                            
                            <div class="controls-row">
                                <button class="play-btn" id="playBtn" title="Play/Pause (Space)">▶️</button>
                                <div class="time-display" id="timeDisplay">00:00 / 00:00</div>
                                
                                <div class="volume-control">
                                    <button class="volume-btn" title="Mute/Unmute">🔊</button>
                                </div>
                                
                                <button class="fullscreen-btn" title="Fullscreen (F)" onclick="document.getElementById('mainVideo').requestFullscreen()">⛶</button>
                            </div>
                        </div>
                    </div>
                    
                    <!-- COMPACT Video Info -->
                    <div class="video-info">
                        <h1 class="video-title">{video_title}</h1>
                        
                        <div class="video-meta">
                            <div class="meta-item">
                                <span>📚</span>
                                <span>{len(self.topics)} Topics</span>
                            </div>
                            <div class="meta-item">
                                <span>⏱️</span>
                                <span>{duration_display}</span>
                            </div>
                            <div class="meta-item">
                                <span>📅</span>
                                <span>{datetime.now().strftime('%b %d, %Y')}</span>
                            </div>
                            <div class="meta-item">
                                <span>🎯</span>
                                <span>Interactive</span>
                            </div>
                        </div>
                        
                        <!-- COMPACT Currently Playing -->
                        <div class="current-topic-display" id="currentTopicDisplay">
                            <h3>📺 Ready</h3>
                            <p class="current-topic-title">Click any topic to jump</p>
                            <p>⚡ Use keyboard shortcuts</p>
                        </div>
                    </div>
                </div>
                
                <!-- Right Side: SMALLER Table of Contents -->
                <div class="toc-section">
                    <div class="toc-header">
                        <h2 class="toc-title">
                            <span>📋</span>
                            <span>Contents</span>
                        </h2>
                        <p class="toc-subtitle">Click to jump to section</p>
                    </div>
                    
                    <div class="topics-container">
                        {topics_html}
                    </div>
                    
                    <div class="shortcuts-panel">
                        <div class="shortcuts-title">⌨️ Shortcuts</div>
                        <div class="shortcuts-grid">
                            <div class="shortcut-item">
                                <span>Play/Pause</span>
                                <span class="shortcut-key">Space</span>
                            </div>
                            <div class="shortcut-item">
                                <span>±10s</span>
                                <span class="shortcut-key">← →</span>
                            </div>
                            <div class="shortcut-item">
                                <span>Prev/Next</span>
                                <span class="shortcut-key">↑ ↓</span>
                            </div>
                            <div class="shortcut-item">
                                <span>Fullscreen</span>
                                <span class="shortcut-key">F</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                {self.generate_javascript()}
            </script>
        </body>
        </html>
        """
        
        return html_content
    
    def create_timestamped_player(self, video_path, topic_file_path, output_name="compact_sidebyside_player", video_title=None):
        """Create the COMPACT side-by-side timestamped video player"""
        print(f"🎬 CREATING COMPACT SIDE-BY-SIDE TIMESTAMPED VIDEO PLAYER")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*60)
        
        if not self.parse_topic_file(topic_file_path):
            return None
        
        copied_video = self.copy_video_file(video_path)
        if not copied_video:
            return None
        
        if not video_title:
            video_title = f"Lecture Video - {len(self.topics)} Topics"
        
        print(f"🌐 Generating COMPACT side-by-side HTML player...")
        html_content = self.generate_html(video_title)
        
        html_filename = f"{output_name}.html"
        html_path = os.path.join(self.output_dir, html_filename)
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        info_data = {
            'video_file': self.video_path,
            'topic_file': topic_file_path,
            'total_topics': len(self.topics),
            'total_duration_seconds': max(topic['end_seconds'] for topic in self.topics),
            'generated_date': datetime.now().isoformat(),
            'topics': self.topics,
            'layout': 'compact-side-by-side',
            'features': [
                'COMPACT side-by-side layout',
                'TOC width reduced to 320px (from 450px)',
                'Currently Playing section height reduced',
                'More space for video player',
                'Smaller fonts and padding',
                'All timeline functionality preserved',
                'Responsive design',
                'Keyboard shortcuts'
            ]
        }
        
        info_path = os.path.join(self.output_dir, f"{output_name}_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info_data, f, indent=2)
        
        print(f"✅ COMPACT SIDE-BY-SIDE VIDEO PLAYER CREATED!")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🌐 HTML file: {html_path}")
        print(f"📋 Info file: {info_path}")
        print(f"📹 Video file: {copied_video}")
        print(f"📊 Topics processed: {len(self.topics)}")
        
        print(f"\n🎯 COMPACT LAYOUT IMPROVEMENTS:")
        print(f"   ✅ TOC width: 320px (was 450px) - 130px saved")
        print(f"   ✅ Currently Playing: Reduced height & padding")
        print(f"   ✅ Video meta: Smaller badges and fonts")
        print(f"   ✅ Topic items: Compact padding and fonts")
        print(f"   ✅ Shortcuts panel: Smaller grid")
        print(f"   ✅ More space for video player")
        
        print(f"\n📚 TOPICS OVERVIEW:")
        for i, topic in enumerate(self.topics, 1):
            print(f"  {i:2d}. {topic['name']} ({topic['start_time']} - {topic['end_time']})")
        
        return {
            'html_path': html_path,
            'video_path': copied_video,
            'info_path': info_path,
            'output_dir': self.output_dir,
            'topics_count': len(self.topics)
        }


def main():
    """Main function"""
    print("🎬 COMPACT SIDE-BY-SIDE TIMESTAMPED VIDEO PLAYER GENERATOR")
    print("📦 Creates compact layout with more space for video")
    print("="*60)
    
    video_path = input("Enter video file path (default: output_cleaned_vad_mod.mp4): ").strip()
    if not video_path:
        video_path = "output_cleaned_vad_mod.mp4"
    
    topic_file = input("Enter topic segmentation file path (default: topic_segmentation_*.txt): ").strip()
    if not topic_file:
        import glob
        topic_files = glob.glob("topic_segmentation_*.txt")
        if topic_files:
            topic_file = topic_files[0]
            print(f"📄 Found topic file: {topic_file}")
        else:
            print("❌ No topic segmentation file found")
            return
    
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        return
    
    if not os.path.exists(topic_file):
        print(f"❌ Topic file not found: {topic_file}")
        return
    
    video_title = input("Enter video title (optional): ").strip()
    if not video_title:
        video_title = "Lecture Video with Interactive Timeline"
    
    output_name = input("Enter output name (default: compact_player): ").strip()
    if not output_name:
        output_name = "compact_player"
    
    print(f"\n🚀 Creating COMPACT side-by-side video player...")
    print(f"📹 Video: {video_path}")
    print(f"📄 Topics: {topic_file}")
    print(f"🎯 Title: {video_title}")
    print(f"📁 Output: {output_name}")
    print("-" * 60)
    
    player_generator = CompactSideBySideTimestampedVideoPlayer()
    result = player_generator.create_timestamped_player(
        video_path=video_path,
        topic_file_path=topic_file,
        output_name=output_name,
        video_title=video_title
    )
    
    if result:
        print(f"\n✨ SUCCESS!")
        print(f"🌐 Open this file in your browser:")
        print(f"   {os.path.abspath(result['html_path'])}")
        print(f"\n💡 COMPACT IMPROVEMENTS:")
        print(f"   📦 Table of Contents: 320px width (was 450px)")
        print(f"   📏 Currently Playing: Reduced height and padding")
        print(f"   🎥 Video Player: More space available")
        print(f"   📝 Compact fonts and spacing throughout")
        print(f"   ⚡ All timeline functionality preserved")
    else:
        print(f"\n❌ Failed to create compact side-by-side video player")


if __name__ == "__main__":
    main()