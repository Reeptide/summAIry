import requests
import json
import time
from datetime import datetime
import threading
import os
import re
import pandas as pd
from typing import List, Dict, Tuple, Optional
import math

class RAGOptimizedTranscriptAnalyzer:
    def __init__(self, model="llama3.1:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url  
        self.finished = False
        self.detected_timestamp_format = None
        
        # Connection settings
        self.max_retries = 3
        self.base_timeout = 1800
        self.connection_timeout = 60

    def detect_timestamp_format(self, text: str) -> str:
        hms_pattern = r'\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]'
        hms_matches = re.findall(hms_pattern, text)
        hm_pattern = r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]'
        hm_matches = re.findall(hm_pattern, text)
        if hms_matches:
            detected_format = 'H:MM:SS'
            sample_timestamps = hms_matches[:3]
        elif hm_matches:
            detected_format = 'H:MM'
            sample_timestamps = hm_matches[:3]
        else:
            detected_format = 'H:MM:SS'
            sample_timestamps = []
        print(f"🕒 Detected timestamp format: {detected_format}")
        if sample_timestamps:
            print(f"📝 Sample timestamps: {sample_timestamps}")
        self.detected_timestamp_format = detected_format
        return detected_format
        
    def extract_timestamps_from_text(self, text: str) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        if not self.detected_timestamp_format:
            self.detect_timestamp_format(text)
        if self.detected_timestamp_format == 'H:MM:SS':
            timestamp_pattern = r'\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]'
            timestamp_content_pattern = r'\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]\s*([^[]*?)(?=\[|$)'
        else:
            timestamp_pattern = r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]'
            timestamp_content_pattern = r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]\s*([^[]*?)(?=\[|$)'
        matches = re.findall(timestamp_pattern, text)
        content_matches = re.findall(timestamp_content_pattern, text, re.DOTALL)
        return matches, content_matches
    
    def get_timestamp_regex_pattern(self) -> str:
        if self.detected_timestamp_format == 'H:MM:SS':
            return r'(\[\d{1,2}:\d{2}:\d{2}\s*-\s*\d{1,2}:\d{2}:\d{2}\])'
        else:
            return r'(\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\])'
    
    def get_timestamp_match_pattern(self) -> str:
        if self.detected_timestamp_format == 'H:MM:SS':
            return r'\[\d{1,2}:\d{2}:\d{2}\s*-\s*\d{1,2}:\d{2}:\d{2}\]'
        else:
            return r'\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]'
    
    def get_format_example(self) -> str:
        if self.detected_timestamp_format == 'H:MM:SS':
            return "topic [0:00:00-0:02:30]"
        else:
            return "topic [0:00-2:30]"
    
    def get_format_description(self) -> str:
        if self.detected_timestamp_format == 'H:MM:SS':
            return "H:MM:SS format"
        else:
            return "H:MM format"
    
    def preprocess_transcript(self, text: str) -> str:
        if not self.detected_timestamp_format:
            self.detect_timestamp_format(text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        if self.detected_timestamp_format == 'H:MM:SS':
            text = re.sub(r'\[(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]', r'[\1 - \2]', text)
        else:
            text = re.sub(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', r'[\1 - \2]', text)
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        return '\n'.join(cleaned_lines)
    
    def calculate_optimal_processing_strategy(self, text: str) -> Dict:
        word_count = len(text.split())
        char_count = len(text)
        timestamps, _ = self.extract_timestamps_from_text(text)
        model_limits_tokens = {
            'llama3.1': 128000,
            'llama3': 128000,
            'llama3.3': 128000,
            'gemma2': 8192,
            'qwen2': 32768,
            'default': 8192
        }
        model_base = self.model.split(':')[0]
        max_tokens = model_limits_tokens.get(model_base, model_limits_tokens['default'])
        usable_token_ratio = 0.95
        max_words = int(max_tokens * usable_token_ratio * 0.75)
        needs_chunking = word_count > max_words
        print(f"📊 MAXIMIZED PROCESSING STRATEGY:")
        print(f"   🔧 Model: {self.model}")
        print(f"   🎯 Base model: {model_base}")
        print(f"   📏 Token limit: {max_tokens:,} tokens")
        print(f"   🚀 Context usage: {usable_token_ratio*100:.0f}% (MAXIMIZED)")
        print(f"   📝 Max usable words: {max_words:,} words")
        print(f"   📄 Transcript words: {word_count:,} words")
        print(f"   🔀 Needs chunking: {needs_chunking}")
        if needs_chunking:
            ratio = word_count / max_words
            print(f"   📈 Size ratio: {ratio:.2f}x over limit")
            print(f"   💡 Consider using llama3.1:70b for larger transcripts")
        else:
            efficiency = (word_count / max_words) * 100
            print(f"   ✅ Efficiency: {efficiency:.1f}% of MAXIMIZED capacity")
        strategy = {
            'word_count': word_count,
            'char_count': char_count,
            'timestamp_segments': len(timestamps),
            'timestamp_format': self.detected_timestamp_format,
            'max_tokens': max_tokens,
            'max_model_words': max_words,
            'needs_chunking': needs_chunking,
            'estimated_processing_time': self._estimate_processing_time(word_count),
            'recommended_timeout': self._calculate_timeout(word_count)
        }
        if needs_chunking:
            strategy['chunk_size'] = int(max_words * 0.9)
            strategy['num_chunks'] = math.ceil(word_count / strategy['chunk_size'])
            print(f"   📦 Chunk size: {strategy['chunk_size']:,} words")
            print(f"   🔢 Number of chunks: {strategy['num_chunks']}")
        else:
            strategy['chunk_size'] = word_count
            strategy['num_chunks'] = 1
        return strategy
    
    def _estimate_processing_time(self, word_count: int) -> int:
        base_time = word_count / 100
        return int(base_time * 2) + 60
    
    def _calculate_timeout(self, word_count: int) -> int:
        estimated_time = self._estimate_processing_time(word_count)
        return max(self.base_timeout, int(estimated_time * 1.5))
    
    def fixed_chunking_strategy(self, text: str, target_chunk_size: int) -> List[str]:
        print(f"\n🔧 APPLYING SMART CHUNKING STRATEGY")
        if not self.detected_timestamp_format:
            self.detect_timestamp_format(text)
        processed_text = self.preprocess_transcript(text)
        words = processed_text.split()
        total_words = len(words)
        print(f"📊 Chunking statistics:")
        print(f"   • Total words: {total_words:,}")
        print(f"   • Target chunk size: {target_chunk_size:,} words")
        print(f"   • Timestamp format: {self.detected_timestamp_format}")
        if total_words <= target_chunk_size:
            print(f"✅ Text fits in single chunk")
            return [processed_text]
        num_chunks = math.ceil(total_words / target_chunk_size)
        print(f"📦 Will create {num_chunks} chunks")
        timestamp_pattern = self.get_timestamp_regex_pattern()
        timestamp_segments = re.split(timestamp_pattern, processed_text)
        segments = []
        match_pattern = self.get_timestamp_match_pattern()
        i = 0
        while i < len(timestamp_segments):
            if i + 1 < len(timestamp_segments) and re.match(match_pattern, timestamp_segments[i]):
                segment_text = timestamp_segments[i] + (timestamp_segments[i + 1] if i + 1 < len(timestamp_segments) else "")
                segments.append({
                    'text': segment_text,
                    'word_count': len(segment_text.split()),
                    'has_timestamp': True
                })
                i += 2
            else:
                if timestamp_segments[i].strip():
                    segments.append({
                        'text': timestamp_segments[i],
                        'word_count': len(timestamp_segments[i].split()),
                        'has_timestamp': False
                    })
                i += 1
        print(f"🎯 Found {len(segments)} timestamp segments")
        chunks = []
        current_chunk_segments = []
        current_word_count = 0
        for segment in segments:
            segment_words = segment['word_count']
            if current_word_count + segment_words > target_chunk_size and current_chunk_segments:
                chunk_text = ' '.join([s['text'] for s in current_chunk_segments])
                chunks.append(chunk_text.strip())
                current_chunk_segments = [segment]
                current_word_count = segment_words
            else:
                current_chunk_segments.append(segment)
                current_word_count += segment_words
        if current_chunk_segments:
            chunk_text = ' '.join([s['text'] for s in current_chunk_segments])
            chunks.append(chunk_text.strip())
        chunk_sizes = [len(chunk.split()) for chunk in chunks]
        print(f"✅ Smart chunking completed:")
        print(f"   • Created {len(chunks)} chunks")
        print(f"   • Chunk sizes: {min(chunk_sizes)}-{max(chunk_sizes)} words")
        print(f"   • Average size: {sum(chunk_sizes)/len(chunk_sizes):.0f} words")
        return chunks

    def create_rag_optimized_prompt(self, full_text: str) -> str:
        if not self.detected_timestamp_format:
            self.detect_timestamp_format(full_text)
        timestamps, _ = self.extract_timestamps_from_text(full_text)
        prompt = f"""TASK: Segment this lecture transcript into COARSE, HIGH-LEVEL topics for educational retrieval and video navigation.

    CRITICAL REQUIREMENTS:
    - **MINIMUM DURATION: Every topic MUST be at least 7 minutes long**
    - **MAXIMUM DURATION: Every topic MUST be at most 15 minutes long**
    - **IF A TOPIC WOULD BE SHORTER THAN 7 MINUTES, YOU MUST MERGE IT WITH THE NEXT TOPIC**
    - **Use 'and' to join merged topic names: "Topic A and Topic B"**
    - **Merged timestamps must cover the full range of both topics**

    INSTRUCTIONS:
    - PRODUCE A COMPLETE, GAP-FREE, NON-OVERLAPPING LIST OF TOPICS
    - EACH TOPIC MUST BE 7-15 MINUTES IN DURATION
    - ONLY CREATE A NEW TOPIC WHEN THE INSTRUCTOR SHIFTS TO A FUNDAMENTALLY DIFFERENT MAIN IDEA
    - COMBINE ALL RELATED DETAILS, EXAMPLES, SUBTOPICS UNDER THEIR PARENT TOPIC
    - USE DESCRIPTIVE TOPIC NAMES BASED ON INSTRUCTOR TERMINOLOGY
    - ENSURE THE ENTIRE LECTURE IS COVERED FROM START TO END
    - OUTPUT FORMAT: topic name [starttime-endtime]
    - NO EXPLANATIONS, BULLET POINTS, OR EXTRA FORMATTING

    REMEMBER: NO TOPIC CAN BE SHORTER THAN 7 MINUTES. MERGE IF NECESSARY.

    TRANSCRIPT:
    {full_text}

    OUTPUT (each topic 7-15 minutes):
    """
        return prompt

    def merge_short_topics(self, topics, min_duration=7):
        if not topics:
            return topics
        
        merged = []
        i = 0
        
        while i < len(topics):
            current_topic = topics[i]
            topic_names = [current_topic['name']]
            start_time = current_topic['start_time']
            end_time = current_topic['end_time']
            total_duration = current_topic['duration_minutes']
            
            # Keep merging until we reach minimum duration or run out of topics
            j = i + 1
            while total_duration < min_duration and j < len(topics):
                next_topic = topics[j]
                topic_names.append(next_topic['name'])
                end_time = next_topic['end_time']
                total_duration += next_topic['duration_minutes']
                j += 1
            
            # Create merged topic
            merged_name = ' and '.join(topic_names)
            merged.append({
                'name': merged_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration_minutes': total_duration,
                'balanced': 7 <= total_duration <= 15,
                'too_short': total_duration < 7,
                'too_long': total_duration > 15
            })
            
            # Move to next unprocessed topic
            i = j
        
        return merged
    
    def debug_topic_durations(self, topics):
        print(f"\n🔍 DEBUG: Topic Duration Analysis")
        print(f"=" * 50)
        
        for i, topic in enumerate(topics):
            duration = topic.get('duration_minutes', 0)
            status = "✅ GOOD" if 7 <= duration <= 15 else "❌ BAD"
            print(f"{i+1:2d}. {topic['name'][:50]:<50} | {duration:5.1f}min | {status}")
        
        total_topics = len(topics)
        balanced = sum(1 for t in topics if 7 <= t.get('duration_minutes', 0) <= 15)
        too_short = sum(1 for t in topics if t.get('duration_minutes', 0) < 7)
        too_long = sum(1 for t in topics if t.get('duration_minutes', 0) > 15)
        
        print(f"\n📊 Summary:")
        print(f"   Total topics: {total_topics}")
        print(f"   Balanced (7-15min): {balanced}")
        print(f"   Too short (<7min): {too_short}")
        print(f"   Too long (>15min): {too_long}")
        print(f"=" * 50)

    def make_single_llm_call(self, full_text: str, timeout: int = None) -> str:
        print(f"\n🚀 MAKING RAG-OPTIMIZED LLM CALL (MAXIMIZED CONTEXT)")
        if not self.detected_timestamp_format:
            self.detect_timestamp_format(full_text)
        strategy = self.calculate_optimal_processing_strategy(full_text)
        if strategy['needs_chunking']:
            print(f"⚠️ Transcript exceeds MAXIMIZED capacity, using chunking")
            return self._handle_large_transcript_call(full_text, strategy)
        else:
            print(f"✅ Transcript fits in MAXIMIZED single call")
            return self._make_true_single_call(full_text, strategy, timeout)
    
    def _make_true_single_call(self, full_text: str, strategy: Dict, timeout: int = None) -> str:
        prompt = self.create_rag_optimized_prompt(full_text)
        if timeout is None:
            timeout = strategy['recommended_timeout']
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.8,
                "num_predict": 8000,
                "repeat_penalty": 1.1,
                "top_k": 30,
                "num_ctx": 128000
            }
        }
        for attempt in range(self.max_retries):
            try:
                print(f"🔄 Attempt {attempt + 1}/{self.max_retries}")
                print(f"⏱️ Timeout: {timeout} seconds")
                print(f"🚀 Context window: {payload['options']['num_ctx']:,} tokens (MAXIMIZED)")
                print(f"🎯 RAG-optimized settings: temp=0.2, top_p=0.8")
                self.finished = False
                progress_thread = threading.Thread(target=self._show_single_call_progress)
                progress_thread.start()
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=timeout, headers={'Connection': 'keep-alive'})
                self.finished = True
                progress_thread.join()
                processing_time = time.time() - start_time
                print(f"\n✅ RAG-optimized LLM call completed in {processing_time:.1f} seconds")
                if response.status_code == 200:
                    result = response.json().get("response", "")
                    if self._validate_rag_topic_output(result):
                        print(f"✅ Valid RAG-focused topic extraction received")
                        return result.strip()
                    else:
                        print(f"⚠️ Invalid RAG format, retrying...")
                        if attempt < self.max_retries - 1:
                            continue
                        else:
                            print(f"❌ All attempts failed, returning raw result")
                            return result.strip()
                else:
                    print(f"❌ HTTP Error {response.status_code}: {response.text}")
                    if attempt < self.max_retries - 1:
                        wait_time = 2 ** attempt
                        print(f"⏳ Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        timeout = min(timeout * 1.2, 3600)
            except requests.exceptions.Timeout:
                print(f"⏰ Request timed out after {timeout} seconds")
                if attempt < self.max_retries - 1:
                    timeout = min(timeout * 1.5, 3600)
                    print(f"🔄 Increasing timeout to {timeout} seconds for retry")
            except requests.exceptions.ConnectionError:
                print(f"🔌 Connection error to Ollama server")
                if attempt < self.max_retries - 1:
                    print(f"⏳ Waiting 5 seconds before retry...")
                    time.sleep(5)
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                if attempt < self.max_retries - 1:
                    print(f"⏳ Waiting 3 seconds before retry...")
                    time.sleep(3)
            finally:
                self.finished = True
        print(f"❌ All retry attempts failed")
        return "Error: Failed to process transcript after all retry attempts"
    
    def _handle_large_transcript_call(self, full_text: str, strategy: Dict) -> str:
        print(f"🔧 Processing large transcript with {strategy['num_chunks']} chunks for RAG")
        chunks = self.fixed_chunking_strategy(full_text, strategy['chunk_size'])
        preliminary_topics = []
        format_example = self.get_format_example()
        for i, chunk in enumerate(chunks, 1):
            print(f"🔍 Processing chunk {i}/{len(chunks)} for RAG topics")
            rag_chunk_prompt = f"""Extract educational topics (7+ minutes each). Format: {format_example}

Focus on: Main concepts, methods, algorithms, procedures, comparisons.
Use instructor's exact terminology. Avoid short topics.

{chunk}

RAG Topics:
"""
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model,
                "prompt": rag_chunk_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "num_predict": 1000,
                    "num_ctx": 131072
                }
            }
            try:
                response = requests.post(url, json=payload, timeout=1200)
                if response.status_code == 200:
                    chunk_topics = response.json().get("response", "").strip()
                    if chunk_topics:
                        preliminary_topics.append(chunk_topics)
            except Exception as e:
                print(f"⚠️ Error processing chunk {i}: {e}")
                continue
        combined_topics = "\n\n".join(preliminary_topics)
        final_rag_prompt = f"""Consolidate into educational RAG topics. Format: {format_example}

Requirements:
- 7-15 minutes per topic minimum
- Educational/conceptual focus
- Instructor's terminology  
- Clear boundaries for retrieval

{combined_topics}

Final RAG Topics:
"""
        payload = {
            "model": self.model,
            "prompt": final_rag_prompt,
            "stream": False,
            "options": {
                "temperature": 0.05,
                "num_predict": 3000,
                "num_ctx": 131072
            }
        }
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=600)
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"❌ Final RAG consolidation failed: {e}")
        return combined_topics
    
    def _validate_rag_topic_output(self, output: str) -> bool:
        if not output or not output.strip():
            return False
        lines = output.strip().split('\n')
        valid_lines = 0
        educational_keywords = ['introduction', 'algorithm', 'method', 'procedure', 'vs', 'comparison', 'testing', 'analysis', 'concept', 'theory', 'practice']
        for line in lines:
            line = line.strip().lower()
            if not line or line.startswith('rag topics') or line.startswith('final') or line.startswith('educational'):
                continue
            has_timestamp = False
            if self.detected_timestamp_format == 'H:MM:SS':
                timestamp_pattern = r'\[?\d{1,2}:\d{2}:\d{2}[^\]]*\d{1,2}:\d{2}:\d{2}\]?'
            else:
                timestamp_pattern = r'\[?\d{1,2}:\d{2}[^\]]*\d{1,2}:\d{2}\]?'
            if re.search(timestamp_pattern, line):
                has_timestamp = True
            has_educational_focus = any(keyword in line for keyword in educational_keywords)
            if has_timestamp and (has_educational_focus or len(line.split()) > 3):
                valid_lines += 1
        non_empty_lines = len([line for line in lines if line.strip() and not any(starter in line.lower() for starter in ['rag topics', 'final', 'educational'])])
        return valid_lines >= max(1, non_empty_lines * 0.4)
    
    def _show_single_call_progress(self):
        dots = 0
        start_time = time.time()
        while not self.finished:
            elapsed = int(time.time() - start_time)
            minutes, seconds = divmod(elapsed, 60)
            print(f"\r🤔 Processing RAG topics - {minutes:02d}:{seconds:02d} elapsed{'.' * (dots % 4)}{' ' * (3 - (dots % 4))}", 
                  end='', flush=True)
            dots += 1
            time.sleep(2)
    
    def parse_rag_topic_list(self, topic_text: str) -> List[Dict]:
        topics = []
        lines = topic_text.strip().split('\n')
        print(f"🔍 Parsing {len(lines)} lines for balanced topics...")
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or any(starter in line.lower() for starter in ['rag topics', 'final', 'balanced', 'task:', 'format:']):
                continue
            clean_line = re.sub(r'^\*\*?|\*\*?$', '', line)
            clean_line = re.sub(r'^\*\s*', '', clean_line)
            clean_line = re.sub(r'^\d+\.\s*', '', clean_line)
            clean_line = clean_line.strip()
            timestamp_match = re.search(r'(.+?)\s*\[?(\d{1,2}:\d{2}:\d{2})\s*-\s*(\d{1,2}:\d{2}:\d{2})\]?', clean_line)
            if timestamp_match:
                topic_name = timestamp_match.group(1).strip()
                start_time = timestamp_match.group(2)
                end_time = timestamp_match.group(3)
                start_parts = start_time.split(':')
                end_parts = end_time.split(':')
                start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
                end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
                duration_minutes = end_minutes - start_minutes
                topics.append({
                    'name': topic_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_minutes': duration_minutes,
                    'full_text': line,
                    'format': 'H:MM:SS',
                    'balanced': 7 <= duration_minutes <= 15,
                    'too_short': duration_minutes < 7,
                    'too_long': duration_minutes > 15
                })
                continue
            timestamp_match = re.search(r'(.+?)\s*\[?(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]?', clean_line)
            if timestamp_match:
                topic_name = timestamp_match.group(1).strip()
                start_time = timestamp_match.group(2)
                end_time = timestamp_match.group(3)
                start_parts = start_time.split(':')
                end_parts = end_time.split(':')
                start_minutes = int(start_parts[0]) * 60 + int(start_parts[1])
                end_minutes = int(end_parts[0]) * 60 + int(end_parts[1])
                duration_minutes = end_minutes - start_minutes
                topics.append({
                    'name': topic_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration_minutes': duration_minutes,
                    'full_text': line,
                    'format': 'H:MM',
                    'balanced': 7 <= duration_minutes <= 15,
                    'too_short': duration_minutes < 7,
                    'too_long': duration_minutes > 15
                })
                continue
        balanced_count = sum(1 for topic in topics if topic.get('balanced', False))
        too_short_count = sum(1 for topic in topics if topic.get('too_short', False))
        too_long_count = sum(1 for topic in topics if topic.get('too_long', False))
        avg_duration = sum(topic.get('duration_minutes', 0) for topic in topics) / len(topics) if topics else 0
        print(f"✅ Successfully parsed {len(topics)} topics")
        print(f"🎯 Balanced topics (7-15 min): {balanced_count}/{len(topics)} ({(balanced_count/len(topics)*100 if topics else 0):.1f}%)")
        print(f"⚠️ Too short topics (<7 min): {too_short_count}")
        print(f"⚠️ Too long topics (>15 min): {too_long_count}")
        print(f"⏱️ Average topic duration: {avg_duration:.1f} minutes")
        return topics

    def format_topics_for_gui(self, topics_list):
        """Format topics for GUI compatibility - ensuring clean output"""
        formatted_lines = []
        
        for topic in topics_list:
            # Extract topic data
            name = topic.get('name', 'Unknown Topic')
            start_time = topic.get('start_time', '0:00')
            end_time = topic.get('end_time', '0:00')
            
            # Clean topic name from any unwanted characters
            name = re.sub(r'^\*\*?|\*\*?$', '', name)  # Remove markdown
            name = re.sub(r'^\d+\.\s*', '', name)  # Remove numbering
            name = name.strip()
            
            # Ensure consistent format: "Topic Name [MM:SS-MM:SS]"
            formatted_line = f"{name} [{start_time}-{end_time}]"
            formatted_lines.append(formatted_line)
        
        return "\n".join(formatted_lines)

    def save_rag_results(self, results: Dict, transcript_filename: str) -> str:
        """Save RAG results in GUI-compatible format"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(transcript_filename))[0]
        topic_filename = f'balanced_topics_{base_name}_{timestamp}.txt'
        
        # Format topics for GUI compatibility
        if 'predicted_topics' in results and results['predicted_topics']:
            gui_formatted_topics = self.format_topics_for_gui(results['predicted_topics'])
            
            with open(topic_filename, 'w', encoding='utf-8') as f:
                f.write("# RAG-Generated Balanced Topics for Video Navigation\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# User: niranjani1401\n")
                f.write(f"# Total Topics: {len(results['predicted_topics'])}\n")
                f.write(f"# Format: Topic Name [MM:SS-MM:SS]\n")
                f.write(f"# Pipeline: Video → Speech → Transcript → RAG → GUI\n\n")
                f.write(gui_formatted_topics)
        else:
            # Fallback to original format
            with open(topic_filename, 'w', encoding='utf-8') as f:
                f.write("# RAG-Generated Topics\n")
                f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"# User: niranjani1401\n\n")
                f.write(results.get('final_topic_list', ''))
        
        print(f"\n💾 GUI-compatible topic results saved to: {topic_filename}")
        return topic_filename

    def test_connection(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                print(f"✅ Connected to Ollama!")
                print(f"📋 Available models: {', '.join(available_models)}")
                if self.model in available_models:
                    print(f"🎯 Using model: {self.model}")
                    return True
                else:
                    print(f"⚠️ Model '{self.model}' not found. Available: {available_models}")
                    return False
            else:
                print(f"❌ Ollama error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            return False

    def analyze_transcript_for_rag(self, transcript_file_path: str) -> Optional[Dict]:
        """Main analysis method with updated user info"""
        print(f"🚀 BALANCED TOPIC ANALYZER")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 User: niranjani1401")
        print(f"🎯 Focus: Balanced 7-15 minute topics for RAG system and video player")
        print("="*60)
        
        if not self.test_connection():
            return None
        try:
            with open(transcript_file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            print(f"📄 Transcript loaded:")
            print(f"   • Characters: {len(transcript):,}")
            print(f"   • Words: {len(transcript.split()):,}")
            timestamp_format = self.detect_timestamp_format(transcript)
            timestamps, _ = self.extract_timestamps_from_text(transcript)
            print(f"   • Timestamp ranges: {len(timestamps)}")
            print(f"   • Format detected: {timestamp_format}")
        except Exception as e:
            print(f"❌ Error loading transcript: {e}")
            return None
        
        print(f"\n🎯 STARTING BALANCED TOPIC ANALYSIS...")
        print(f"🔧 Target: Topics of 7-15 minutes each for optimal RAG and video navigation")
        start_time = time.time()
        final_topic_list = self.make_single_llm_call(transcript)
        
        if "Error:" in final_topic_list:
            print(f"❌ Topic analysis failed: {final_topic_list}")
            return None
        
        predicted_topics = self.parse_rag_topic_list(final_topic_list)
        self.debug_topic_durations(predicted_topics)
        predicted_topics = self.merge_short_topics(predicted_topics, min_duration=7)
        self.debug_topic_durations(predicted_topics)
        
        balanced = sum(1 for topic in predicted_topics if topic.get('balanced', False))
        if balanced < len(predicted_topics):
            print(f"⚠️ Only {balanced}/{len(predicted_topics)} topics are in optimal range")
            print(f"🔄 Adjusting topics to ensure 7-15 minute range...")
            adjusted_topics = self.validate_and_adjust_topics(predicted_topics)
            if len(adjusted_topics) > 0:
                final_topic_list = "\n".join([
                    f"{t['name']} [{t['start_time']}-{t['end_time']}]"
                    for t in adjusted_topics
                ])
                predicted_topics = adjusted_topics
                print(f"✅ Successfully adjusted to {len(adjusted_topics)} balanced topics")
        
        processing_time = time.time() - start_time
        print(f"\n🎉 BALANCED TOPIC ANALYSIS COMPLETED!")
        print(f"⏱️ Total processing time: {processing_time:.1f} seconds ({processing_time/60:.1f} minutes)")
        
        results = {
            'transcript_file': transcript_file_path,
            'model_used': self.model,
            'processing_time': processing_time,
            'timestamp_format': self.detected_timestamp_format,
            'timestamps_found': len(timestamps),
            'predicted_topics': predicted_topics,
            'final_topic_list': final_topic_list,
            'user': 'niranjani1401',
            'generated_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        saved_file = self.save_rag_results(results, transcript_file_path)
        results['saved_file'] = saved_file
        return results

    def validate_and_adjust_topics(self, topics: List[Dict]) -> List[Dict]:
        adjusted_topics = []
        needs_adjustment = False
        too_short = [t for t in topics if t.get('too_short', False)]
        too_long = [t for t in topics if t.get('too_long', False)]
        if too_short or too_long:
            needs_adjustment = True
            print(f"⚠️ Found {len(too_short)} too short and {len(too_long)} too long topics that need adjustment")
        if not needs_adjustment:
            return topics
        topics_text = "\n".join([f"{t['name']} [{t['start_time']}-{t['end_time']}]" for t in topics])
        adjustment_prompt = f"""TASK: Adjust the following transcript topics to ensure optimal duration (7-15 minutes each).

Current topics:
{topics_text}

REQUIREMENTS:
- Merge adjacent too-short topics (<7 minutes) into more cohesive units
- Split too-long topics (>15 minutes) into smaller meaningful segments
- Keep topic names accurate and descriptive using instructor terminology
- Maintain chronological order and ensure no gaps in coverage
- Every topic MUST be 7-15 minutes in duration
- Format: topic name [starttime-endtime]
- DO NOT output any topic shorter than 7 minutes. If necessary, keep merging with subsequent topics until the duration is at least 7 minutes.
If merging, always join names with 'and', and the timestamps must cover the full range

ADJUSTED TOPICS:
"""
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": adjustment_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 3000,
                "top_p": 0.7
            }
        }
        try:
            response = requests.post(url, json=payload, timeout=300)
            if response.status_code == 200:
                adjusted_text = response.json().get("response", "").strip()
                adjusted_topics = self.parse_rag_topic_list(adjusted_text)
                print(f"✅ Successfully adjusted topics to 7-15 minute range")
                return adjusted_topics
        except Exception as e:
            print(f"❌ Topic adjustment failed: {e}")
        return topics

def main():
    print("🚀 RAG-OPTIMIZED TRANSCRIPT ANALYZER")
    print("✨ Features: Educational topics + 7-15min durations + Instructor terminology")
    print("🎯 Purpose: Optimized for RAG retrieval system with clear topic boundaries")
    print(f"👤 User: niranjani1401 | 📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    analyzer = RAGOptimizedTranscriptAnalyzer()
    transcript_file = input("Enter transcript file path: ").strip()
    if not transcript_file:
        transcript_file = "sample_transcript.txt"
    if not os.path.exists(transcript_file):
        print(f"❌ File not found: {transcript_file}")
        return
    print(f"\n🚀 Starting RAG-optimized analysis...")
    print(f"📄 File: {transcript_file}")
    print(f"⚡ Mode: Educational topic extraction")
    print(f"🎯 Target: 7-15 minute topics with clear boundaries")
    print(f"🔧 Focus: Algorithms, methods, procedures, comparisons")
    print("-" * 60)
    result = analyzer.analyze_transcript_for_rag(transcript_file_path=transcript_file)
    if result:
        rag_suitable = sum(1 for topic in result['predicted_topics'] if topic.get('balanced', False))
        avg_duration = sum(topic.get('duration_minutes', 0) for topic in result['predicted_topics']) / len(result['predicted_topics']) if result['predicted_topics'] else 0
        print(f"\n✨ RAG SUCCESS!")
        print(f"📊 Total topics extracted: {len(result['predicted_topics'])}")
        print(f"🎯 RAG-suitable topics (7+ min): {rag_suitable}/{len(result['predicted_topics'])} ({(rag_suitable/len(result['predicted_topics'])*100 if result['predicted_topics'] else 0):.1f}%)")
        print(f"⏱️ Average topic duration: {avg_duration:.1f} minutes")
        print(f"🕒 Timestamp format: {result['timestamp_format']}")
        print(f"⏱️ Processing time: {result['processing_time']:.1f} seconds")
        print(f"💾 RAG results saved to: {result['saved_file']}")
        if avg_duration >= 7:
            print(f"🏆 EXCELLENT: Average duration meets RAG requirements!")
        elif avg_duration >= 5:
            print(f"⚠️ GOOD: Most topics suitable for RAG, some may be short")
        else:
            print(f"❌ WARNING: Topics may be too short for optimal RAG performance")
    else:
        print("\n❌ RAG analysis failed.")

if __name__ == "__main__":
    main()