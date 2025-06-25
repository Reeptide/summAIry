import requests
import json
import time
from datetime import datetime
import threading
import os
import re
import google.generativeai as genai
from bert_score import score
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pandas as pd

class EnhancedTranscriptAnalyzer:
    def __init__(self, model="gemma3:4b", base_url="http://localhost:11434", gemini_api_key=None):
        self.model = model
        self.base_url = base_url  
        self.finished = False
        self.global_context = {
            'topics': [],
            'concepts': [],
            'timeline': [],
            'key_figures': [],
            'formulas': []
        }
        
        # Initialize Gemini
        if gemini_api_key:
            genai.configure(api_key=gemini_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-flash-2.5')
        else:
            self.gemini_model = None
            
        # Initialize sentence transformer for additional similarity metrics
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence transformer loaded for additional similarity metrics")
        except Exception as e:
            print(f"⚠️ Could not load sentence transformer: {e}")
            self.sentence_model = None
        
    def extract_timestamps_from_text(self, text):
        """Extract all timestamps from text and return them with their content"""
        timestamp_pattern = r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]'
        matches = re.findall(timestamp_pattern, text)
        
        # Also extract the content with timestamps
        timestamp_content_pattern = r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]\s*([^[]*?)(?=\[|$)'
        content_matches = re.findall(timestamp_content_pattern, text, re.DOTALL)
        
        return matches, content_matches
    
    def detect_topic_boundaries(self, text):
        """Detect potential topic boundaries using linguistic and structural cues"""
        # Common topic transition indicators for lectures
        transition_patterns = [
            r'\b(?:now|next|moving on|let\'s talk about|let\'s discuss|another|furthermore|however|but|in contrast|on the other hand)\b',
            r'\b(?:so|therefore|thus|hence|consequently|as a result)\b',
            r'\b(?:first|second|third|finally|lastly|in conclusion)\b',
            r'\b(?:what about|how about|let\'s see|let\'s look at|consider)\b',
            r'\b(?:okay|alright|right|well then|so then)\b',
            r'\b(?:chapter|section|part|topic|lesson|unit)\b',
        ]
        
        sentences = re.split(r'[.!?]+', text)
        topic_scores = []
        
        for i, sentence in enumerate(sentences):
            score = 0
            sentence_lower = sentence.lower()
            
            # Check for transition indicators
            for pattern in transition_patterns:
                if re.search(pattern, sentence_lower):
                    score += 1
            
            # Check for questions (often indicate topic shifts)
            if '?' in sentence:
                score += 0.5
            
            # Check for new timestamp (strong topic boundary indicator)
            if re.search(r'\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]', sentence):
                score += 2
            
            # Penalize very short sentences
            if len(sentence.split()) < 3:
                score -= 0.5
            
            topic_scores.append(score)
        
        return sentences, topic_scores
    
    def chunk_transcript_smart(self, text, max_chunk_size=2500, min_chunk_size=1000, overlap_sentences=7):
        """
        Smart chunking strategy optimized for topical segmentation with the following features:
        1. Preserves timestamp boundaries (never splits within a timestamp segment)
        2. Respects topic boundaries using linguistic cues
        3. Maintains semantic coherence
        4. Provides intelligent overlap for context continuity
        5. Handles edge cases gracefully
        """
        
        # First, split by timestamp segments to preserve temporal structure
        timestamp_pattern = r'(\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\])'
        timestamp_segments = re.split(timestamp_pattern, text)
        
        # Reconstruct complete timestamp segments with their content
        complete_segments = []
        i = 0
        while i < len(timestamp_segments):
            if i + 1 < len(timestamp_segments) and re.match(r'\[\d{1,2}:\d{2}\s*-\s*\d{1,2}:\d{2}\]', timestamp_segments[i]):
                # This is a timestamp, combine with following content
                timestamp_part = timestamp_segments[i]
                content_part = timestamp_segments[i + 1] if i + 1 < len(timestamp_segments) else ""
                complete_segments.append({
                    'timestamp': timestamp_part,
                    'content': content_part,
                    'full_text': timestamp_part + content_part,
                    'word_count': len((timestamp_part + content_part).split())
                })
                i += 2
            else:
                # Non-timestamp content (usually intro text)
                if timestamp_segments[i].strip():
                    complete_segments.append({
                        'timestamp': None,
                        'content': timestamp_segments[i],
                        'full_text': timestamp_segments[i],
                        'word_count': len(timestamp_segments[i].split())
                    })
                i += 1
        
        print(f"📊 Found {len(complete_segments)} timestamp segments")
        
        # Now create chunks respecting topic boundaries
        chunks = []
        current_chunk_segments = []
        current_word_count = 0
        
        for seg_idx, segment in enumerate(complete_segments):
            segment_words = segment['word_count']
            
            # Check if adding this segment would exceed max size
            if current_word_count + segment_words > max_chunk_size and current_chunk_segments:
                
                # Look for optimal break point within current chunk using topic detection
                chunk_text = ' '.join([s['full_text'] for s in current_chunk_segments])
                
                # If chunk is too small, try to add current segment anyway
                if current_word_count < min_chunk_size and current_word_count + segment_words <= max_chunk_size * 1.2:
                    current_chunk_segments.append(segment)
                    current_word_count += segment_words
                    continue
                
                # Create the chunk
                chunks.append({
                    'text': chunk_text,
                    'segments': current_chunk_segments.copy(),
                    'word_count': current_word_count,
                    'start_time': current_chunk_segments[0]['timestamp'] if current_chunk_segments[0]['timestamp'] else "N/A",
                    'end_time': current_chunk_segments[-1]['timestamp'] if current_chunk_segments[-1]['timestamp'] else "N/A"
                })
                
                # Calculate intelligent overlap
                overlap_segments = self.calculate_smart_overlap(current_chunk_segments, overlap_sentences)
                
                # Start new chunk with overlap + current segment
                current_chunk_segments = overlap_segments + [segment]
                current_word_count = sum(s['word_count'] for s in current_chunk_segments)
                
            else:
                # Add segment to current chunk
                current_chunk_segments.append(segment)
                current_word_count += segment_words
        
        # Handle the last chunk
        if current_chunk_segments:
            chunk_text = ' '.join([s['full_text'] for s in current_chunk_segments])
            chunks.append({
                'text': chunk_text,
                'segments': current_chunk_segments,
                'word_count': current_word_count,
                'start_time': current_chunk_segments[0]['timestamp'] if current_chunk_segments[0]['timestamp'] else "N/A",
                'end_time': current_chunk_segments[-1]['timestamp'] if current_chunk_segments[-1]['timestamp'] else "N/A"
            })
        
        # Extract just the text for processing (maintain compatibility)
        chunk_texts = [chunk['text'] for chunk in chunks]
        
        # Print chunking statistics
        print(f"📄 Smart chunking results:")
        print(f"   • Total chunks: {len(chunk_texts)}")
        print(f"   • Avg chunk size: {sum(chunk['word_count'] for chunk in chunks) / len(chunks):.0f} words")
        print(f"   • Size range: {min(chunk['word_count'] for chunk in chunks)}-{max(chunk['word_count'] for chunk in chunks)} words")
        print(f"   • Timestamp segments preserved: {len(complete_segments)}")
        
        # Store chunk metadata for potential future use
        self.chunk_metadata = chunks
        
        return chunk_texts
    
    def calculate_smart_overlap(self, segments, overlap_sentences=3):
        """Calculate intelligent overlap based on semantic boundaries and timestamps"""
        if len(segments) <= 1:
            return []
        
        # Always include the last timestamp segment for continuity
        overlap_segments = [segments[-1]]
        
        # Try to include additional segments up to overlap_sentences
        remaining_sentences = overlap_sentences - 1
        for i in range(len(segments) - 2, -1, -1):
            if remaining_sentences <= 0:
                break
            
            segment_sentences = len(re.split(r'[.!?]+', segments[i]['content']))
            if segment_sentences <= remaining_sentences:
                overlap_segments.insert(0, segments[i])
                remaining_sentences -= segment_sentences
        
        return overlap_segments
    
    def chunk_transcript(self, text, max_chunk_size=2500, overlap=512):
        """Wrapper to maintain compatibility - uses smart chunking"""
        return self.chunk_transcript_smart(text, max_chunk_size, overlap_sentences=3)
        
    def extract_context_from_previous_chunks(self, processed_results):
        """Build cumulative context from previously processed chunks with timestamp awareness"""
        if not processed_results:
            return ""
        
        context = "\n**CONTEXT FROM PREVIOUS CHUNKS:**\n"
        
        # Extract topics and timestamps mentioned so far
        topics_found = []
        last_timestamps = []
        
        for i, result in enumerate(processed_results, 1):
            # Extract timestamps from results
            timestamp_matches = re.findall(r'(\d{1,2}:\d{2})', result)
            if timestamp_matches:
                last_timestamps.extend(timestamp_matches[-3:])  # Keep last few timestamps
            
            # Extract topics with their timestamps from simple format
            lines = result.split('\n')
            for line in lines:
                if ':' in line and '[' in line and ']' in line:
                    topics_found.append(f"Chunk {i}: {line.strip()}")
        
        if last_timestamps:
            context += f"- **Latest timestamps processed**: {', '.join(last_timestamps[-5:])}\n"
        if topics_found:
            context += f"- **Topics covered so far**: {'; '.join(topics_found[:5])}\n"
        
        context += f"- **Previous chunks processed**: {len(processed_results)}\n"
        context += "**IMPORTANT: When analyzing the current chunk:**\n"
        context += "- Output ONLY in format: Topic Name : [MM:SS - MM:SS]\n"
        context += "- Each topic on a separate line\n"
        context += "- No additional text or explanations\n\n"
        
        return context
    
    def analyze_chunk_with_context(self, chunk, chunk_num, total_chunks, previous_results, method="streaming"):
        """Analyze chunk with simple topic:timestamp output format"""
        print(f"\n{'='*60}")
        print(f"🔄 CHUNK {chunk_num}/{total_chunks} - Started at: {datetime.now().strftime('%H:%M:%S')}")
        print(f"📝 Chunk size: {len(chunk.split())} words")
        
        # Extract timestamps from current chunk
        timestamps, content_with_timestamps = self.extract_timestamps_from_text(chunk)
        if timestamps:
            print(f"⏰ Timestamps found: {len(timestamps)} ranges")
            print(f"   First: [{timestamps[0][0]} - {timestamps[0][1]}]")
            print(f"   Last: [{timestamps[-1][0]} - {timestamps[-1][1]}]")
        print('='*60)
        
        # Build context from previous chunks
        context = self.extract_context_from_previous_chunks(previous_results)
        
        url = f"{self.base_url}/api/generate"
        
        prompt = f"""
You are analyzing part {chunk_num} of {total_chunks} of a lecture transcript for TOPICAL SEGMENTATION.

{context}

**CRITICAL OUTPUT FORMAT REQUIREMENT:**
You MUST output ONLY in this exact format:
Topic Name : [MM:SS - MM:SS]

**RULES:**
1. Each topic on a separate line
2. Use exact timestamps from the transcript
3. No additional text, explanations, or descriptions
4. No bullet points, numbers, or other formatting
5. Just: Topic Name : [MM:SS - MM:SS]

**EXAMPLES OF CORRECT OUTPUT:**
Introduction to Machine Learning : [00:00 - 02:30]
Linear Regression Basics : [02:31 - 05:45]
Neural Networks Overview : [05:46 - 08:20]

**CHUNK {chunk_num} TRANSCRIPT:**
{chunk}

Extract topics and their timestamps in the required format ONLY:
"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": method == "streaming",
            "options": {
                "temperature": 0.3,
                "top_p": 0.8,
                "num_predict": 1000
            }
        }
        
        if method == "streaming":
            return self._process_streaming(url, payload, chunk_num)
        else:
            return self._process_non_streaming(url, payload, chunk_num, total_chunks)
    
    def _process_streaming(self, url, payload, chunk_num):
        """Process with streaming output"""
        try:
            response = requests.post(url, json=payload, stream=True, timeout=600)
            
            if response.status_code == 200:
                chunk_response = ""
                print(f"📝 Analysis for chunk {chunk_num}:")
                print("-" * 40)
                
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                chunk_text = json_response['response']
                                print(chunk_text, end='', flush=True)
                                chunk_response += chunk_text
                            
                            if json_response.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                print(f"\n✅ Chunk {chunk_num} completed at: {datetime.now().strftime('%H:%M:%S')}")
                return chunk_response
            else:
                return f"Error processing chunk {chunk_num}: {response.status_code}"
        except Exception as e:
            return f"Error in chunk {chunk_num}: {str(e)}"
    
    def _process_non_streaming(self, url, payload, chunk_num, total_chunks):
        """Process without streaming (with progress indicator)"""
        self.finished = False
        progress_thread = threading.Thread(target=self.show_progress, args=(chunk_num, total_chunks))
        progress_thread.start()
        
        try:
            response = requests.post(url, json=payload, timeout=1200)
            self.finished = True
            progress_thread.join()
            
            print(f"\n✅ Chunk {chunk_num} completed at: {datetime.now().strftime('%H:%M:%S')}")
            
            if response.status_code == 200:
                result = response.json()["response"]
                print(f"\n📄 Results for chunk {chunk_num}:")
                print("-" * 40)
                print(result)
                return result
            else:
                return f"Error processing chunk {chunk_num}: {response.status_code}"
        except Exception as e:
            self.finished = True
            progress_thread.join()
            return f"Error in chunk {chunk_num}: {str(e)}"
    
    def generate_final_topic_list(self, all_results):
        """Generate final consolidated topic list in simple format"""
        print(f"\n🔗 GENERATING FINAL TOPIC LIST...")
        print(f"📊 Combining topics from {len(all_results)} chunks")
        
        url = f"{self.base_url}/api/generate"
        
        # Combine all chunk results
        combined_topics = []
        for i, result in enumerate(all_results, 1):
            if result.strip():
                combined_topics.append(f"# From Chunk {i}:\n{result}\n")
        
        all_chunk_topics = "\n".join(combined_topics)
        
        prompt = f"""
You have a set of topic extractions from multiple segments of a lecture transcript, each with a specific timestamp range. Your task is to generate a **final, consolidated, and chronologically ordered list of all unique topics** covered in the entire lecture.

**YOUR GOAL:** Ensure that every single topic discussed in the video is represented in the final output, covering the entire duration.

**INPUT DATA:**
The topic extractions will be provided in the following format (example):
CHUNK RESULTS:
Topic A : [00:00 - 01:30]
Topic B : [01:20 - 02:45]
Topic C : [02:30 - 03:50]
Topic A : [03:40 - 04:10]
... (and so on for all chunk results)

**OUTPUT FORMAT REQUIREMENT:**
You MUST output ONLY in this exact format, with each topic on a new line:
Topic Name : [MM:SS - MM:SS]

**RULES FOR CONSOLIDATION:**
1.  **Include All Topics:** Every unique topic identified across all chunks must be present in the final list. Do not omit any topics.
2.  **Remove Duplicates:** If the same topic appears multiple times, keep only one entry for that topic. The timestamp range for the duplicate topic should be the **most comprehensive** one (i.e., the earliest start time and the latest end time among all occurrences of that topic).
    * Example: If you have "Topic X : [05:00 - 06:00]" and "Topic X : [05:30 - 06:30]", the consolidated entry should be "Topic X : [05:00 - 06:30]".
3.  **Chronological Order:** The final list must be sorted strictly by the start timestamp of each topic, from earliest to latest.
4.  **No Extraneous Information:** Your output must contain ONLY the consolidated topic list. Do not include any introductory text, explanations, numbering, or additional formatting.
5.  **Exact Format:** Adhere strictly to "Topic Name : [MM:SS - MM:SS]".

**Provide the final consolidated topic list based on the provided CHUNK RESULTS:**

"""
        
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": 0.2,
                "top_p": 0.8,
                "num_predict": 2000
            }
        }
        
        try:
            response = requests.post(url, json=payload, stream=True, timeout=600)
            
            if response.status_code == 200:
                final_list = ""
                print(f"📝 Generating final topic list:")
                print("=" * 60)
                
                for line in response.iter_lines():
                    if line:
                        try:
                            json_response = json.loads(line)
                            if 'response' in json_response:
                                chunk_text = json_response['response']
                                print(chunk_text, end='', flush=True)
                                final_list += chunk_text
                            
                            if json_response.get('done', False):
                                break
                        except json.JSONDecodeError:
                            continue
                
                print(f"\n✅ Final topic list completed!")
                return final_list
            else:
                return f"Error generating final list: {response.status_code}"
        except Exception as e:
            return f"Error generating final list: {str(e)}"
    
    def get_gemini_ground_truth(self, transcript_text):
        """Get ground truth topic segmentation from Gemini API"""
        if not self.gemini_model:
            print("❌ Gemini API not configured. Please provide API key.")
            return None
        
        print(f"\n🔮 GENERATING GROUND TRUTH WITH GEMINI...")
        print(f"📊 Transcript length: {len(transcript_text)} characters")
        
        prompt = f"""
You are analyzing a lecture transcript for TOPICAL SEGMENTATION. Extract all the main topics covered in this lecture with their exact timestamps.

**CRITICAL OUTPUT FORMAT REQUIREMENT:**
You MUST output ONLY in this exact format:
Topic Name : [MM:SS - MM:SS]

**RULES:**
1. Each topic on a separate line
2. Use exact timestamps from the transcript
3. No additional text, explanations, or descriptions
4. No bullet points, numbers, or other formatting
5. Just: Topic Name : [MM:SS - MM:SS]

**EXAMPLES OF CORRECT OUTPUT:**
Introduction to Machine Learning : [00:00 - 02:30]
Linear Regression Basics : [02:31 - 05:45]
Neural Networks Overview : [05:46 - 08:20]

**TRANSCRIPT:**
{transcript_text}

Extract topics and their timestamps in the required format ONLY:
"""
        
        try:
            response = self.gemini_model.generate_content(prompt)
            ground_truth = response.text.strip()
            
            print(f"✅ Gemini ground truth generated!")
            print("=" * 60)
            print(ground_truth)
            print("=" * 60)
            
            return ground_truth
        except Exception as e:
            print(f"❌ Error getting Gemini ground truth: {e}")
            return None
    
    def parse_topic_list(self, topic_text):

        topics = []
        lines = topic_text.strip().split('\n')
    
        print(f"🔍 Parsing {len(lines)} lines for topics...")
    
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
                
            print(f"Processing line {line_num}: {line}")
            
            # Format 1: Topic Name : [MM:SS - MM:SS]
            if ' : [' in line and line.endswith(']'):
                parts = line.split(' : [')
                if len(parts) == 2:
                    topic_name = parts[0].strip()
                    timestamp_part = '[' + parts[1]
                    
                    timestamp_match = re.search(r'\[(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\]', timestamp_part)
                    if timestamp_match:
                        start_time = timestamp_match.group(1)
                        end_time = timestamp_match.group(2)
                        
                        topics.append({
                            'name': topic_name,
                            'start_time': start_time,
                            'end_time': end_time,
                            'full_text': line
                        })
                        print(f"  ✅ Format 1 parsed: {topic_name}")
                        continue
            
            # Format 2: Topic Name (MM:SS - MM:SS) - YOUR ACTUAL FORMAT
            timestamp_match = re.search(r'(.+?)\s*\((\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})\)', line)
            if timestamp_match:
                topic_name = timestamp_match.group(1).strip()
                start_time = timestamp_match.group(2)
                end_time = timestamp_match.group(3)
                
                topics.append({
                    'name': topic_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'full_text': line
                })
                print(f"  ✅ Format 2 parsed: {topic_name}")
                continue
            
            # Format 3: Topic Name (MM:SS-MM:SS) - no spaces around dash
            timestamp_match = re.search(r'(.+?)\s*\((\d{1,2}:\d{2})-(\d{1,2}:\d{2})\)', line)
            if timestamp_match:
                topic_name = timestamp_match.group(1).strip()
                start_time = timestamp_match.group(2)
                end_time = timestamp_match.group(3)
                
                topics.append({
                    'name': topic_name,
                    'start_time': start_time,
                    'end_time': end_time,
                    'full_text': line
                })
                print(f"  ✅ Format 3 parsed: {topic_name}")
                continue
                
            print(f"  ❌ Could not parse: {line}")
        
        print(f"✅ Successfully parsed {len(topics)} topics")
        return topics
        
    def calculate_bert_scores(self, predicted_topics, ground_truth_topics):
        """Calculate BERT scores between predicted and ground truth topics"""
        print(f"\n🎯 CALCULATING BERT SCORES...")
        print(f"📊 Predicted topics: {len(predicted_topics)}")
        print(f"📊 Ground truth topics: {len(ground_truth_topics)}")
        
        if not predicted_topics or not ground_truth_topics:
            print("❌ Cannot calculate BERT scores: empty topic lists")
            return None
        
        # Extract topic names for comparison
        pred_names = [topic['name'] for topic in predicted_topics]
        gt_names = [topic['name'] for topic in ground_truth_topics]
        
        print(f"🔍 Predicted topic names:")
        for i, name in enumerate(pred_names, 1):
            print(f"   {i}. {name}")
        
        print(f"🎯 Ground truth topic names:")
        for i, name in enumerate(gt_names, 1):
            print(f"   {i}. {name}")
        
        try:
            # Calculate BERT scores
            print(f"\n🧠 Computing BERT scores...")
            P, R, F1 = score(pred_names, gt_names, lang='en', verbose=True)
            
            # Convert to numpy for easier handling
            P_scores = P.numpy()
            R_scores = R.numpy()
            F1_scores = F1.numpy()
            
            # Calculate overall metrics
            bert_results = {
                'precision': {
                    'scores': P_scores,
                    'mean': float(np.mean(P_scores)),
                    'std': float(np.std(P_scores))
                },
                'recall': {
                    'scores': R_scores,
                    'mean': float(np.mean(R_scores)),
                    'std': float(np.std(R_scores))
                },
                'f1': {
                    'scores': F1_scores,
                    'mean': float(np.mean(F1_scores)),
                    'std': float(np.std(F1_scores))
                }
            }
            
            print(f"✅ BERT scores calculated successfully!")
            print(f"📊 Mean Precision: {bert_results['precision']['mean']:.4f} (±{bert_results['precision']['std']:.4f})")
            print(f"📊 Mean Recall: {bert_results['recall']['mean']:.4f} (±{bert_results['recall']['std']:.4f})")
            print(f"📊 Mean F1: {bert_results['f1']['mean']:.4f} (±{bert_results['f1']['std']:.4f})")
            
            return bert_results
            
        except Exception as e:
            print(f"❌ Error calculating BERT scores: {e}")
            return None
    
    def calculate_additional_metrics(self, predicted_topics, ground_truth_topics):
        """Calculate additional similarity metrics using sentence transformers"""
        if not self.sentence_model:
            return None
        
        print(f"\n📈 CALCULATING ADDITIONAL SIMILARITY METRICS...")
        
        try:
            pred_names = [topic['name'] for topic in predicted_topics]
            gt_names = [topic['name'] for topic in ground_truth_topics]
            
            # Get embeddings
            pred_embeddings = self.sentence_model.encode(pred_names)
            gt_embeddings = self.sentence_model.encode(gt_names)
            
            # Calculate cosine similarity matrix
            similarity_matrix = cosine_similarity(pred_embeddings, gt_embeddings)
            
            # Find best matches for each predicted topic
            best_matches = []
            for i, pred_topic in enumerate(pred_names):
                best_match_idx = np.argmax(similarity_matrix[i])
                best_similarity = similarity_matrix[i][best_match_idx]
                
                best_matches.append({
                    'predicted': pred_topic,
                    'matched_ground_truth': gt_names[best_match_idx],
                    'similarity': float(best_similarity)
                })
            
            # Calculate overall metrics
            similarities = [match['similarity'] for match in best_matches]
            
            additional_metrics = {
                'cosine_similarity': {
                    'mean': float(np.mean(similarities)),
                    'std': float(np.std(similarities)),
                    'best_matches': best_matches
                },
                'topic_count_diff': len(predicted_topics) - len(ground_truth_topics)
            }
            
            print(f"✅ Additional metrics calculated!")
            print(f"📊 Mean Cosine Similarity: {additional_metrics['cosine_similarity']['mean']:.4f}")
            print(f"📊 Topic Count Difference: {additional_metrics['topic_count_diff']}")
            
            return additional_metrics
            
        except Exception as e:
            print(f"❌ Error calculating additional metrics: {e}")
            return None
    
    def save_evaluation_results(self, results, transcript_filename):
        """Save comprehensive evaluation results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(transcript_filename))[0]
        
        # Save detailed results as JSON
        json_filename = f'evaluation_results_{base_name}_{timestamp}.json'
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        # Save summary as text
        summary_filename = f'evaluation_summary_{base_name}_{timestamp}.txt'
        with open(summary_filename, 'w', encoding='utf-8') as f:
            f.write("TOPIC SEGMENTATION EVALUATION SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Transcript: {transcript_filename}\n")
            f.write(f"Model: {self.model}\n\n")
            
            # BERT Scores
            if results.get('bert_scores'):
                bert = results['bert_scores']
                f.write("BERT SCORES:\n")
                f.write(f"  Precision: {bert['precision']['mean']:.4f} (±{bert['precision']['std']:.4f})\n")
                f.write(f"  Recall: {bert['recall']['mean']:.4f} (±{bert['recall']['std']:.4f})\n")
                f.write(f"  F1: {bert['f1']['mean']:.4f} (±{bert['f1']['std']:.4f})\n\n")
            
            # Additional metrics
            if results.get('additional_metrics'):
                add_metrics = results['additional_metrics']
                f.write("ADDITIONAL METRICS:\n")
                f.write(f"  Mean Cosine Similarity: {add_metrics['cosine_similarity']['mean']:.4f}\n")
                f.write(f"  Topic Count Difference: {add_metrics['topic_count_diff']}\n\n")
            
            # Topic comparison
            f.write("TOPIC COMPARISON:\n")
            f.write(f"  Predicted Topics ({len(results.get('predicted_topics', []))}):\n")
            for i, topic in enumerate(results.get('predicted_topics', []), 1):
                f.write(f"    {i}. {topic['name']} : [{topic['start_time']} - {topic['end_time']}]\n")
            
            f.write(f"\n  Ground Truth Topics ({len(results.get('ground_truth_topics', []))}):\n")
            for i, topic in enumerate(results.get('ground_truth_topics', []), 1):
                f.write(f"    {i}. {topic['name']} : [{topic['start_time']} - {topic['end_time']}]\n")
            
            # Best matches
            if results.get('additional_metrics') and 'best_matches' in results['additional_metrics']['cosine_similarity']:
                f.write(f"\n  BEST MATCHES (Cosine Similarity):\n")
                for match in results['additional_metrics']['cosine_similarity']['best_matches']:
                    f.write(f"    {match['predicted']} → {match['matched_ground_truth']} (sim: {match['similarity']:.4f})\n")
        
        print(f"\n💾 Evaluation results saved:")
        print(f"   📄 Detailed results: {json_filename}")
        print(f"   📋 Summary: {summary_filename}")
        
        return json_filename, summary_filename
    
    def show_progress(self, chunk_num=None, total_chunks=None):
        """Show animated progress indicator"""
        dots = 0
        while not self.finished:
            if chunk_num and total_chunks:
                print(f"\r🤔 Processing chunk {chunk_num}/{total_chunks} - Llama is thinking{'.' * (dots % 4)}{' ' * (3 - (dots % 4))}", end='', flush=True)
            else:
                print(f"\r🤔 Llama is thinking{'.' * (dots % 4)}{' ' * (3 - (dots % 4))}", end='', flush=True)
            dots += 1
            time.sleep(1.5)
    
    def save_simple_results(self, final_topic_list, transcript_filename, method_used):
        """Save simple topic:timestamp format results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = os.path.splitext(os.path.basename(transcript_filename))[0]
        filename = f'topic_segmentation_{base_name}_{method_used}_{timestamp}.txt'
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_topic_list)
        
        print(f"\n💾 Simple topic segmentation saved to: {filename}")
        return filename
    
    def test_connection(self):
        """Test if Ollama is accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                available_models = [model['name'] for model in models]
                print(f"✅ Connected to Ollama!")
                print(f"📋 Available models: {', '.join(available_models)}")
                
                if self.model in available_models:
                    print(f"🎯 Using model: {self.model}")
                    return True
                else:
                    print(f"⚠️  Model '{self.model}' not found. Available: {available_models}")
                    return False
            else:
                print(f"❌ Ollama responded with error: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to Ollama: {e}")
            print("💡 Make sure Ollama is running: 'ollama serve'")
            return False
    
    def analyze_full_transcript_with_context(self, transcript_file_path, method="streaming", chunk_size=2500, evaluate_with_gemini=False):
        """Enhanced main method with BERT score evaluation"""
        print(f"🚀 ENHANCED TOPIC SEGMENTATION ANALYZER WITH BERT EVALUATION")
        print(f"📅 Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"👤 User: Atulk47")
        print(f"🎯 Output Format: Topic : [MM:SS - MM:SS]")
        print(f"🔬 Evaluation: {'Enabled' if evaluate_with_gemini else 'Disabled'}")
        print("="*60)
        
        # Test connection first
        if not self.test_connection():
            return None
        
        # Read transcript
        try:
            with open(transcript_file_path, 'r', encoding='utf-8') as f:
                transcript = f.read()
            print(f"📄 Transcript loaded: {len(transcript)} characters, {len(transcript.split())} words")
            
            # Check timestamp format
            timestamps, _ = self.extract_timestamps_from_text(transcript)
            print(f"⏰ Timestamp analysis: Found {len(timestamps)} timestamp ranges")
            if timestamps:
                print(f"   Sample: [{timestamps[0][0]} - {timestamps[0][1]}] to [{timestamps[-1][0]} - {timestamps[-1][1]}]")
            
        except FileNotFoundError:
            print(f"❌ File not found: {transcript_file_path}")
            return None
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return None
        
        # Get ground truth from Gemini if evaluation is enabled
        ground_truth_text = None
        ground_truth_topics = None
        
        if evaluate_with_gemini:
            ground_truth_text = self.get_gemini_ground_truth(transcript)
            if ground_truth_text:
                ground_truth_topics = self.parse_topic_list(ground_truth_text)
                print(f"🎯 Ground truth parsed: {len(ground_truth_topics)} topics")
        
        # Use smart chunking optimized for topical segmentation
        chunks = self.chunk_transcript_smart(transcript, max_chunk_size=chunk_size)
        
        # Process chunks with simple output format
        results = []
        total_start_time = time.time()
        
        for i, chunk in enumerate(chunks, 1):
            chunk_start_time = time.time()
            
            # Pass previous results for context
            result = self.analyze_chunk_with_context(chunk, i, len(chunks), results, method)
            results.append(result)
            
            chunk_duration = time.time() - chunk_start_time
            print(f"⏱️  Chunk {i} took: {chunk_duration:.1f} seconds")
            
            # Brief pause between chunks
            if i < len(chunks):
                print("💤 Pausing 3 seconds before next chunk...")
                time.sleep(3)
        
        print(f"\n🔗 All chunks processed! Generating final topic list...")
        
        # Generate final consolidated topic list
        final_topic_list = self.generate_final_topic_list(results)
        
        total_duration = time.time() - total_start_time
        print(f"\n🎉 TOPIC SEGMENTATION COMPLETED!")
        print(f"⏱️  Total time: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
        print(f"📊 Processed {len(chunks)} chunks")
        print(f"🎯 Output format: Topic : [MM:SS - MM:SS]")
        
        # Save simple results
        saved_file = self.save_simple_results(final_topic_list, transcript_file_path, method)
        
        # Parse predicted topics
        predicted_topics = self.parse_topic_list(final_topic_list)
        print(f"🔍 Predicted topics parsed: {len(predicted_topics)} topics")
        
        # Evaluation results
        evaluation_results = {
            'transcript_file': transcript_file_path,
            'model_used': self.model,
            'processing_time': total_duration,
            'chunks_processed': len(chunks),
            'timestamps_found': len(timestamps),
            'predicted_topics': predicted_topics,
            'predicted_topic_list': final_topic_list,
            'ground_truth_topics': ground_truth_topics,
            'ground_truth_topic_list': ground_truth_text,
            'file_saved': saved_file
        }
        
        # Calculate BERT scores if ground truth is available
        if evaluate_with_gemini and ground_truth_topics and predicted_topics:
            print(f"\n🎯 STARTING EVALUATION...")
            
            # Calculate BERT scores
            bert_scores = self.calculate_bert_scores(predicted_topics, ground_truth_topics)
            evaluation_results['bert_scores'] = bert_scores
            
            # Calculate additional metrics
            additional_metrics = self.calculate_additional_metrics(predicted_topics, ground_truth_topics)
            evaluation_results['additional_metrics'] = additional_metrics
            
            # Save evaluation results
            json_file, summary_file = self.save_evaluation_results(evaluation_results, transcript_file_path)
            evaluation_results['evaluation_files'] = {
                'json': json_file,
                'summary': summary_file
            }
            
            print(f"\n🏆 EVALUATION COMPLETE!")
            if bert_scores:
                print(f"📊 BERT F1 Score: {bert_scores['f1']['mean']:.4f}")
            if additional_metrics:
                print(f"📊 Cosine Similarity: {additional_metrics['cosine_similarity']['mean']:.4f}")
        
        return evaluation_results

def main():
    """Main function with BERT evaluation capabilities"""
    print("🎓 ENHANCED TOPIC SEGMENTATION ANALYZER WITH BERT EVALUATION")
    print("🎯 Output Format: Topic : [MM:SS - MM:SS]")
    print("🧠 Smart Chunking + Simple Output + BERT Evaluation")
    print("="*60)
    
    # Get API keys
    gemini_api_key = input("Enter Gemini API key (or press Enter to skip evaluation): ").strip()
    if not gemini_api_key:
        gemini_api_key = None
        print("⚠️ No Gemini API key provided. Evaluation will be skipped.")
    
    # Initialize analyzer
    analyzer = EnhancedTranscriptAnalyzer(gemini_api_key=gemini_api_key)
    
    # Get transcript file path
    transcript_file = input("Enter transcript file path (default: C:\\Users\\aimlc\\OneDrive\\Desktop\\test\\sample_transcript.txt): ").strip()
    if not transcript_file:
        transcript_file = "C:\\Users\\aimlc\\OneDrive\\Desktop\\test\\sample_transcript.txt"

    if not os.path.exists(transcript_file):
        print(f"❌ File not found: {transcript_file}")
        return
    
    # Choose evaluation mode
    evaluate_with_gemini = False
    if gemini_api_key:
        eval_choice = input("\n🔬 Enable BERT evaluation with Gemini ground truth? (y/N): ").strip().lower()
        evaluate_with_gemini = eval_choice in ['y', 'yes']
    
    # Choose method
    print("\n📋 Choose processing method:")
    print("1. Streaming (see output as it's generated) - Recommended")
    print("2. Non-streaming (with progress indicators)")
    
    method_choice = input("Enter choice (1 or 2, default=1): ").strip()
    method = "streaming" if method_choice != "2" else "non-streaming"
    
    # Choose chunk size
    print(f"\n📦 Current chunk size: 2500 words")
    chunk_size = input("Enter different chunk size (or press Enter for default): ").strip()
    chunk_size = int(chunk_size) if chunk_size.isdigit() else 2500
    
    print(f"\n🚀 Starting enhanced topic segmentation analysis...")
    print(f"🔧 Method: {method}")
    print(f"📦 Chunk size: {chunk_size} words")
    print(f"🎯 Output: Topic : [MM:SS - MM:SS] format only")
    print(f"🔬 Evaluation: {'Enabled' if evaluate_with_gemini else 'Disabled'}")
    print("-" * 60)
    
    # Run analysis
    result = analyzer.analyze_full_transcript_with_context(
        transcript_file_path=transcript_file,
        method=method,
        chunk_size=chunk_size,
        evaluate_with_gemini=evaluate_with_gemini
    )
    
    if result:
        print(f"\n✨ SUCCESS! Topic segmentation saved to: {result['file_saved']}")
        print(f"📊 Summary: {result['chunks_processed']} chunks processed")
        print(f"⏰ Timestamps found: {result['timestamps_found']} ranges")
        print(f"⏱️  Total time: {result['processing_time']/60:.1f} minutes")
        
        if evaluate_with_gemini and result.get('bert_scores'):
            print(f"\n🏆 EVALUATION RESULTS:")
            bert = result['bert_scores']
            print(f"📊 BERT Precision: {bert['precision']['mean']:.4f}")
            print(f"📊 BERT Recall: {bert['recall']['mean']:.4f}")
            print(f"📊 BERT F1: {bert['f1']['mean']:.4f}")
            
            if result.get('additional_metrics'):
                add_metrics = result['additional_metrics']
                print(f"📊 Cosine Similarity: {add_metrics['cosine_similarity']['mean']:.4f}")
            
            if result.get('evaluation_files'):
                print(f"\n📁 Evaluation files saved:")
                print(f"   📄 Detailed: {result['evaluation_files']['json']}")
                print(f"   📋 Summary: {result['evaluation_files']['summary']}")
    else:
        print("\n❌ Analysis failed. Check the errors above.")

if __name__ == "__main__":
    # Check required packages
    required_packages = [
        ('bert_score', 'bert_score'),
        ('sentence_transformers', 'sentence_transformers'), 
        ('google-generativeai', 'google.generativeai')  # Package name vs import name
    ]
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print(f"💡 Install them with: pip install {' '.join(missing_packages)}")
        exit(1)
    
    main()