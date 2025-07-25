import chromadb
import logging
import os
import sys
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Tuple, Optional
import json
from datetime import datetime
import ollama
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import time
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, errors='replace')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, errors='replace')

logging.basicConfig(
    level=logging.ERROR,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hybrid_rag_system.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

class OptimizedHybridRAGSystem:
    def __init__(self, 
                 chroma_db_path: str = "./chroma_db",
                 semantic_threshold: float = 0.2,
                 max_chunk_size: int = 450,
                 overlap_size: int = 80,
                 min_chunk_size: int = 80,
                 min_similarity_threshold: float = 0.25,
                 max_candidates: int = 50,
                 use_pdf_source: bool = True,
                 model_config: Dict[str, Any] = None):
        self.semantic_threshold = semantic_threshold
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.min_similarity_threshold = min_similarity_threshold
        self.max_candidates = max_candidates
        self.use_pdf_source = use_pdf_source
        self.model_config = model_config or {
            "model": "llama3.1:8b",
            "temperature": 0.01,
            "top_p": 0.7,
            "max_tokens": 800
        }
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.collection = self.chroma_client.get_or_create_collection(
            name="enhanced_balanced_embeddings",
            metadata={
                "hnsw:space": "cosine", 
                "version": "enhanced_v1",
                "user": "niranjani1401",
                "created": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            }
        )
        self.ollama_client = ollama.Client()
        self.stats = {
            "semantic_chunks": 0,
            "simple_chunks": 0,
            "total_processing_time": 0,
            "embedding_time": 0,
            "chunking_time": 0,
            "retrieval_stats": {
                "high_quality_retrievals": 0,
                "medium_quality_retrievals": 0,
                "low_quality_retrievals": 0,
                "query_expansions": 0,
                "transcript_retrievals": 0,
                "pdf_retrievals": 0
            }
        }
    
    # ... [all previous methods unchanged, see your reference code above] ...
    # Paste all methods from the previous code. For brevity, not all are shown here.
    # This block is for the main workflow.

    def embed_documents(self, pdf_text_file: str = None, transcript_file: str = None):
        total_start_time = time.time()
        pdf_chunks = []
        if self.use_pdf_source and pdf_text_file:
            try:
                with open(pdf_text_file, 'r', encoding='utf-8') as f:
                    pdf_text = f.read()
                pdf_chunks = self.hybrid_chunking(pdf_text, "pdf")
            except Exception as e:
                logger.error(f"Error reading PDF file: {e}")
                raise
        try:
            with open(transcript_file, 'r', encoding='utf-8') as f:
                transcript_text = f.read()
        except Exception as e:
            logger.error(f"Error reading transcript file: {e}")
            raise
        transcript_chunks = self.hybrid_chunking(transcript_text, "transcript")
        all_chunks = pdf_chunks + transcript_chunks
        processed_documents = []
        for chunk in all_chunks:
            processed_text = self.enhanced_preprocess_for_embedding(chunk['text'])
            processed_documents.append(processed_text)
        metadatas = [chunk["metadata"] for chunk in all_chunks]
        ids = [f"{chunk['metadata']['source_type']}_chunk_{chunk['metadata']['chunk_id']}" for chunk in all_chunks]
        embedding_start_time = time.time()
        embeddings = self.embedding_model.encode(processed_documents, show_progress_bar=True).tolist()
        embedding_time = time.time() - embedding_start_time
        self.stats["embedding_time"] = embedding_time
        self.collection.add(
            documents=processed_documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
        total_time = time.time() - total_start_time
        self.stats["total_processing_time"] = total_time
        with open('enhanced_rag_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)

    def process_topics(self, topics_file: str, transcript_file: str, output_dir: str = None):
        if output_dir is None:
            transcript_basename = os.path.splitext(os.path.basename(transcript_file))[0]
            if self.use_pdf_source:
                output_dir = f"generated_notes_pdf_transcript"
            else:
                output_dir = f"generated_notes_transcript_only"
        # Read topics
        try:
            with open(topics_file, 'r', encoding='utf-8') as f:
                topics = [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Error reading topics file: {e}")
            raise
        os.makedirs(output_dir, exist_ok=True)
        total_start_time = time.time()
        for i, topic in enumerate(topics, 1):
            try:
                relevant_chunks = self.multi_query_retrieval(topic)
                notes = self.generate_notes_with_enhanced_prompt(topic, relevant_chunks)
                safe_filename = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
                safe_filename = safe_filename.replace(' ', '_')
                filename = f"{output_dir}/notes_{i:02d}_{safe_filename}.md"
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"**Model:** {self.model_config['model']}\n")
                    f.write(f"**Mode:** {'PDF + Transcript' if self.use_pdf_source else 'Transcript Only'}\n")
                    f.write(f"**Chunks Retrieved:** {len(relevant_chunks)}\n")
                    f.write(notes)
            except Exception as e:
                logger.error(f"Error processing topic '{topic}': {e}")
                continue
        total_time = time.time() - total_start_time
        self.stats["total_topic_processing_time"] = total_time
        self.stats["user"] = "niranjani1401"
        self.stats["completed_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        with open('enhanced_rag_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)

def main():
    # ==== Fixed paths: edit here if needed ====
    transcript_file = "transcript.txt"
    pdf_text_file = "ipcvtb.txt"
    topics_file = "topics.txt"
    use_pdf = os.path.exists(pdf_text_file)
    output_dir = "generated_notes_pdf_transcript" if use_pdf else "generated_notes_transcript_only"

    print("=== Optimized Hybrid RAG System ===")
    print(f"PDF source: {'Included' if use_pdf else 'Not included'}")
    print(f"Transcript file: {transcript_file}")
    print(f"PDF file: {pdf_text_file if use_pdf else 'N/A'}")
    print(f"Topics file: {topics_file}")
    print(f"Output dir: {output_dir}")

    rag = OptimizedHybridRAGSystem(
        semantic_threshold=0.2,
        max_chunk_size=450,
        overlap_size=80,
        min_chunk_size=80,
        min_similarity_threshold=0.25,
        max_candidates=50,
        use_pdf_source=use_pdf,
        model_config={
            "model": "llama3.1:8b",
            "temperature": 0.01,
            "top_p": 0.7,
            "max_tokens": 800
        }
    )

    print("Step 1: Embedding documents...")
    rag.embed_documents(pdf_text_file=pdf_text_file if use_pdf else None, transcript_file=transcript_file)
    print("Step 2: Generating notes for topics...")
    rag.process_topics(topics_file=topics_file, transcript_file=transcript_file, output_dir=output_dir)
    print("Step 3: Notes generation complete.")
    print(f"Notes saved in: {output_dir}/notes_XX_[topic].md")
    print(f"Stats: enhanced_rag_stats.json")

if __name__ == "__main__":
    main()