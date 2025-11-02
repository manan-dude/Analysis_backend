# main.py - FastAPI Backend for AI Analyst Platform
import uvicorn
import os
import json
import time
import re
import base64
import textwrap
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
import uuid
import os
from pathlib import Path
UPLOAD_DIR = "./uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
import asyncio
import fitz  # PyMuPDF
import chromadb
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
try:
    import ffmpeg
    from faster_whisper import WhisperModel
except ImportError:
    ffmpeg = None
    WhisperModel = None

# =========================================================================
# Logging Configuration
# =========================================================================

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('app.log')  # File output
    ]
)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
logger.info("🔧 Loading environment variables...")

# =========================================================================
# Configuration
# =========================================================================

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    logger.error("❌ GEMINI_API_KEY not found in environment!")
    raise ValueError("GEMINI_API_KEY not found in environment.")
else:
    logger.info("✅ GEMINI_API_KEY loaded successfully")

# Initialize clients
logger.info("🚀 Initializing Gemini client...")
client = genai.Client(api_key=API_KEY)
logger.info("✅ Gemini client initialized")

# Models
EMBEDDING_MODEL = 'models/text-embedding-004'
GENERATIVE_MODEL = 'gemini-2.5-flash'
logger.info(f"📊 Using Embedding Model: {EMBEDDING_MODEL}")
logger.info(f"🤖 Using Generative Model: {GENERATIVE_MODEL}")

# ChromaDB setup for RAG
CHROMA_DB_PATH = "./chroma_db"
logger.info(f"💾 Setting up ChromaDB at: {CHROMA_DB_PATH}")
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
logger.info("✅ ChromaDB client initialized")

# =========================================================================
# FastAPI App Setup
# =========================================================================

app = FastAPI(title="AI Analyst Platform API", version="1.0.0")
logger.info("🌐 FastAPI app created")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("✅ CORS middleware configured")

# =========================================================================
# Pydantic Models
# =========================================================================

class VideoPitchRequest(BaseModel):
    youtube_url: str

class CompetitorRequest(BaseModel):
    company_name: str
    company_url: str

class RAGQueryRequest(BaseModel):
    query: str
    collection_name: str = "user_documents"

class AnalysisResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    

# =========================================================================
# Helper Function - Extract YouTube Video ID
# =========================================================================

def extract_youtube_id(url: str) -> str:
    """Extract video ID from YouTube URL."""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    raise ValueError("Invalid YouTube URL")

# =========================================================================
# Helper Functions - Competitor Analysis
# =========================================================================

def analyze_competitor(company_identifier: str) -> Dict[str, Any]:
    """Find and analyze competitor using Gemini with Google Search."""
    logger.info(f"🔍 Starting competitor analysis for: {company_identifier}")
    
    prompt = f"""
    You are an expert market research analyst.
    Your task is to identify and analyze the primary, most direct competitor for:
    
    Target Company: **{company_identifier}**
    
    Provide comprehensive competitor analysis in the following JSON format:
    {{
        "target_company": "{company_identifier}",
        "competitor_name": "Name of Competitor",
        "competitor_url": "Official Website URL",
        "market_position": "Their market ranking/position",
        "target_audience": "Who their customers are",
        "pricing_strategy": "Their pricing model/approach",
        "key_products_services": ["Product 1", "Product 2", "Product 3"],
        "technology_stack": "Technologies they use",
        "funding_revenue": "Financial information if available",
        "geographic_presence": "Regions they operate in",
        "justification": "Why they are a key competitor (3-5 sentences)",
        "key_strength": "Most significant strength or USP",
        "competitive_advantages": ["Advantage 1", "Advantage 2", "Advantage 3"],
        "weaknesses": ["Weakness 1", "Weakness 2"],
        "recent_news": "Latest updates or developments",
        "social_media_presence": "Social media engagement summary"
    }}
    """
    
    try:
        logger.info("📡 Sending request to Gemini API with Google Search tool...")
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=prompt,
            config={"tools": [{"google_search": {}}]},
        )
        logger.info("✅ Received response from Gemini API")
        
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        analysis_data = json.loads(json_text)
        logger.info(f"✅ Successfully analyzed competitor: {analysis_data.get('competitor_name')}")
        return analysis_data
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ JSON parsing error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to parse AI response: {str(e)}")
    except Exception as e:
        logger.error(f"❌ Competitor analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Competitor analysis failed: {str(e)}")

# Initialize Whisper model (will download on first run)
whisper_model = None
if WhisperModel:
    try:
        whisper_model = WhisperModel("base", device="cpu", compute_type="int8")
        logger.info("✅ Whisper model initialized successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to initialize Whisper model: {e}")

# =========================================================================
# Helper Functions for Video Upload
# =========================================================================

async def extract_audio_from_video(video_path: str, audio_path: str) -> bool:
    """Extract audio from video file using FFmpeg."""
    try:
        await asyncio.to_thread(
            lambda: ffmpeg.input(video_path)
            .output(audio_path, acodec="mp3", ac=1, ar=16000)
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        return True
    except Exception as e:
        logger.error(f"FFmpeg extraction failed: {e}")
        return False

async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file using Faster-Whisper."""
    try:
        segments, info = await asyncio.to_thread(
            whisper_model.transcribe,
            audio_path,
            language="en"
        )
        return " ".join([seg.text for seg in segments])
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

def cleanup_files(*file_paths):
    """Clean up temporary files."""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup {file_path}: {e}")


# =========================================================================
# NEW Video Upload Analysis Endpoint
# =========================================================================
# =========================================================================
# CORRECTED Video Upload Endpoint
# =========================================================================

# @app.post("/api/video-pitch/upload")
# async def analyze_uploaded_video(file: UploadFile = File(...)):
#     """Process uploaded video using Gemini's native video understanding."""
    
#     logger.info("="*70)
#     logger.info("📤 VIDEO UPLOAD ANALYSIS REQUEST")

#     # Use a temporary file path
#     temp_video_path = f"./uploads/temp_{uuid.uuid4()}.mp4"
    
#     try:
#         # 1. Save video to a temporary file
#         with open(temp_video_path, "wb") as f:
#             f.write(await file.read())
        
#         # 2. Upload video to Gemini File API
#         logger.info("☁️ Uploading video to Gemini...")
#         video_file = client.files.upload(file=temp_video_path)
        
#         # 3. Wait for Gemini to process
#         logger.info("⏳ Waiting for Gemini to process video...")
#         while video_file.state.name == "PROCESSING":
#             time.sleep(2)
#             video_file = client.files.get(name=video_file.name)
        
#         if video_file.state.name == "FAILED":
#             raise Exception("Gemini video processing failed")
            
#         # 4. Analyze video with Gemini
#         logger.info("🤖 Analyzing video with Gemini...")
#         analysis_prompt = f"""
# You are an expert pitch analyst. Analyze the following video transcript from a pitch presentation.

# Provide a comprehensive analysis in the following JSON format:

# {{
#     "video_title": "Extract or infer the pitch/company name",
#     "duration": "Estimate duration from transcript",
#     "channel": "Speaker/Company name if mentioned",
#     "executive_summary": "2-3 sentence overview of the pitch",
#     "key_insights": [
#         "First major insight",
#         "Second major insight",
#         "Third major insight",
#         "Fourth major insight",
#         "Fifth major insight"
#     ],
#     "main_points": [
#         "First main point from the pitch",
#         "Second main point",
#         "Third main point",
#         "Fourth main point"
#     ],
#     "notable_slides": [
#         {{
#             "timestamp": "0:45",
#             "description": "Problem statement slide - described the market gap"
#         }},
#         {{
#             "timestamp": "2:30",
#             "description": "Solution overview - demonstrated the product"
#         }},
#         {{
#             "timestamp": "5:15",
#             "description": "Business model - explained revenue streams"
#         }},
#         {{
#             "timestamp": "7:20",
#             "description": "Traction metrics - showed growth numbers"
#         }}
#     ],
#     "action_items": [
#         "First recommendation or action item",
#         "Second recommendation",
#         "Third recommendation"
#     ]
# }}

# Focus on business insights, problem-solution fit, market opportunity, competitive advantages, and traction indicators.
# Output ONLY the JSON, no additional text.
# """
#         response = client.models.generate_content(
#             model=GENERATIVE_MODEL,
#             contents=[analysis_prompt, video_file]
#         )
    
        
#         # 5. Parse JSON response (THE CRITICAL FIX)
#         json_text = response.text.strip().replace("``````", "").strip()
        
#         try:
#             # ✅ CORRECT: Parse the string into a Python dictionary
#             analysis_data = json.loads(json_text)
#         except json.JSONDecodeError as e:
#             logger.error(f"❌ Failed to parse JSON: {e}")
#             raise Exception("Failed to parse AI analysis response")

#         # 6. Add metadata
#         analysis_data['filename'] = file.filename
        
#         # ✅ CORRECT: Return the parsed dictionary
#         return JSONResponse(content={
#             "success": True,
#             "data": analysis_data  # Now this is a proper dictionary
#         })
        
#     except Exception as e:
#         logger.error(f"❌ Video upload analysis failed: {e}")
#         return JSONResponse(
#             status_code=500,
#             content={"success": False, "error": str(e)}
#         )
    
#     finally:
#         # 7. Cleanup
#         if os.path.exists(temp_video_path):
#             os.remove(temp_video_path)
#         if 'video_file' in locals() and video_file:
#             try:
#                 client.files.delete(name=video_file.name)
#             except Exception:
#                 pass

@app.post("/api/video-pitch/upload")
async def analyze_uploaded_video(file: UploadFile = File(...)):
    """Process uploaded video: extract audio, transcribe, and analyze."""
    
    logger.info("="*70)
    logger.info("📤 VIDEO UPLOAD ANALYSIS REQUEST")
    
    if not ffmpeg or not whisper_model:
        return JSONResponse(
            status_code=503,
            content={"success": False, "error": "Video upload dependencies not installed. Run: pip install ffmpeg-python faster-whisper"}
        )

    file_id = str(uuid.uuid4())
    video_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp4")
    audio_path = os.path.join(UPLOAD_DIR, f"{file_id}.mp3")
    
    try:
        # 1. Save uploaded video
        with open(video_path, "wb") as f:
            f.write(await file.read())
        
        # 2. Extract audio
        if not await extract_audio_from_video(video_path, audio_path):
            raise Exception("Failed to extract audio from video")
            
        # 3. Transcribe audio
        transcript = await transcribe_audio(audio_path)
        print("transcript:", transcript)
        
        # 4. Analyze with Gemini
        analysis_prompt = f"""
You are an expert pitch analyst. Analyze the following video transcript from a pitch presentation.

TRANSCRIPT:
---
{transcript[:15000]}  # Limit to avoid token limits
---

Provide a comprehensive analysis in the following JSON format:

{{
    "video_title": "Extract or infer the pitch/company name",
    "duration": "Estimate duration from transcript",
    "channel": "Speaker/Company name if mentioned",
    "executive_summary": "2-3 sentence overview of the pitch",
    "key_insights": [
        "First major insight",
        "Second major insight",
        "Third major insight",
        "Fourth major insight",
        "Fifth major insight"
    ],
    "main_points": [
        "First main point from the pitch",
        "Second main point",
        "Third main point",
        "Fourth main point"
    ],
    "notable_slides": [
        {{
            "timestamp": "0:45",
            "description": "Problem statement slide - described the market gap"
        }},
        {{
            "timestamp": "2:30",
            "description": "Solution overview - demonstrated the product"
        }},
        {{
            "timestamp": "5:15",
            "description": "Business model - explained revenue streams"
        }},
        {{
            "timestamp": "7:20",
            "description": "Traction metrics - showed growth numbers"
        }}
    ],
    "action_items": [
        "First recommendation or action item",
        "Second recommendation",
        "Third recommendation"
    ]
}}

Focus on business insights, problem-solution fit, market opportunity, competitive advantages, and traction indicators.
Output ONLY the JSON, no additional text.
"""
        
    
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=analysis_prompt,
        )
        # 4. Correctly Parse the Response
        try:
            raw_text = response.candidates[0].content.parts[0].text
            json_text = raw_text.strip()
            if json_text.startswith("```"):
                json_text = json_text[7:]
            if json_text.endswith("```"):
                json_text = json_text[:-3]
            
            analysis_data = json.loads(json_text.strip())
            
        except (IndexError, AttributeError, json.JSONDecodeError) as e:
            logger.error(f"❌ Failed to parse JSON from Gemini: {e}")
            raise Exception("Failed to parse AI analysis response.")
        
        # 5. Add Metadata
        analysis_data['source'] = 'uploaded_video'
        analysis_data['filename'] = file.filename
        
        # 6. Return Correctly Formatted JSON
        return JSONResponse(content={
            "success": True,
            "data": analysis_data
        })
    
    except Exception as e:
        logger.error(f"❌ Video upload analysis failed: {e}")
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
    
    finally:
        # 7. Cleanup Files
        cleanup_files(video_path, audio_path)
# =========================================================================
# Helper Functions - RAG Analysis
# =========================================================================

def extract_text_from_pdf_bytes(pdf_bytes: bytes, max_pages: int = 10) -> List[str]:
    """Extract text from PDF bytes and chunk it."""
    logger.info(f"📄 Extracting text from PDF (max {max_pages} pages)...")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    total_pages = len(doc)
    logger.info(f"📖 PDF has {total_pages} pages")
    
    full_text = ""
    
    for page_num in range(min(max_pages, total_pages)):
        page = doc.load_page(page_num)
        text = page.get_text()
        full_text += f"\n--- Page {page_num + 1} ---\n" + text
        logger.debug(f"✅ Extracted text from page {page_num + 1}")
    
    logger.info("✂️ Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", "•", "!", "?", " ", ""]
    )
    
    chunks = text_splitter.split_text(full_text)
    logger.info(f"✅ Created {len(chunks)} text chunks")
    return chunks


# =========================================================================
# Video Pitch Analysis Endpoint
# =========================================================================

# @app.post("/api/video-pitch/analyze")
# async def analyze_video_pitch(request: VideoPitchRequest):
#     """Analyze YouTube video pitch and extract insights."""
#     logger.info("="*70)
#     logger.info("🎥 VIDEO PITCH ANALYSIS REQUEST")
#     logger.info(f"YouTube URL: {request.youtube_url}")
#     logger.info("="*70)
    
#     try:
#         # Extract video ID
#         video_id = extract_youtube_id(request.youtube_url)
#         logger.info(f"📹 Extracted video ID: {video_id}")
        
#         # Get transcript using the NEW API (v1.2.2+)
#         logger.info("📜 Fetching transcript...")
        
#         try:
#             # Initialize the API
#             ytt_api = YouTubeTranscriptApi()
            
#             # Fetch transcript (this returns a FetchedTranscript object)
#             fetched_transcript = ytt_api.fetch(video_id, languages=['en'])
            
#             # Convert to raw data (list of dicts)
#             transcript_list = fetched_transcript.to_raw_data()
            
#             logger.info(f"✅ Transcript fetched: {len(transcript_list)} entries")
            
#         except Exception as transcript_error:
#             logger.error(f"❌ Failed to fetch transcript: {transcript_error}")
#             return JSONResponse(
#                 status_code=400,
#                 content={
#                     "success": False, 
#                     "error": f"Could not fetch transcript: {str(transcript_error)}. The video may not have captions enabled or may be restricted."
#                 }
#             )
        
#         # Combine transcript into full text
#         full_transcript = " ".join([entry['text'] for entry in transcript_list])
#         logger.info(f"✅ Combined transcript: {len(full_transcript)} characters")
        
#         # Analyze with Gemini (your prompt remains the same)
#         # Analyze with Gemini
#         analysis_prompt = f"""
# You are an expert pitch analyst. Analyze the following video transcript from a pitch presentation.

# TRANSCRIPT:
# ---
# {full_transcript[:15000]}  # Limit to avoid token limits
# ---

# Provide a comprehensive analysis in the following JSON format:

# {{
#     "video_title": "Extract or infer the pitch/company name",
#     "duration": "Estimate duration from transcript",
#     "channel": "Speaker/Company name if mentioned",
#     "executive_summary": "2-3 sentence overview of the pitch",
#     "key_insights": [
#         "First major insight",
#         "Second major insight",
#         "Third major insight",
#         "Fourth major insight",
#         "Fifth major insight"
#     ],
#     "main_points": [
#         "First main point from the pitch",
#         "Second main point",
#         "Third main point",
#         "Fourth main point"
#     ],
#     "notable_slides": [
#         {{
#             "timestamp": "0:45",
#             "description": "Problem statement slide - described the market gap"
#         }},
#         {{
#             "timestamp": "2:30",
#             "description": "Solution overview - demonstrated the product"
#         }},
#         {{
#             "timestamp": "5:15",
#             "description": "Business model - explained revenue streams"
#         }},
#         {{
#             "timestamp": "7:20",
#             "description": "Traction metrics - showed growth numbers"
#         }}
#     ],
#     "action_items": [
#         "First recommendation or action item",
#         "Second recommendation",
#         "Third recommendation"
#     ]
# }}

# Focus on business insights, problem-solution fit, market opportunity, competitive advantages, and traction indicators.
# Output ONLY the JSON, no additional text.
# """
        
#         logger.info("🤖 Analyzing transcript with Gemini...")
#         response = client.models.generate_content(
#             model=GENERATIVE_MODEL,
#             contents=analysis_prompt,
#         )
#         print(response.text)
        
#         # ---- NEW: Robust JSON Parsing ----
#         json_text = response.text.strip()
        
#         if not json_text:
#             logger.error("❌ Gemini returned an empty response.")
#             return JSONResponse(
#                 status_code=500,
#                 content={"success": False, "error": "AI model returned an empty response."}
#             )
            
#         # Aggressively remove markdown formatting
#         if json_text.startswith("```"):
#             json_text = json_text[7:]
#         if json_text.endswith("```"):
#             json_text = json_text[:-3]
#         json_text = json_text.strip()
        
#         try:
#             analysis_data = json.loads(json_text)
#         except json.JSONDecodeError as e:
#             logger.error(f"❌ Failed to parse JSON response: {e}")
#             logger.error(f"Raw response: {json_text[:500]}")
#             return JSONResponse(
#                 status_code=500,
#                 content={"success": False, "error": "Failed to parse AI analysis response"}
#             )
        
#         # Add original URL and metadata to response
#         analysis_data['youtube_url'] = request.youtube_url
#         analysis_data['video_id'] = video_id
#         analysis_data['transcript_language'] = fetched_transcript.language
#         analysis_data['transcript_length'] = len(transcript_list)
        
#         logger.info("✅ Video pitch analysis completed successfully")
#         logger.info("="*70)
        
#         return JSONResponse(content={
#             "success": True,
#             "data": analysis_data
#         })
        
#     except ValueError as e:
#         logger.error(f"❌ Invalid YouTube URL: {e}")
#         return JSONResponse(
#             status_code=400,
#             content={"success": False, "error": "Invalid YouTube URL format"}
#         )
    
#     except Exception as e:
#         logger.error(f"❌ Video pitch analysis failed: {e}")
#         logger.error("="*70)
#         return JSONResponse(
#             status_code=500,
#             content={"success": False, "error": str(e)}
#         )
        
def embed_function_chroma(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using Gemini API."""
    logger.info(f"🔢 Generating embeddings for {len(texts)} texts...")
    str_texts = [str(t) for t in texts]
    
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=str_texts,
        )
        embeddings = [item.values for item in response.embeddings]
        logger.info(f"✅ Generated {len(embeddings)} embeddings")
        return embeddings
    except Exception as e:
        logger.error(f"❌ Embedding generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Embedding generation failed: {str(e)}")

def create_rag_collection(chunks: List[str], collection_name: str):
    """Create or update ChromaDB collection."""
    logger.info(f"💾 Creating RAG collection: {collection_name}")
    
    try:
        chroma_client.delete_collection(name=collection_name)
        logger.info(f"🗑️ Deleted existing collection: {collection_name}")
    except:
        logger.info(f"ℹ️ No existing collection to delete")
    
    embeddings = embed_function_chroma(chunks)
    ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
    
    collection = chroma_client.create_collection(name=collection_name)
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
    )
    
    logger.info(f"✅ Collection '{collection_name}' created with {len(chunks)} chunks")
    return collection

def query_rag(collection_name: str, question: str, k: int = 8) -> str:
    """Query RAG collection and get AI response."""
    logger.info(f"🔍 Querying RAG collection: {collection_name}")
    logger.info(f"❓ Question: {question}")
    
    try:
        collection = chroma_client.get_collection(name=collection_name)
        logger.info(f"✅ Retrieved collection: {collection_name}")
    except Exception as e:
        logger.error(f"❌ Collection not found: {collection_name}")
        raise HTTPException(status_code=404, detail="Collection not found. Please upload a document first.")
    
    query_embedding = embed_function_chroma([question])[0]
    logger.info(f"🔍 Searching for top {k} relevant chunks...")
    
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents']
    )
    retrieved_documents = results['documents'][0]
    logger.info(f"✅ Retrieved {len(retrieved_documents)} relevant chunks")
    
    context = "\n---\n".join(retrieved_documents)
    
    rag_prompt = textwrap.dedent(f"""
        You are an expert analyst. Answer the question based on the provided context.
        
        CONTEXT:
        ---
        {context}
        ---
        
        QUESTION: {question}
        
        Provide a clear, detailed answer based only on the context provided.
    """)
    
    logger.info("🤖 Generating AI response...")
    response = client.models.generate_content(
        model=GENERATIVE_MODEL,
        contents=rag_prompt,
    )
    
    logger.info("✅ AI response generated successfully")
    return response.text

# =========================================================================
# Helper Functions - AI Analyzer (Plot Extraction)
# =========================================================================

def get_page_image_data(pdf_bytes: bytes, page_num: int) -> str:
    """Render PDF page to base64 JPEG."""
    logger.debug(f"🖼️ Rendering page {page_num} to image...")
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if page_num > len(doc):
        logger.warning(f"⚠️ Page {page_num} exceeds document length")
        return None
    
    page = doc.load_page(page_num - 1)
    zoom = 3
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    
    img_bytes = pix.tobytes("jpeg")
    base64_img = base64.b64encode(img_bytes).decode('utf-8')
    logger.debug(f"✅ Page {page_num} rendered successfully")
    return base64_img

def extract_structured_page_data(base64_image: str, page_num: int) -> Optional[Dict[str, Any]]:
    """Extract tables and plots from PDF page image using multimodal analysis."""
    logger.info(f"📊 Extracting structured data from page {page_num}...")
    
    sde_prompt = """
    You are an expert visual data extraction agent. Analyze the attached image and extract ALL structured data.
    
    Combine ALL extracted data into a single JSON object with two top-level keys: "Tables" and "Plots".
    
    **For TABLES:** Extract row-by-row, column-by-column data.
    Format: [{"Title": "Table Name", "Data": [["Header1", "Header2"], [value1, value2]]}]
    
    **For PLOTS:** Extract all data points (X and Y coordinates).
    Format: [{"Chart_Title": "Title", "Series_Name": "Series", "X_Axis_Data": [...], "Y_Axis_Data": [...]}]
    
    Output ONLY JSON within ```json code block.
    If no data found, output: {"Tables": [], "Plots": []}
    """
    
    image_part = {
        "inlineData": {
            "data": base64_image,
            "mimeType": "image/jpeg"
        }
    }
    
    try:
        logger.info(f"🤖 Sending page {page_num} to Gemini for analysis...")
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=[image_part, sde_prompt]
        )
        
        json_match = re.search(r"```json\s*(\{.*?})\s*```", response.text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(1).strip())
            if not data.get("Tables") and not data.get("Plots"):
                logger.info(f"ℹ️ Page {page_num}: No structured data found")
                return None
            logger.info(f"✅ Page {page_num}: Found {len(data.get('Tables', []))} tables, {len(data.get('Plots', []))} plots")
            return data
        logger.info(f"ℹ️ Page {page_num}: No JSON data in response")
        return None
        
    except json.JSONDecodeError as e:
        logger.error(f"❌ Page {page_num}: JSON parsing error - {e}")
        return None
    except Exception as e:
        logger.error(f"❌ Page {page_num}: Extraction error - {e}")
        return None

def generate_document_analysis(extracted_data: List[Dict[str, Any]], pages_analyzed: int) -> Dict[str, Any]:
    """Generate comprehensive analysis summary from extracted data."""
    logger.info("📝 Generating comprehensive document analysis...")
    
    total_tables = sum(len(page.get('Tables', [])) for page in extracted_data)
    total_plots = sum(len(page.get('Plots', [])) for page in extracted_data)
    
    logger.info(f"📊 Summary: {total_tables} tables, {total_plots} plots from {pages_analyzed} pages")
    
    # Generate AI analysis
    # Generate AI analysis
    analysis_prompt = f"""
You are a senior financial analyst. A document was analyzed and extracted:
- {total_tables} data tables
- {total_plots} charts/graphs
- from {pages_analyzed} total pages.

Using this context, generate a **professional investment-focused report**.

✅ Your task:
Provide insights as if you reviewed a full company financial report and are advising an investor:
- explain revenue trends
- profitability direction
- risks
- future outlook
- whether the company is a good investment
- why or why not

✅ Output STRICTLY in JSON (no explanation outside JSON).  
✅ Follow this exact structure:

{{
    "key_insights": [
        "Clear, sharp insights about the company's performance and future",
        "Investment-focused observations, not description of extracted tables"
    ],
    "financial_metrics": {{
        "revenue_trends": "Explain long-term revenue performance and whether growth is accelerating or slowing",
        "profit_margins": "Are margins improving? stable? declining?",
        "growth_rates": "Is the company growing fast? moderately? stagnating?",
        "kpis": "Key KPIs relevant for investors (ROE, ROA, CAC, LTV, etc.)"
    }},
    "visual_analysis": {{
        "chart_types": "What types of charts typically represent such financial data (trend lines, YoY bars, pie charts)",
        "completeness_score": "Rate perceived completeness of financial visuals (1–10)",
        "quality_assessment": "Interpret if the underlying data looks strong, weak, volatile, or stable"
    }},
    "recommendations": [
        "Clear investment recommendation (Buy/Hold/Sell)",
        "Why this recommendation is justified",
        "Actionable strategic suggestion for investors"
    ],
    "risk_factors": [
        "Major risks like market volatility, competition, debt, declining demand, etc.",
        "Any warning signs investors must consider"
    ]
}}

Make the insights **specific, realistic, investment oriented**, and **not generic**.
"""
    
    try:
        logger.info("🤖 Generating AI analysis summary...")
        response = client.models.generate_content(
            model=GENERATIVE_MODEL,
            contents=analysis_prompt,
        )
        json_text = response.text.strip().replace("```json", "").replace("```", "")
        analysis = json.loads(json_text)
        logger.info("✅ AI analysis summary generated successfully")
    except Exception as e:
        logger.warning(f"⚠️ Failed to generate AI analysis, using fallback: {e}")
        analysis = {
            "key_insights": ["Data extraction completed successfully"],
            "financial_metrics": {"revenue_trends": "N/A", "profit_margins": "N/A", "growth_rates": "N/A", "kpis": "N/A"},
            "visual_analysis": {"chart_types": "Various", "completeness_score": "N/A", "quality_assessment": "Good"},
            "recommendations": ["Review extracted visualizations"],
            "risk_factors": ["Limited data availability"]
        }
    
    return {
        "document_overview": {
            "pages_analyzed": pages_analyzed,
            "document_type": "Financial/Business Report",
            "analysis_timestamp": datetime.now().isoformat()
        },
        "data_summary": {
            "tables_found": total_tables,
            "charts_found": total_plots,
            "key_metrics": ["Revenue", "Growth", "Performance"],
            "date_range": "Multi-year analysis"
        },
        **analysis
    }

# =========================================================================
# API Endpoints
# =========================================================================

@app.get("/")
async def root():
    logger.info("📍 Root endpoint accessed")
    return {"message": "AI Analyst Platform API", "version": "1.0.0", "status": "running"}

@app.get("/health")
async def health_check():
    logger.info("🏥 Health check endpoint accessed")
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# -------------------------------------------------------------------------
# Competitor Analysis Endpoints
# -------------------------------------------------------------------------

@app.post("/api/competitor/analyze")
async def analyze_competitor_endpoint(request: CompetitorRequest):
    """Analyze competitor for given company."""
    logger.info("="*70)
    logger.info("🔍 COMPETITOR ANALYSIS REQUEST")
    logger.info(f"Company Name: {request.company_name}")
    logger.info(f"Company URL: {request.company_url}")
    logger.info("="*70)
    
    try:
        company_identifier = f"{request.company_name} ({request.company_url})"
        result = analyze_competitor(company_identifier)
        
        logger.info("✅ Competitor analysis completed successfully")
        logger.info("="*70)
        return JSONResponse(content={
            "success": True,
            "data": result
        })
    except Exception as e:
        logger.error(f"❌ Competitor analysis failed: {e}")
        logger.error("="*70)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# -------------------------------------------------------------------------
# RAG Analyzer Endpoints
# -------------------------------------------------------------------------

@app.post("/api/rag/upload")
async def upload_rag_document(file: UploadFile = File(...)):
    """Upload document for RAG analysis."""
    logger.info("="*70)
    logger.info("📤 RAG DOCUMENT UPLOAD REQUEST")
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Content Type: {file.content_type}")
    logger.info("="*70)
    
    try:
        if not file.filename.endswith('.pdf'):
            logger.error(f"❌ Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info("📖 Reading PDF file...")
        pdf_bytes = await file.read()
        logger.info(f"✅ Read {len(pdf_bytes)} bytes")
        
        chunks = extract_text_from_pdf_bytes(pdf_bytes, max_pages=5)
        
        collection_name = f"user_doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        create_rag_collection(chunks, collection_name)
        
        logger.info("✅ RAG document upload completed successfully")
        logger.info("="*70)
        return JSONResponse(content={
            "success": True,
            "data": {
                "message": "Document uploaded and indexed successfully",
                "collection_name": collection_name,
                "chunks_created": len(chunks)
            }
        })
    except Exception as e:
        logger.error(f"❌ RAG upload failed: {e}")
        logger.error("="*70)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.post("/api/rag/query")
async def query_rag_endpoint(request: RAGQueryRequest):
    """Query the RAG system."""
    logger.info("="*70)
    logger.info("💬 RAG QUERY REQUEST")
    logger.info(f"Collection: {request.collection_name}")
    logger.info(f"Query: {request.query}")
    logger.info("="*70)
    
    try:
        answer = query_rag(request.collection_name, request.query)
        
        logger.info("✅ RAG query completed successfully")
        logger.info(f"📝 Answer length: {len(answer)} characters")
        logger.info("="*70)
        return JSONResponse(content={
            "success": True,
            "data": {
                "query": request.query,
                "answer": answer
            }
        })
    except Exception as e:
        logger.error(f"❌ RAG query failed: {e}")
        logger.error("="*70)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# -------------------------------------------------------------------------
# AI Analyzer (Plot Extraction) Endpoints
# -------------------------------------------------------------------------

@app.post("/api/ai-analyzer/analyze")
async def analyze_pdf_plots(file: UploadFile = File(...)):
    """Analyze PDF and extract plots, tables, and generate insights."""
    logger.info("="*70)
    logger.info("🤖 AI ANALYZER REQUEST")
    logger.info(f"Filename: {file.filename}")
    logger.info(f"Content Type: {file.content_type}")
    logger.info("="*70)
    
    try:
        if not file.filename.endswith('.pdf'):
            logger.error(f"❌ Invalid file type: {file.filename}")
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        logger.info("📖 Reading PDF file...")
        pdf_bytes = await file.read()
        logger.info(f"✅ Read {len(pdf_bytes)} bytes")
        
        pages_to_analyze = 6
        all_page_results = []
        
        # Extract data from each page
        logger.info(f"🔍 Analyzing up to {pages_to_analyze} pages...")
        for page_num in range(5, pages_to_analyze + 1):
            image_b64 = get_page_image_data(pdf_bytes, page_num)
            if not image_b64:
                logger.info(f"ℹ️ Reached end of document at page {page_num}")
                break
            
            extracted_data = extract_structured_page_data(image_b64, page_num)
            
            if extracted_data:
                all_page_results.append({
                    'page_num': page_num,
                    'image_b64': image_b64,
                    'extracted_data': extracted_data
                })
        
        logger.info(f"✅ Successfully analyzed {len(all_page_results)} pages")
        
        # Generate comprehensive analysis
        analysis_summary = generate_document_analysis(
            [page['extracted_data'] for page in all_page_results],
            len(all_page_results)
        )
        
        logger.info("✅ AI analysis completed successfully")
        logger.info("="*70)
        return JSONResponse(content={
            "success": True,
            "data": {
                "pages_analyzed": len(all_page_results),
                "extracted_pages": all_page_results,
                "analysis_summary": analysis_summary
            }
        })
        
    except Exception as e:
        logger.error(f"❌ AI analysis failed: {e}")
        logger.error("="*70)
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# =========================================================================
# Startup Event
# =========================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("="*70)
    logger.info("🚀 AI ANALYST PLATFORM API STARTING UP")
    logger.info("="*70)
    logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
    logger.info(f"🌐 Server: FastAPI")
    logger.info(f"📊 Version: 1.0.0")
    logger.info(f"🔧 Environment: Development")
    logger.info("="*70)
    logger.info("✅ All systems ready!")
    logger.info("📡 API Documentation: http://localhost:8000/docs")
    logger.info("="*70)

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("="*70)
    logger.info("🛑 AI ANALYST PLATFORM API SHUTTING DOWN")
    logger.info(f"📅 Timestamp: {datetime.now().isoformat()}")
    logger.info("="*70)

# =========================================================================
# Run Server
# =========================================================================

if __name__ == "__main__":
    
    # run this app for deploying so it keeps the end points without uvicorn
    logger.info("🎬 Starting server with uvicorn...")
    uvicorn.run(app,host="0.0.0.0",port=8000, log_level="info")
    