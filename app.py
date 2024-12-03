import streamlit as st
import re
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
import yt_dlp
import json
from requests.exceptions import Timeout, RequestException
import os
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# Load environment variables
load_dotenv()

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
def openrouter_completion(messages):
    """Send request to OpenRouter API using meta-llama/llama-3.2-3b-instruct:free model"""
    try:
        # Debug log
        st.write(f"Sending request to OpenRouter using meta-llama/llama-3.2-3b-instruct:free...")
        
        headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "HTTP-Referer": "https://github.com/aiafterdark/youtube-summarizer-app",
            "X-Title": "YouTube Summarizer App"
        }
        
        payload = {
            "model": "meta-llama/llama-3.2-3b-instruct:free",
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,  # Reduced to ensure we stay within limits
            "top_p": 0.8,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3,
            "stream": False
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30  # Reduced timeout since we're using smaller chunks
        )
        
        # Debug log
        st.write("Received response from OpenRouter")
        
        if response.status_code != 200:
            error_msg = f"OpenRouter API returned status code {response.status_code}"
            try:
                error_data = response.json()
                if 'error' in error_data:
                    error_msg += f": {error_data['error']}"
            except:
                pass
            st.error(error_msg)
            return None
            
        result = response.json()
        
        # Debug the API response
        if 'choices' not in result or not result['choices']:
            st.error(f"Unexpected API response format: {json.dumps(result, indent=2)}")
            return None
            
        content = result.get('choices', [{}])[0].get('message', {}).get('content')
        if not content:
            st.error("Empty content in OpenRouter response")
            return None
            
        return content.strip()
        
    except Timeout:
        st.error("OpenRouter request timed out. Please try again.")
        return None
    except RequestException as e:
        st.error(f"OpenRouter API Error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Unexpected error while calling OpenRouter: {str(e)}")
        return None

def get_transcript_yt_dlp(video_id):
    """Enhanced transcript retrieval using yt-dlp"""
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        # Initial configuration for video info
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'srv1',  # Prefer srv1 format for better compatibility
            'subtitleslangs': ['en'],
            'skip_download': True,
            'format': 'best'
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            try:
                info = ydl.extract_info(url, download=False)
                
                # Store video details in session state
                video_details = {
                    'title': info.get('title', ''),
                    'description': info.get('description', ''),
                    'duration': info.get('duration', 0),
                    'view_count': info.get('view_count', 0),
                    'uploader': info.get('uploader', ''),
                }
                st.session_state["video_details"] = video_details

                # First try manual subtitles
                if info.get('subtitles') and info['subtitles'].get('en'):
                    st.info("Found manual English subtitles")
                    transcript_list = []
                    for entry in info['subtitles']['en']:
                        if isinstance(entry, dict) and 'ext' in entry:
                            if entry['ext'] == 'srv1':
                                transcript_list.append({
                                    'text': entry.get('text', ''),
                                    'start': float(entry.get('start', 0)),
                                    'duration': float(entry.get('duration', 0))
                                })
                    if transcript_list:
                        return transcript_list

                # If no manual subtitles, try automatic captions
                if info.get('automatic_captions') and info['automatic_captions'].get('en'):
                    st.info("Found automatic English captions")
                    auto_caps = info['automatic_captions']['en']
                    transcript_list = []
                    
                    for entry in auto_caps:
                        if isinstance(entry, dict):
                            transcript_list.append({
                                'text': entry.get('text', ''),
                                'start': float(entry.get('start', 0)),
                                'duration': float(entry.get('duration', 0))
                            })
                    
                    if transcript_list:
                        return transcript_list
                    
                st.warning("No English subtitles or automatic captions found")
                return None

            except Exception as e:
                st.error(f"Error extracting video info: {str(e)}")
                return None
                
    except Exception as e:
        st.error(f"yt-dlp error: {str(e)}")
        return None
    
    return None

def get_transcript(video_id):
    """Get transcript with enhanced error handling and logging"""
    transcript = None
    error_messages = []
    
    # Method 1: YouTube Transcript API (primary method)
    try:
        st.info("Attempting to retrieve transcript using YouTube Transcript API...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        if transcript:
            st.success("Successfully retrieved transcript using YouTube Transcript API")
            return transcript
    except (NoTranscriptFound, TranscriptsDisabled) as e:
        error_msg = f"YouTube Transcript API failed: {str(e)}"
        st.warning(error_msg)
        error_messages.append(error_msg)
    except Exception as e:
        error_msg = f"YouTube Transcript API failed with unexpected error: {str(e)}"
        st.warning(error_msg)
        error_messages.append(error_msg)

    # Method 2: yt-dlp (fallback)
    try:
        st.info("Attempting to retrieve transcript using yt-dlp...")
        transcript = get_transcript_yt_dlp(video_id)
        if transcript:
            st.success("Successfully retrieved transcript using yt-dlp")
            return transcript
        else:
            error_msg = "yt-dlp could not find any transcripts or captions"
            error_messages.append(error_msg)
    except Exception as e:
        error_msg = f"yt-dlp method failed: {str(e)}"
        st.warning(error_msg)
        error_messages.append(error_msg)

    # If all methods fail
    error_message = "\n".join(error_messages) if error_messages else "No available transcripts found"
    st.error("Could not retrieve transcript. Please try another video.")
    st.error(f"Details:\n{error_message}")
    return None

def process_transcript(transcript):
    """Process transcript into a clean format"""
    if not transcript:
        return ""
    
    try:
        # Extract text from transcript entries and join with spaces
        text = ' '.join([str(entry.get('text', '')).strip() for entry in transcript if entry.get('text')])
        
        # Clean the text
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
        text = re.sub(r'\[[^\]]+\]', '', text)  # Remove metadata markers
        text = re.sub(r'\([^)]+\)', '', text)  # Remove parenthetical content
        text = re.sub(r'♪.*?♪', '', text)  # Remove musical note sections
        text = re.sub(r'\s*\n\s*', ' ', text)  # Replace newlines with spaces
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Add periods to help with sentence splitting if missing
        text = re.sub(r'([a-zA-Z0-9])\s+([A-Z])', r'\1. \2', text)
        
        return text.strip()
    except Exception as e:
        st.error(f"Error processing transcript: {str(e)}")
        return ""

def chunk_text(text, chunk_size):
    """Split text into chunks of approximately equal size"""
    try:
        if not text:
            st.warning("No text to process")
            return []
            
        # Clean and prepare the text
        text = re.sub(r'\s+', ' ', text).strip()
        
        # If text is shorter than chunk size, return it as a single chunk
        if len(text) <= chunk_size:
            return [text]
        
        # Split into sentences (considering multiple punctuation marks)
        sentences = []
        current_sentence = []
        
        # Split text into words first
        words = text.split()
        
        for word in words:
            current_sentence.append(word)
            # Check if word ends with sentence-ending punctuation
            if word and word[-1] in '.!?':
                sentences.append(' '.join(current_sentence))
                current_sentence = []
        
        # Add any remaining words as a sentence
        if current_sentence:
            sentences.append(' '.join(current_sentence))
        
        # Remove empty sentences and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            # If a single sentence is longer than chunk_size, split it by words
            if sentence_size > chunk_size:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_size = 0
                
                # Split long sentence into smaller parts
                words = sentence.split()
                temp_chunk = []
                temp_size = 0
                
                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if temp_size + word_size > chunk_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_size = word_size
                    else:
                        temp_chunk.append(word)
                        temp_size += word_size
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            
            # If adding this sentence would exceed chunk_size, start a new chunk
            elif current_size + sentence_size + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            
            # Add sentence to current chunk
            else:
                current_chunk.append(sentence)
                current_size += sentence_size + 1
        
        # Add any remaining sentences
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        # Final cleanup of chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        if not chunks:
            st.warning("No chunks were generated from the text")
            return []
            
        return chunks
        
    except Exception as e:
        st.error(f"Error chunking text: {str(e)}")
        return []

def process_chunks_with_rate_limit(chunks, system_prompt):
    """Process chunks with rate limit handling and progress tracking"""
    summaries = []
    total_chunks = len(chunks)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks, 1):
        status_text.text(f"Processing chunk {i} of {total_chunks}...")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please provide a detailed summary of this video transcript part:\n\n{chunk}"}
        ]
        
        # Show current chunk being processed
        with st.container():
            st.text(f"Processing chunk {i}/{total_chunks}: {chunk[:100]}...")
        
        summary = openrouter_completion(messages)
        if summary:
            summaries.append(summary)
            # Show chunk summary
            with st.container():
                st.markdown(summary)
        else:
            st.error(f"Failed to process chunk {i}")
            continue
        
        progress_bar.progress(i / total_chunks)
        
        # Add a small delay between chunks
        if i < total_chunks:
            time.sleep(2)
    
    progress_bar.empty()
    status_text.empty()
    
    return summaries

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11})(?:[?&]|$)',  # Standard and parameterized URLs
        r'(?:youtu\.be\/|youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})(?:[?&]|$)',  # Short URLs and shorts with parameters
        r'(?:embed\/)([0-9A-Za-z_-]{11})',  # Embed URLs
        r'^([0-9A-Za-z_-]{11})$'  # Direct video IDs
    ]
    
    if not url:
        return None
    
    # Remove any whitespace and handle mobile URLs
    url = url.strip()
    url = url.replace('http://youtu.be/', 'https://youtu.be/')
    url = url.replace('http://youtube.com/', 'https://youtube.com/')
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def main():
    st.set_page_config(
        page_title="YouTube Video Summarizer",
        page_icon="🎥",
        layout="wide"
    )

    st.markdown("""
        <h1 style='margin-bottom: 0;'>
            YouTube Video Summarizer
            <span style='font-size: 1rem; color: #19bfb7; margin-left: 10px;'>by AI AfterDark</span>
        </h1>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("Get AI-powered summaries of any YouTube video and chat with the content")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "current_url" not in st.session_state:
        st.session_state["current_url"] = ""
    if "current_summary" not in st.session_state:
        st.session_state["current_summary"] = ""
    if "video_details" not in st.session_state:
        st.session_state["video_details"] = None
    if "chunk_summaries" not in st.session_state:
        st.session_state["chunk_summaries"] = []

    # Main content area
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.markdown("### Video Summary")
        with st.container():
            video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        
        # Reset chat if URL changes
        if video_url != st.session_state["current_url"]:
            st.session_state["messages"] = []
            st.session_state["current_url"] = video_url
            st.session_state["video_details"] = None
            st.session_state["chunk_summaries"] = []
        
        if video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                video_container = st.container()
                with video_container:
                    st.markdown(
                        f'<div style="width: 100%;">'
                        f'<div style="position: relative; padding-bottom: 56.25%; height: 0; overflow: hidden;">'
                        f'<iframe src="https://www.youtube.com/embed/{video_id}" '
                        f'style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; border: 0;" '
                        f'allowfullscreen></iframe>'
                        f'</div></div>',
                        unsafe_allow_html=True
                    )
        
        with st.container():
            chunk_size = st.slider(
                "Summary Detail Level",
                min_value=1000,
                max_value=10000,
                value=4000,
                step=1000,
                help="Adjust this to control how detailed the summary should be. Lower values create more detailed summaries. (Long Podcast (1hr+) should be 7000+)"
            )

        if st.button("Generate Summary", type="primary"):
            if not video_url:
                st.error("Please enter a valid YouTube URL")
            else:
                try:
                    with st.spinner("Fetching video information..."):
                        video_id = extract_video_id(video_url)
                        if not video_id:
                            st.error("Invalid YouTube URL")
                            return
                        
                        transcript = get_transcript(video_id)
                        if not transcript:
                            st.error("Could not fetch transcript")
                            return
                        
                        transcript_text = process_transcript(transcript)
                        
                        # Create processing status container
                        with st.container():
                            st.markdown("### Processing Details")
                            
                            # Create tabs for different sections
                            transcript_tab, processing_tab = st.tabs(["Transcript", "Processing Status"])
                            
                            with transcript_tab:
                                st.markdown("#### Transcript Preview")
                                st.text(transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text)
                            
                            with processing_tab:
                                # Split text into chunks
                                chunks = chunk_text(transcript_text, chunk_size)
                                if not chunks:
                                    st.error("Could not split transcript into chunks")
                                    return
                                
                                st.write(f"Processing {len(chunks)} chunks...")
                                
                                # Create progress tracking
                                progress_bar = st.progress(0)
                                status_text = st.empty()
                                chunk_status = st.empty()
                                
                                # Process chunks with rate limit handling
                                chunk_summaries = []
                                for i, chunk in enumerate(chunks, 1):
                                    status_text.text(f"Processing chunk {i}/{len(chunks)}...")
                                    
                                    messages = [
                                        {"role": "system", "content": """You are a professional content summarizer. Your task is to create a clear, concise summary of this video transcript section.
                                        Focus on:
                                        1. The main topics and key points discussed
                                        2. Important facts, figures, and statements
                                        3. Any conclusions or significant insights
                                        
                                        Ignore any technical artifacts or formatting. Present the information in a well-structured, easy-to-read format."""},
                                        {"role": "user", "content": f"Please provide a detailed summary of this video transcript part:\n\n{chunk}"}
                                    ]
                                    
                                    chunk_status.text(f"Current chunk preview: {chunk[:100]}...")
                                    
                                    summary = openrouter_completion(messages)
                                    if summary:
                                        chunk_summaries.append(summary)
                                        st.markdown(f"✓ Chunk {i} processed")
                                    else:
                                        st.error(f"Failed to process chunk {i}")
                                        continue
                                    
                                    progress_bar.progress(i / len(chunks))
                                    
                                    # Add a small delay between chunks
                                    if i < len(chunks):
                                        time.sleep(2)
                                
                                progress_bar.empty()
                                status_text.empty()
                                chunk_status.empty()
                                
                                # Store chunk summaries in session state
                                st.session_state["chunk_summaries"] = chunk_summaries
                                
                                # Display chunk summaries in a table
                                if chunk_summaries:
                                    st.markdown("#### Individual Chunk Summaries")
                                    for i, summary in enumerate(chunk_summaries, 1):
                                        with st.container():
                                            st.markdown(f"**Chunk {i}**")
                                            st.markdown(summary)
                                            st.markdown("---")
                        
                        # Generate final summary if we have chunk summaries
                        st.write("Generating final summary...")
                        final_summary_prompt = f"""Here are the summaries of different parts of the video. Please create a cohesive final summary:

{chr(10).join(f'Part {i+1}:{chr(10)}{summary}{chr(10)}---' for i, summary in enumerate(chunk_summaries))}"""
                        
                        messages = [
                            {"role": "system", "content": """You are a professional content summarizer. Create a comprehensive summary of this video by combining the key points from all sections.
                            Your summary should:
                            1. Start with a brief overview of the video's main topic
                            2. Present the key points in a logical, flowing narrative
                            3. Include important details, quotes, or statistics
                            4. End with the main conclusions or takeaways
                            
                            Format the summary with clear sections and bullet points where appropriate."""},
                            {"role": "user", "content": final_summary_prompt}
                        ]
                        
                        final_summary = openrouter_completion(messages)
                        if final_summary:
                            st.session_state["current_summary"] = final_summary
                            st.markdown("### Summary")
                            st.markdown(final_summary)
                        else:
                            st.error("Failed to generate final summary")
                            return
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                
        # Display video details 
        if st.session_state.get("video_details"):
            with st.container():
                st.markdown("### Video Details")
                details = st.session_state["video_details"]
                st.markdown(f"""
                **Title:** {details['title']}
                **Uploader:** {details['uploader']}
                **Duration:** {details['duration']} seconds
                **Views:** {details['view_count']:,}
                """)
        
        if st.session_state.get("current_summary"):
            with st.container():
                pass

    with col2:
        st.markdown("### Chat with Video Content")
        if "current_summary" in st.session_state and st.session_state["current_summary"]:
            chat_container = st.container()
            with chat_container:
                # Display chat messages
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

                # Chat input
                if prompt := st.chat_input("Ask anything about the video..."):
                    # Add user message
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Generate assistant response
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            messages = [
                                {"role": "system", "content": f"""You are a helpful AI assistant that answers questions about a video based STRICTLY on its summary.
                                You must ONLY use information from this summary to answer questions. If the information needed to answer the question
                                is not in the summary, say so clearly. DO NOT make up or infer information that is not explicitly stated in the summary.
                                
                                Here is the video summary to reference:
                                {st.session_state["current_summary"]}"""},
                                {"role": "user", "content": "What information can I find in this summary?"},
                                {"role": "assistant", "content": "I can help you with questions about the specific content mentioned in the summary above. I'll only reference information that's explicitly stated in it. If you ask about something not covered in the summary, I'll let you know that I don't have that information."},
                                {"role": "user", "content": prompt}
                            ]
                            
                            response = openrouter_completion(messages)
                            if response:
                                st.session_state.messages.append({"role": "assistant", "content": response})
                                st.markdown(response)
                            else:
                                st.error("Failed to generate response")
        else:
            st.info("Generate a video summary first to start chatting!")

    # Reset chat button
    if st.button("Reset Chat"):
        st.session_state["messages"] = []
        st.session_state["current_summary"] = ""
        st.session_state["video_details"] = None
        st.session_state["chunk_summaries"] = []

if __name__ == "__main__":
    main()
