import streamlit as st
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
from pytube import YouTube
import re
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import requests
from bs4 import BeautifulSoup
import yt_dlp

# Load environment variables
load_dotenv()

def openrouter_completion(messages, model="meta-llama/llama-3.2-3b-instruct:free"):
    try:
        # Get API key from environment variable or Streamlit secrets
        api_key = os.getenv('OPENROUTER_API_KEY') or st.secrets['OPENROUTER_API_KEY']
        
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://youtube-summarizer.streamlit.app",
                "X-Title": "YouTube Summarizer by AI Afterdark",
            },
            model=model,
            messages=messages
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"OpenRouter API Error: {str(e)}")
        return None

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: None
)
def openrouter_completion_with_retry(messages, model="meta-llama/llama-3.2-3b-instruct:free"):
    return openrouter_completion(messages, model)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def get_transcript_with_retry(func, *args, **kwargs):
    return func(*args, **kwargs)

def get_transcript_yt_dlp(video_id):
    """Get transcript using yt-dlp"""
    try:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': True,
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            if 'subtitles' in info and info['subtitles']:
                # Process subtitles
                subtitles = info['subtitles'].get('en', [])
                if subtitles:
                    return [{'text': sub['text'], 'start': 0, 'duration': 0} for sub in subtitles]
            elif 'automatic_captions' in info and info['automatic_captions']:
                # Process automatic captions
                auto_caps = info['automatic_captions'].get('en', [])
                if auto_caps:
                    return [{'text': cap['text'], 'start': 0, 'duration': 0} for cap in auto_caps]
    except Exception as e:
        return None
    return None

def get_transcript(video_id):
    """Get transcript with multiple fallback options"""
    transcript = None
    error_messages = []
    
    # Method 1: YouTube Transcript API with retry
    try:
        transcript = get_transcript_with_retry(
            YouTubeTranscriptApi.get_transcript,
            video_id,
            languages=['en']
        )
    except Exception as e:
        error_messages.append(f"Method 1 failed: {str(e)}")

    # Method 2: Pytube
    if not transcript:
        try:
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            if yt.captions:
                caption = None
                if 'en' in yt.captions:
                    caption = yt.captions['en']
                elif 'a.en' in yt.captions:
                    caption = yt.captions['a.en']
                elif yt.captions:
                    caption = list(yt.captions.values())[0]
                
                if caption:
                    transcript_text = caption.generate_srt_captions()
                    lines = transcript_text.split('\n\n')
                    transcript = []
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = line.split('\n')
                        if len(parts) >= 3:
                            text = ' '.join(parts[2:])
                            transcript.append({
                                'text': text,
                                'start': 0,
                                'duration': 0
                            })
        except Exception as e:
            error_messages.append(f"Method 2 failed: {str(e)}")

    # Method 3: yt-dlp
    if not transcript:
        try:
            transcript = get_transcript_yt_dlp(video_id)
        except Exception as e:
            error_messages.append(f"Method 3 failed: {str(e)}")

    if not transcript:
        st.error("Could not retrieve transcript. Please try another video.")
        st.error("Technical details:")
        for msg in error_messages:
            st.error(msg)
        return None

    return transcript

def process_chunks_with_rate_limit(chunks, system_prompt):
    """Process chunks with rate limit handling"""
    summaries = []
    total_chunks = len(chunks)
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, chunk in enumerate(chunks, 1):
        status_text.text(f"Processing chunk {i} of {total_chunks}...")
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Please provide a detailed summary of this video transcript part: {chunk}"}
        ]
        
        summary = openrouter_completion_with_retry(messages)
        if summary:
            summaries.append(summary)
        
        # Update progress bar
        progress_bar.progress(i / total_chunks)
        
        # Add a small delay between chunks to avoid rate limits
        if i < total_chunks:
            time.sleep(2)  # 2-second delay between chunks
    
    progress_bar.empty()
    status_text.empty()
    return summaries

def chunk_text(text, chunk_size):
    """Split text into chunks of approximately equal size"""
    words = text.split()
    chunks = []
    current_chunk = []
    current_size = 0
    
    for word in words:
        current_size += len(word) + 1  # +1 for space
        if current_size > chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_size = len(word)
        else:
            current_chunk.append(word)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def extract_video_id(url):
    """Extract YouTube video ID from URL"""
    pattern = r'(?:v=|\/)([0-9A-Za-z_-]{11}).*'
    match = re.search(pattern, url)
    return match.group(1) if match else None

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

    # Main content area
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        st.markdown("### Video Summary")
        # Constrain the input width to match column
        with st.container():
            video_url = st.text_input("Enter YouTube Video URL", placeholder="https://www.youtube.com/watch?v=...")
        
        # Reset chat if URL changes
        if video_url != st.session_state["current_url"]:
            st.session_state["messages"] = []
            st.session_state["current_url"] = video_url
        
        if video_url:
            video_id = extract_video_id(video_url)
            if video_id:
                # Create a container for video that fills the column width
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
        
        # Chunking control in a container to match width
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
                    with st.spinner("Fetching video transcript..."):
                        video_id = extract_video_id(video_url)
                        if not video_id:
                            st.error("Invalid YouTube URL")
                            return
                        
                        transcript = get_transcript(video_id)
                        if not transcript:
                            return
                        
                        text = " ".join([entry["text"] for entry in transcript])
                        if not text.strip():
                            st.error("Retrieved transcript is empty")
                            return
                        
                        # Split text into chunks
                        chunks = chunk_text(text, chunk_size)
                        
                    with st.spinner(f"Generating summary from {len(chunks)} chunks..."):
                        # Process chunks with rate limit handling
                        chunk_summaries = process_chunks_with_rate_limit(
                            chunks,
                            """You are a professional content summarizer. Create a detailed, 
                            well-structured summary of this part of the video transcript. Focus on the main points, key insights, 
                            and important details."""
                        )
                        
                        # Generate final summary if we have chunk summaries
                        if chunk_summaries:
                            final_summary_prompt = "\n\n".join(chunk_summaries)
                            messages = [
                                {"role": "system", "content": """You are a professional content summarizer. Create a cohesive, 
                                well-structured final summary from these chunk summaries. Format the output in markdown with 
                                clear sections. Focus on maintaining narrative flow and connecting ideas across chunks."""},
                                {"role": "user", "content": f"Create a final summary from these chunk summaries:\n\n{final_summary_prompt}"}
                            ]
                            
                            final_summary = openrouter_completion_with_retry(messages)
                            if final_summary:
                                st.session_state["current_summary"] = final_summary
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                
        # Display summary in a container that fills the column width
        if "current_summary" in st.session_state and st.session_state["current_summary"]:
            with st.container():
                st.markdown("### Current Summary")
                st.markdown(st.session_state["current_summary"])

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
                                {"role": "system", "content": f"""You are a helpful AI assistant that answers questions about a video based on its summary. 
                                Here's the video summary to reference:\n\n{st.session_state["current_summary"]}"""},
                                *st.session_state.messages,
                            ]
                            response = openrouter_completion_with_retry(messages)
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.markdown(response)
        else:
            st.info("Generate a video summary first to start chatting!")

    # Reset chat button
    if st.button("Reset Chat"):
        st.session_state["messages"] = []
        st.session_state["current_summary"] = ""

if __name__ == "__main__":
    main()
