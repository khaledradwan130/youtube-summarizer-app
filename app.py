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
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={
                "model": "meta-llama/llama-3.2-3b-instruct:free",
                "messages": messages,
                "timeout": 30
            },
            timeout=30
        )
        
        # Debug log
        st.write("Received response from OpenRouter")
        
        response.raise_for_status()
        result = response.json()
        
        if not result.get('choices', [{}])[0].get('message', {}).get('content'):
            st.error("Empty response from OpenRouter")
            return None
            
        return result['choices'][0]['message']['content']
    except Timeout:
        st.error("OpenRouter request timed out. Please try again.")
        return None
    except RequestException as e:
        st.error(f"OpenRouter API Error: {str(e)}")
        return None

def get_video_info(video_id):
    """Get video information and transcript using yt-dlp"""
    ydl_opts = {
        'writesubtitles': True,
        'writeautomaticsub': True,
        'subtitleslangs': ['en'],
        'skip_download': True,
        'quiet': True,
        'no_warnings': True,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            
            # Debug log
            st.write("Fetching video info...")
            
            info = ydl.extract_info(video_url, download=False)
            
            # Get video details
            video_details = {
                'title': info.get('title', ''),
                'description': info.get('description', ''),
                'duration': info.get('duration', 0),
                'view_count': info.get('view_count', 0),
                'uploader': info.get('uploader', ''),
            }
            
            # Debug log
            st.write("Fetching transcript...")
            
            # Get transcript
            transcript = []
            if 'subtitles' in info and info['subtitles'].get('en'):
                # Try to get manual subtitles first
                transcript_url = info['subtitles']['en'][0]['url']
                response = requests.get(transcript_url)
                if response.status_code == 200:
                    transcript = process_transcript(response.text)
            
            if not transcript and 'automatic_captions' in info and info['automatic_captions'].get('en'):
                # Fall back to automatic captions if no manual subtitles
                transcript_url = info['automatic_captions']['en'][0]['url']
                response = requests.get(transcript_url)
                if response.status_code == 200:
                    transcript = process_transcript(response.text)
            
            transcript_text = ' '.join(transcript) if transcript else ''
            
            # Verify transcript content
            if not transcript_text.strip():
                st.error("No transcript content found")
                return video_details, None
                
            # Debug log
            st.write(f"Transcript length: {len(transcript_text)} characters")
            
            return video_details, transcript_text
    except Exception as e:
        st.error(f"Error fetching video information: {str(e)}")
        return None, None

def process_transcript(transcript_text):
    """Process the transcript text into a clean format"""
    # Remove XML/HTML tags and metadata
    transcript_text = re.sub(r'<[^>]+>', '', transcript_text)
    transcript_text = re.sub(r'\{[^}]+\}', '', transcript_text)
    
    # Split into lines and process
    lines = transcript_text.split('\n')
    transcript = []
    current_text = ''
    
    for line in lines:
        # Skip empty lines, timing info, and metadata
        if (line.strip() and 
            not line[0].isdigit() and 
            '-->' not in line and 
            'WEBVTT' not in line and
            'Kind:' not in line and
            'Language:' not in line and
            'Style:' not in line):
            # Clean the line of any remaining metadata-like content
            cleaned_line = re.sub(r'\[[^\]]+\]', '', line.strip())
            cleaned_line = re.sub(r'\([^)]+\)', '', cleaned_line)
            if cleaned_line:
                current_text += ' ' + cleaned_line
        elif current_text:
            transcript.append(current_text.strip())
            current_text = ''
    
    if current_text:
        transcript.append(current_text.strip())
    
    return transcript

def chunk_text(text, chunk_size):
    """Split text into chunks of approximately equal size"""
    # Calculate a more appropriate chunk size based on the input parameter
    # This will help create fewer, larger chunks when chunk_size is increased
    effective_chunk_size = chunk_size * 2  # Double the chunk size to reduce number of chunks
    
    # Split into sentences first to maintain context
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        if current_size + sentence_size > effective_chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            current_chunk.append(sentence)
            current_size += sentence_size
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def process_chunks_with_rate_limit(chunks, system_prompt):
    """Process chunks with rate limit handling and progress tracking"""
    summaries = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        for i, chunk in enumerate(chunks):
            status_text.text(f"Processing chunk {i+1}/{len(chunks)}")
            progress_bar.progress((i) / len(chunks))
            
            # Show current chunk being processed
            with st.expander(f"Processing chunk {i+1}/{len(chunks)}"):
                st.text(chunk[:200] + "...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": chunk}
            ]
            
            summary = openrouter_completion(messages)
            if not summary:
                st.error(f"Failed to process chunk {i+1}")
                return None
            
            summaries.append(summary)
            # Show chunk summary
            with st.expander(f"Chunk {i+1} Summary"):
                st.markdown(summary)
            
            time.sleep(1)  # Rate limiting
        
        progress_bar.progress(1.0)
        status_text.text("All chunks processed successfully!")
        return summaries
    except Exception as e:
        st.error(f"Error processing chunks: {str(e)}")
        return None
    finally:
        progress_bar.empty()
        status_text.empty()

def extract_video_id(url):
    """Extract YouTube video ID from various URL formats"""
    patterns = [
        r'(?:v=|\/)([0-9A-Za-z_-]{11}).*',
        r'(?:youtu\.be\/|youtube\.com\/shorts\/)([0-9A-Za-z_-]{11})',
        r'(?:embed\/)([0-9A-Za-z_-]{11})',
        r'^([0-9A-Za-z_-]{11})$'
    ]
    
    if not url:
        return None
        
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
                        
                        video_details, transcript_text = get_video_info(video_id)
                        if not video_details:
                            st.error("Could not fetch video information")
                            return
                            
                        if not transcript_text:
                            st.error("Could not fetch transcript")
                            return
                        
                        st.session_state["video_details"] = video_details
                        
                        # Show transcript preview
                        with st.expander("View transcript"):
                            st.text(transcript_text[:1000] + "...")
                        
                        # Split text into chunks
                        chunks = chunk_text(transcript_text, chunk_size)
                        if not chunks:
                            st.error("Could not split transcript into chunks")
                            return
                            
                        st.write(f"Split transcript into {len(chunks)} chunks")
                        
                    with st.spinner(f"Generating summary from {len(chunks)} chunks..."):
                        # Process chunks with rate limit handling
                        chunk_summaries = process_chunks_with_rate_limit(
                            chunks,
                            """You are a professional content summarizer. Your task is to create a clear, concise summary of this video transcript section.
                            Focus on:
                            1. The main topics and key points discussed
                            2. Important facts, figures, and statements
                            3. Any conclusions or significant insights
                            
                            Ignore any technical artifacts or formatting. Present the information in a well-structured, easy-to-read format."""
                        )
                        
                        if not chunk_summaries:
                            st.error("Failed to generate chunk summaries")
                            return
                        
                        # Generate final summary if we have chunk summaries
                        final_summary_prompt = "\n\n".join(chunk_summaries)
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
                        if not final_summary:
                            st.error("Failed to generate final summary")
                            return
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                
        # Display video details and summary
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
                st.markdown("### Summary")
                st.markdown(st.session_state["current_summary"])

    with col2:
        st.markdown("### Chat with Video Content")
        if st.session_state.get("current_summary"):
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
                            response = openrouter_completion(messages)
                            if not response:
                                st.error("Failed to generate response")
                                return
                            st.session_state.messages.append({"role": "assistant", "content": response})
                            st.markdown(response)
        else:
            st.info("Generate a video summary first to start chatting!")

    # Reset chat button
    if st.button("Reset Chat"):
        st.session_state["messages"] = []
        st.session_state["current_summary"] = ""
        st.session_state["video_details"] = None

if __name__ == "__main__":
    main()
