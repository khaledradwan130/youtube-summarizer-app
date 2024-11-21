import re
import json
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
import streamlit as st
import yt_dlp

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:v=|\/videos\/|embed\/|youtu.be\/|\/v\/|\/e\/|watch\?v%3D|watch\?feature=player_embedded&v=|%2Fvideos%2F|embed%\u200C\u200B2F|youtu.be%2F|%2Fv%2F)([^#\&\?\n]*)',
    ]
    
    if not url:
        return None
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Get transcript using YouTube Transcript API"""
    try:
        st.info("Attempting to retrieve transcript using YouTube Transcript API...")
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        st.success("Successfully retrieved transcript using YouTube Transcript API")
        return transcript
    except Exception as e:
        st.warning(f"YouTube Transcript API failed: {str(e)}")
        st.info("Attempting fallback method with yt-dlp...")
        return get_transcript_yt_dlp(video_id)

def get_transcript_yt_dlp(video_id):
    """Fallback method to get transcript using yt-dlp"""
    try:
        video_url = f'https://www.youtube.com/watch?v={video_id}'
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitlesformat': 'json3',
            'skip_download': True,
            'quiet': True
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if 'subtitles' in info and info['subtitles']:
                # Try to get manual subtitles first
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in info['subtitles']:
                        st.write(f"Found manual subtitles in {lang}")
                        return [{'text': caption['text']} for caption in info['subtitles'][lang][0]['fragments']]
            
            if 'automatic_captions' in info and info['automatic_captions']:
                # Fall back to automatic captions
                for lang in ['en', 'en-US', 'en-GB']:
                    if lang in info['automatic_captions']:
                        st.write(f"Found automatic captions in {lang}")
                        return [{'text': caption['text']} for caption in info['automatic_captions'][lang][0]['fragments']]
        
        st.error("No transcript found using yt-dlp")
        return None
    except Exception as e:
        st.error(f"Error with yt-dlp: {str(e)}")
        return None

def process_transcript(transcript):
    """Process transcript into a clean format"""
    if not transcript:
        return ""
    
    # Extract text from transcript entries and join with spaces
    text = ' '.join([entry.get('text', '') for entry in transcript])
    
    # Clean the text
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'\[[^\]]+\]', '', text)  # Remove metadata markers
    text = re.sub(r'\([^)]+\)', '', text)  # Remove parenthetical content
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    return text.strip()

def chunk_text(text, chunk_size):
    """Split text into chunks of approximately equal size"""
    # Clean and prepare the text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is shorter than chunk_size, return it as a single chunk
    if len(text) <= chunk_size:
        return [text]
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence_length = len(sentence)
        
        # Handle very long sentences
        if sentence_length > chunk_size:
            # If there's content in current_chunk, add it to chunks
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            # Split long sentence into smaller parts
            words = sentence.split()
            current_part = []
            current_part_length = 0
            
            for word in words:
                word_length = len(word) + 1  # +1 for space
                if current_part_length + word_length > chunk_size:
                    if current_part:
                        chunks.append(' '.join(current_part))
                        current_part = []
                        current_part_length = 0
                current_part.append(word)
                current_part_length += word_length
            
            if current_part:
                chunks.append(' '.join(current_part))
            continue
        
        # Check if adding the sentence would exceed chunk_size
        if current_length + sentence_length + 1 > chunk_size:
            # Add current chunk to chunks and start a new one
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_length += sentence_length + 1
    
    # Add any remaining content
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def ollama_completion(messages, model="artifish/llama3.2-uncensored:latest"):
    """Get completion from Ollama API"""
    try:
        response = requests.post('http://localhost:11434/api/chat',
                               json={
                                   "model": model,
                                   "messages": messages,
                                   "stream": False
                               })
        response.raise_for_status()
        return response.json()['message']['content']
    except Exception as e:
        st.error(f"Error calling Ollama API: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    st.set_page_config(page_title="YouTube Video Summarizer (Local)", layout="wide")
    
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
            video_url = st.text_input("Enter YouTube URL", key="video_url")
        
        # Reset chat if URL changes
        if video_url != st.session_state["current_url"]:
            st.session_state["messages"] = []
            st.session_state["current_url"] = video_url
            st.session_state["current_summary"] = ""
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
                help="Adjust this to control how detailed the summary should be. Lower values create more detailed summaries."
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
                        
                        # Get transcript
                        transcript = get_transcript(video_id)
                        if not transcript:
                            st.error("Could not fetch transcript")
                            return
                        
                        # Process transcript into clean text
                        transcript_text = process_transcript(transcript)
                        if not transcript_text:
                            st.error("Could not process transcript")
                            return
                        
                        # Show transcript preview
                        with st.expander("View transcript"):
                            st.text(transcript_text[:1000] + "..." if len(transcript_text) > 1000 else transcript_text)
                        
                        # Split text into chunks
                        chunks = chunk_text(transcript_text, chunk_size)
                        if not chunks:
                            st.error("Could not split transcript into chunks")
                            return
                        
                        st.write(f"Split transcript into {len(chunks)} chunks")
                        
                        # Process each chunk
                        summaries = []
                        for i, chunk in enumerate(chunks, 1):
                            with st.spinner(f"Processing chunk {i}/{len(chunks)}..."):
                                messages = [
                                    {"role": "system", "content": "You are a helpful AI assistant that creates concise and accurate summaries."},
                                    {"role": "user", "content": f"Please summarize this text chunk from a video transcript. Focus on the key points and maintain context:\n\n{chunk}"}
                                ]
                                
                                chunk_summary = ollama_completion(messages)
                                if chunk_summary:
                                    summaries.append(chunk_summary)
                                    with st.expander(f"Chunk {i} Summary"):
                                        st.write(chunk_summary)
                        
                        if summaries:
                            # Generate final summary
                            final_summary_prompt = "Based on these chunk summaries, create a coherent final summary of the video:\n\n" + "\n\n".join(summaries)
                            messages = [
                                {"role": "system", "content": "You are a helpful AI assistant that creates final video summaries from chunk summaries. Create a well-structured, coherent summary that flows naturally."},
                                {"role": "user", "content": final_summary_prompt}
                            ]
                            
                            final_summary = ollama_completion(messages)
                            if final_summary:
                                st.session_state["current_summary"] = final_summary
                                st.markdown("### Summary")
                                st.markdown(final_summary)
                
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
    
    # Display current summary outside the Generate Summary button block
    if "current_summary" in st.session_state and st.session_state["current_summary"]:
        st.markdown("### Summary")
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
                                {"role": "system", "content": f"""You are a helpful AI assistant that answers questions about a video based STRICTLY on its summary.
                                You must ONLY use information from this summary to answer questions. If the information needed to answer the question
                                is not in the summary, say so clearly. DO NOT make up or infer information that is not explicitly stated in the summary.
                                
                                Here is the video summary to reference:
                                {st.session_state["current_summary"]}"""},
                                {"role": "user", "content": "What information can I find in this summary?"},
                                {"role": "assistant", "content": "I can help you with questions about the specific content mentioned in the summary above. I'll only reference information that's explicitly stated in it. If you ask about something not covered in the summary, I'll let you know that I don't have that information."},
                                {"role": "user", "content": prompt}
                            ]
                            
                            response = ollama_completion(messages)
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

if __name__ == "__main__":
    main()
