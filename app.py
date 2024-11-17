import streamlit as st
import os
from openai import OpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import re
from dotenv import load_dotenv
import time
from tenacity import retry, stop_after_attempt, wait_exponential

# Load environment variables
load_dotenv()

def openrouter_completion(messages, model="meta-llama/llama-3.2-3b-instruct"):
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
def openrouter_completion_with_retry(messages, model="meta-llama/llama-3.2-3b-instruct"):
    return openrouter_completion(messages, model)

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
                        transcript = YouTubeTranscriptApi.get_transcript(video_id)
                        text = " ".join([entry["text"] for entry in transcript])
                        
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
        if "current_summary" in st.session_state:
            chat_container = st.container()
            with chat_container:
                # Display messages in chronological order
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Chat input at the bottom
                if prompt := st.chat_input("Ask anything about the video..."):
                    # Add user message and display immediately
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Generate assistant response
                    messages = [
                        {"role": "system", "content": """You are a knowledgeable AI assistant that helps users understand video content. 
                        Base your answers only on the information provided in the summary."""},
                        {"role": "user", "content": f"Here is the video summary: {st.session_state['current_summary']}"},
                        {"role": "user", "content": f"Please answer this question about the video: {prompt}"}
                    ]
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        assistant_response = openrouter_completion_with_retry(messages)
                        if assistant_response:
                            st.session_state.messages.append({"role": "assistant", "content": assistant_response})
                            st.markdown(assistant_response)
                            st.rerun()
        else:
            st.info("Generate a video summary first to start chatting!")

    # Reset chat button
    if st.button("Reset Chat"):
        st.session_state["messages"] = []
        st.session_state["current_summary"] = ""

if __name__ == "__main__":
    main()
