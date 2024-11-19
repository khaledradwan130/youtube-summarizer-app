# YouTube Video Summarizer
Created by AI Afterdark - Building Innovation with AI at Night
An AI-powered application that generates summaries of YouTube videos and enables interactive conversations about their content. This tool supports both cloud deployment (using OpenRouter) and local deployment (using Ollama).
<img src="img/YoutubeVideoSummarizer.gif" alt="Demo" autoplay loop>

## Live Demo
Experience the YouTube Video Summarizer in action: [https://aiafterdark-youtube-summarizer.streamlit.app/](https://aiafterdark-youtube-summarizer.streamlit.app/)

## Features
- Robust YouTube video transcript extraction with multiple fallback methods:
  - YouTube Transcript API (primary)
  - Pytube captions
  - yt-dlp caption extraction
- AI-powered content summarization
- Interactive Q&A about video content
- Support for both cloud and local AI models
- Adjustable summary detail levels
- Clean, responsive UI
- Comprehensive error handling and reporting

## Prerequisites
- Python 3.11+
- pip (Python package manager)
- Git
- Ollama (optional, for local deployment)

## Quick Start
### 1. Clone the Repository
```bash
git clone https://github.com/AIAfterDark/youtube-summarizer-app.git
cd youtube-summarizer-app
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
#### Cloud Deployment (OpenRouter)
1. Create a `.env` file in the root directory
2. Add your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_api_key_here
```

#### Local Deployment (Ollama)
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull your preferred model:
```bash
ollama pull llama2
```

### 5. Run the Application
```bash
streamlit run app.py
```

## Using Local Models with Ollama
To use Ollama locally, modify the `openrouter_completion` function in `app.py`:
```python
import requests
def ollama_completion(messages, model="llama2"):
    try:
        response = requests.post(
            'http://localhost:11434/api/chat',
            json={
                "model": model,
                "messages": messages
            }
        )
        return response.json()['message']['content']
    except Exception as e:
        st.error(f"Ollama API Error: {str(e)}")
        return None
# Replace OpenRouter function calls with Ollama
def openrouter_completion(messages, model="llama2"):
    return ollama_completion(messages, model)
```

## Configuration
### Summary Detail Level
- Short videos (<30 mins): 4000
- Long content (1hr+): 7000+

### Available Models
#### OpenRouter Models:
- meta-llama/llama-2-13b-chat
- anthropic/claude-2
- openai/gpt-3.5-turbo

#### Ollama Models:
- llama2
- codellama
- mistral
- neural-chat

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is open source and available under the MIT License. Created by AI Afterdark - feel free to use and modify, but please credit us!

## Credits
Created by Will at AI Afterdark
Built using:
- Streamlit for web interface
- OpenRouter for cloud AI
- Ollama for local AI
- YouTube Transcript API for content extraction

## Contact
- Twitter: @AIAfterdark
- GitHub: AI Afterdark

---
Built by AI Afterdark - Innovating with AI at Night
