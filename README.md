# YouTube Video Summarizer
Created by AI Afterdark - Building Innovation with AI at Night
An AI-powered application that generates summaries of YouTube videos and enables interactive conversations about their content.
<img src="img/YoutubeVideoSummarizer.gif" alt="Demo" autoplay loop>

## Live Demo
Experience the YouTube Video Summarizer in action: [https://aiafterdark-youtube-summarizer.streamlit.app/](https://aiafterdark-youtube-summarizer.streamlit.app/)

## Features
- Robust YouTube video transcript extraction with multiple fallback methods:
  - YouTube Transcript API (primary)
  - Pytube captions
  - yt-dlp caption extraction
- AI-powered content summarization using OpenRouter's LLMs
- Interactive Q&A about video content
- Adjustable summary detail levels
- Clean, responsive UI
- Comprehensive error handling and reporting

## Cloud Deployment (Default)
### Prerequisites
- Python 3.11+
- pip (Python package manager)
- Git
- OpenRouter API key

### Quick Start
1. Clone the Repository
```bash
git clone https://github.com/AIAfterDark/youtube-summarizer-app.git
cd youtube-summarizer-app
```

2. Set Up Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

4. Configure Environment
Create a `.env` file in the root directory and add your OpenRouter API key:
```env
OPENROUTER_API_KEY=your_api_key_here
```

5. Run the Application
```bash
streamlit run app.py
```

## Local Deployment (Ollama)
The app-local.py version allows you to run the summarizer using Ollama on your local machine, which is free and doesn't require an API key.

### Prerequisites
- All requirements from Cloud Deployment
- Ollama installed on your machine

### Setup Steps
1. Install Ollama
   - Download from [ollama.ai](https://ollama.ai)
   - Follow the installation instructions for your OS
   - Make sure Ollama is running in the background

2. Pull Your Preferred Model
```bash
# Pull the default model (recommended)
ollama pull llama2

# Or pull other supported models
ollama pull codellama
ollama pull mistral
ollama pull neural-chat
```

3. Run the Local Version
```bash
streamlit run app-local.py
```

### Available Local Models
The following models are tested and supported in app-local.py:
- llama2 (default, recommended)
- codellama
- mistral
- neural-chat

### Configuration
You can modify these settings in app-local.py:
- Default model: Change `model="llama2"` in the `ollama_completion` function
- API endpoint: Default is `http://localhost:11434/api/chat`
- Timeout settings: Default is 30 seconds

## Configuration
### Summary Detail Level
Adjust the chunk size based on video length:
- Short videos (<30 mins): 4000
- Long content (1hr+): 7000+

### Cloud Models (app.py)
The app uses OpenRouter's API to access various LLM models:
- meta-llama/llama-2-13b-chat (default)
- anthropic/claude-2
- openai/gpt-3.5-turbo

## Contributing
We welcome contributions! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

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

# YouTube Video Summarizer

An AI-powered Streamlit app that generates summaries of YouTube videos and allows you to chat with the content. Available in two versions: Cloud (OpenRouter) and Local (Ollama).

## Features

- YouTube video transcript extraction
- Intelligent text chunking and processing
- AI-powered summarization
- Interactive chat with video content
- Multiple transcript retrieval methods
- Cloud and Local deployment options

## Versions

### Cloud Version (app.py)
- Uses OpenRouter API for AI inference
- Requires OpenRouter API key
- Better for deployment and sharing
- Uses meta-llama/llama-3.2-3b-instruct:free model

### Local Version (app-local.py)
- Uses Ollama for local AI inference
- No API key required
- Better for privacy and offline use
- Supports multiple Ollama models

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/youtube-summarizer-app.git
cd youtube-summarizer-app
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Setup based on version:

### For Cloud Version (app.py):
1. Get an API key from [OpenRouter](https://openrouter.ai/)
2. Create a `.env` file:
```bash
OPENROUTER_API_KEY=your_api_key_here
```
3. Run the app:
```bash
streamlit run app.py
```

### For Local Version (app-local.py):
1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Pull the Llama2 model:
```bash
ollama pull llama2
```
3. Start Ollama:
```bash
ollama serve
```
4. Run the app:
```bash
streamlit run app-local.py
```

## Usage

1. Enter a YouTube URL
2. Adjust the Summary Detail Level slider (1000-10000)
3. Click "Generate Summary"
4. View the generated summary
5. Use the chat to ask questions about the video content

## Features in Detail

### Transcript Retrieval
- Primary: YouTube Transcript API
- Fallback: yt-dlp
- Supports both manual and auto-generated captions

### Text Processing
- Smart chunking based on sentence boundaries
- Context-aware summarization
- Clean transcript formatting

### Chat Interface
- Context-aware responses
- Strictly based on generated summary
- Clear indication when information isn't available

## Requirements

- Python 3.7+
- Streamlit
- youtube-transcript-api
- yt-dlp
- For Cloud Version: OpenRouter API key
- For Local Version: Ollama

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License - feel free to use this project as you wish.
