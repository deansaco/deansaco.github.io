# Dean Sacoransky's Personal Portfolio with AI Chatbot

This repository contains the code for Dean Sacoransky's personal portfolio website, which includes an AI-powered chatbot that can answer questions about Dean's professional work and experience.

## Features

- Clean, responsive portfolio website
- Interactive timeline of professional experience
- AI chatbot with RAG (Retrieval Augmented Generation) functionality
- Python Flask backend with OpenAI API integration

## Setup Instructions

### 1. Set Up the Python Backend

First, you need to set up the Python backend server that powers the RAG functionality:

1. Install Python 3.8 or higher if you don't have it already
2. Clone this repository and navigate to the project directory
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```
4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
5. Get an OpenAI API key from [https://openai.com](https://openai.com)
6. Create or edit the `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ```
7. Start the Python server:
   ```bash
   python rag_server.py
   ```
   The server will start on `http://localhost:5000`

### 2. Serve the Frontend

You can serve the frontend using any web server. For development, you can use Python's built-in HTTP server:

```bash
# In a new terminal
python -m http.server 8000
```

Then open your browser and navigate to `http://localhost:8000`

## How It Works

The chatbot uses RAG (Retrieval Augmented Generation) technology to provide accurate answers about Dean's professional experience:

1. **Backend Server**: A Flask server handles API requests and maintains the content embeddings in memory.
2. **Content Indexing**: When the server starts, it creates vector embeddings for all of Dean's work content using OpenAI's embedding API.
3. **Semantic Search**: When a user asks a question, the system converts the question to an embedding and finds the most relevant content using cosine similarity.
4. **Response Generation**: The retrieved content is sent to OpenAI's GPT-4 model along with the user's question to generate a natural and informative response.

## API Endpoints

The Python backend provides the following API endpoints:

- `GET /api/health` - Check if the server is running
- `POST /api/chat` - Submit a query and get a RAG-generated response

## Customizing the Content

To update the content that the chatbot knows about:

1. Open `rag_server.py`
2. Find the `deans_content` list
3. Modify or add to the list with your own information
4. Each content item should include at least a `title` and `content` field:
   ```python
   {
       "title": "Project Name",
       "content": "Detailed description of the project",
       "tech": "Technologies used (optional)",
       "link": "URL to the project (optional)"
   }
   ```

## Deployment

For production deployment:

1. Host the Python backend on a service like Heroku, AWS, or Google Cloud
2. Update the `BACKEND_URL` in `index.html` to point to your deployed backend
3. Host the frontend files on a static hosting service like GitHub Pages, Netlify, or Vercel

## Security Considerations

- The current implementation stores the API key in a `.env` file, which is more secure than client-side JavaScript
- For production, consider using environment variables in your hosting platform
- You may want to add rate limiting and authentication to the API endpoints

## License

This project is licensed under the MIT License - see the LICENSE file for details. 