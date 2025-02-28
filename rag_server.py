import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import re
import time
from urllib.parse import urlparse

# Load environment variables from .env file (if it exists)
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY environment variable not set")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Dean's work content for RAG - now just metadata and links
deans_content = [
    {
        "title": "Radar Localization",
        "summary": "Developed source code for interfacing with the ARS408 radar Husky Robot, implementing localization algorithm using DBSCAN clustering, Kalman filtering, and GNN data association for autonomous robot control.",
        "link": "https://github.com/offroad-robotics/radar-localization"
    },
    {
        "title": "ICRA Publication",
        "summary": "Published 'Towards unsupervised filtering of millimetre-wave radar returns for autonomous vehicle road following' at IEEE International Conference on Robotics and Automation 2023.",
        "link": "https://qspace.library.queensu.ca/items/dc30e61c-09e6-42e4-8db2-b8d8defd40e0"
    },
    {
        "title": "Master's Thesis",
        "summary": "Authored thesis on 'Autonomous Vehicle Localization Using Automotive Radar and Reflective Lane Markers'.",
        "link": "https://qspace.library.queensu.ca/items/f5725262-4dd0-4992-a00d-ce5b683dc001"
    },
    {
        "title": "r1-reasoning-rag",
        "summary": "Open source project using deepseek's r1 reasoning to agentically retrieve, discard, and synthesize information from a knowledge base to answer complex questions.",
        "link": "https://github.com/deansaco/r1-reasoning-rag"
    },
    {
        "title": "IBM RAG Cookbook",
        "summary": "Contributed to a compendium of tips, tricks, and techniques for implementing and optimizing Retrieval Augmented Generation (RAG) solutions for large enterprise clients.",
        "link": "https://www.ibm.com/architectures/papers/rag-cookbook"
    },
    {
        "title": "Building RAG LLM Agents",
        "summary": "Created IBM Developer tutorial on building Agentic RAG with watsonx, LangGraph, Elasticsearch, and Tavily.",
        "link": "https://developer.ibm.com/tutorials/awb-build-rag-llm-agents/"
    }
]

# Content chunks and embeddings cache
content_chunks = []
content_embeddings = None

def scrape_content(url):
    """Scrape content from a URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        domain = urlparse(url).netloc
        
        # Different parsing strategies based on domain
        if 'github.com' in domain:
            return scrape_github(response.text, url)
        elif 'ieeexplore.ieee.org' in domain:
            return scrape_ieee(response.text)
        elif 'qspace.library.queensu.ca' in domain:
            return scrape_qspace(response.text)
        elif 'ibm.com' in domain:
            return scrape_ibm(response.text)
        else:
            # Generic scraping
            return scrape_generic(response.text)
            
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return f"Error scraping content: {str(e)}"

def scrape_github(html_content, url):
    """Scrape content from GitHub"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Get README content
    readme = soup.find('article', class_='markdown-body')
    if readme:
        # Extract text from README
        text = readme.get_text(separator='\n', strip=True)
        
        # Also try to get repository description
        repo_desc = soup.find('p', class_='f4 my-3')
        if repo_desc:
            text = repo_desc.get_text(strip=True) + "\n\n" + text
            
        return text
    
    # If no README found, try to get repository description and file structure
    result = []
    
    # Repository description
    repo_desc = soup.find('p', class_='f4 my-3')
    if repo_desc:
        result.append(repo_desc.get_text(strip=True))
    
    # File structure
    file_list = soup.find_all('a', class_='js-navigation-open')
    if file_list:
        result.append("Repository files:")
        for file in file_list[:20]:  # Limit to first 20 files
            result.append(f"- {file.get_text(strip=True)}")
    
    return "\n".join(result) if result else f"Unable to extract content from {url}"

def scrape_ieee(html_content):
    """Scrape content from IEEE Xplore"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result = []
    
    # Title
    title = soup.find('h1', class_='document-title')
    if title:
        result.append(title.get_text(strip=True))
    
    # Abstract
    abstract = soup.find('div', class_='abstract-text')
    if abstract:
        result.append("Abstract:")
        result.append(abstract.get_text(strip=True))
    
    # Authors
    authors = soup.find_all('a', class_='author')
    if authors:
        result.append("Authors:")
        for author in authors:
            result.append(f"- {author.get_text(strip=True)}")
    
    # Publication info
    pub_info = soup.find('div', class_='u-pb-1')
    if pub_info:
        result.append("Publication Info:")
        result.append(pub_info.get_text(strip=True))
    
    return "\n".join(result) if result else "Unable to extract content from IEEE Xplore"

def scrape_qspace(html_content):
    """Scrape content from QSpace"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    result = []
    
    # Title
    title = soup.find('h2', class_='item-page-title')
    if title:
        result.append(title.get_text(strip=True))
    
    # Abstract
    abstract = soup.find('div', class_='simple-item-view-description')
    if abstract:
        result.append("Abstract:")
        result.append(abstract.get_text(strip=True))
    
    # Metadata
    metadata = soup.find_all('div', class_='simple-item-view-other')
    for item in metadata:
        result.append(item.get_text(strip=True))
    
    return "\n".join(result) if result else "Unable to extract content from QSpace"

def scrape_ibm(html_content):
    """Scrape content from IBM website"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Get main content
    main_content = soup.find('main')
    if main_content:
        # Extract all paragraphs
        paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
        text = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        return text
    
    return "Unable to extract content from IBM website"

def scrape_generic(html_content):
    """Generic scraping for any website"""
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.extract()
    
    # Get text
    text = soup.get_text(separator='\n', strip=True)
    
    # Clean up text
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    
    return text

def chunk_text(text, max_chunk_size=1000, overlap=100):
    """Split text into overlapping chunks"""
    chunks = []
    
    # Clean text
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If text is shorter than max chunk size, return it as a single chunk
    if len(text) <= max_chunk_size:
        return [text]
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    current_chunk = ""
    for sentence in sentences:
        # If adding this sentence would exceed max_chunk_size
        if len(current_chunk) + len(sentence) > max_chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Keep some overlap for context
            current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
        
        current_chunk += " " + sentence
    
    # Add the last chunk if it's not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def create_embedding(text):
    """Create an embedding for the given text using OpenAI API"""
    try:
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error creating embedding: {e}")
        raise

def initialize_content():
    """Scrape content from links, chunk it, and create embeddings"""
    global content_chunks, content_embeddings
    
    if content_chunks and content_embeddings is not None:
        return content_chunks, content_embeddings
    
    print("Initializing content by scraping links...")
    
    content_chunks = []
    
    for item in deans_content:
        try:
            print(f"Scraping content from: {item['link']}")
            scraped_content = scrape_content(item['link'])
            
            # Chunk the content
            chunks = chunk_text(scraped_content)
            
            # Add metadata to each chunk
            for i, chunk in enumerate(chunks):
                content_chunks.append({
                    "title": item["title"],
                    "link": item["link"],
                    "chunk_id": f"{item['title']}_chunk_{i}",
                    "content": chunk
                })
            
            # Be nice to servers and avoid rate limiting
            time.sleep(1)
            
        except Exception as e:
            print(f"Error processing {item['title']}: {e}")
            # Add the summary as a fallback
            content_chunks.append({
                "title": item["title"],
                "link": item["link"],
                "chunk_id": f"{item['title']}_summary",
                "content": item["summary"]
            })
    
    print(f"Content chunking complete. Created {len(content_chunks)} chunks.")
    
    # Create embeddings for all chunks
    print("Creating embeddings for content chunks...")
    content_with_embeddings = []
    
    for chunk in content_chunks:
        try:
            embedding = create_embedding(chunk["content"])
            
            content_with_embeddings.append({
                **chunk,
                "embedding": embedding
            })
        except Exception as e:
            print(f"Error creating embedding for {chunk['chunk_id']}: {e}")
    
    print("Content embeddings initialized successfully!")
    content_embeddings = content_with_embeddings
    
    return content_chunks, content_embeddings

def find_similar_content(query, content_with_embeddings, similarity_threshold=0.7, max_results=5):
    """Find content items similar to the query using cosine similarity"""
    query_embedding = create_embedding(query)
    
    # Convert embeddings to numpy arrays for efficient computation
    query_embedding_np = np.array([query_embedding])
    
    content_embeddings_np = np.array([item["embedding"] for item in content_with_embeddings])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_embedding_np, content_embeddings_np)[0]
    
    # Create list of content items with their similarity scores
    content_with_scores = []
    for i, similarity in enumerate(similarities):
        content_with_scores.append({
            **content_with_embeddings[i],
            "similarity": float(similarity)  # Convert numpy float to Python float for JSON serialization
        })
    
    # Sort by similarity score in descending order
    content_with_scores.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Filter by similarity threshold and limit results
    filtered_results = [item for item in content_with_scores if item["similarity"] > similarity_threshold]
    return filtered_results[:max_results]

def generate_rag_response(query):
    """Generate a response using RAG approach"""
    try:
        # Make sure content is initialized
        _, content_with_embeddings = initialize_content()
        
        # Find relevant content
        relevant_content = find_similar_content(query, content_with_embeddings)
        
        # Format context for the prompt
        context = ""
        if relevant_content:
            context_parts = []
            for item in relevant_content:
                context_parts.append(f"Source: {item['title']} (Link: {item['link']})\nContent: {item['content']}")
            context = "\n\n".join(context_parts)
        
        # Create messages for the chat completion
        messages = [
            {
                "role": "system",
                "content": """You are an AI assistant for Dean Sacoransky, designed to answer questions about his professional work, experience, and projects. 
                Be friendly, helpful, and concise. If you're not sure about something, be honest about your limitations.
                Base your responses on the following information about Dean's work:"""
            }
        ]
        
        # Add context if available
        if context:
            messages.append({
                "role": "system", 
                "content": context
            })
        else:
            # Fallback to summaries if no content was retrieved
            fallback_context = "\n\n".join([f"{item['title']}: {item['summary']}" for item in deans_content])
            messages.append({
                "role": "system", 
                "content": "No detailed content was retrieved, but here are summaries of Dean's work:\n\n" + fallback_context
            })
        
        # Add user query
        messages.append({
            "role": "user",
            "content": query
        })
        
        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        print(f"Error generating RAG response: {e}")
        return f"I'm sorry, I couldn't process your request at the moment. Error: {str(e)}"

# API Routes
@app.route('/api/chat', methods=['POST'])
def chat():
    """API endpoint for chat"""
    try:
        data = request.json
        query = data.get('query')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        response = generate_rag_response(query)
        return jsonify({"response": response})
    
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Simple health check endpoint"""
    return jsonify({"status": "ok", "message": "RAG server is running"})

# Main entry point
if __name__ == '__main__':
    # Initialize content on startup
    initialize_content()
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 