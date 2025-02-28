"""
Test script for the RAG implementation in rag_server.py

This script tests the embedding creation and RAG response generation
functionality without starting the Flask server.
"""

import os
from dotenv import load_dotenv
from rag_server import create_embedding, initialize_content_embeddings, generate_rag_response

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Error: OPENAI_API_KEY environment variable not set.")
    print("Please set it in your .env file or export it to your environment.")
    exit(1)

def test_create_embedding():
    """Test creating embeddings"""
    print("\n=== Testing Embedding Creation ===")
    try:
        text = "Autonomous vehicle localization using radar"
        embedding = create_embedding(text)
        
        # Check if embedding is a list/array with expected dimensions
        if isinstance(embedding, list) and len(embedding) > 0:
            print(f"✅ Successfully created embedding with {len(embedding)} dimensions")
            return True
        else:
            print(f"❌ Embedding creation failed: Invalid embedding format")
            return False
    
    except Exception as e:
        print(f"❌ Embedding creation failed: {e}")
        return False

def test_initialize_embeddings():
    """Test initializing content embeddings"""
    print("\n=== Testing Content Embeddings Initialization ===")
    try:
        embeddings = initialize_content_embeddings()
        
        if embeddings and len(embeddings) > 0:
            print(f"✅ Successfully initialized {len(embeddings)} content embeddings")
            return True
        else:
            print(f"❌ Content embeddings initialization failed: No embeddings created")
            return False
    
    except Exception as e:
        print(f"❌ Content embeddings initialization failed: {e}")
        return False

def test_rag_response():
    """Test generating RAG responses"""
    print("\n=== Testing RAG Response Generation ===")
    try:
        queries = [
            "What was Dean's work on robotics?",
            "Tell me about Dean's experience at IBM",
            "What is r1-reasoning-rag?",
            "What are Dean's technical skills?"
        ]
        
        for query in queries:
            print(f"\nQuery: {query}")
            response = generate_rag_response(query)
            print(f"Response: {response[:100]}...")  # Show first 100 chars
        
        print("\n✅ RAG response generation test completed")
        return True
    
    except Exception as e:
        print(f"❌ RAG response generation failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing RAG functionality...")
    
    # Run tests
    embedding_test = test_create_embedding()
    
    if embedding_test:
        embeddings_test = test_initialize_embeddings()
        
        if embeddings_test:
            rag_test = test_rag_response()
            
            if rag_test:
                print("\n=== All tests passed! ===")
                print("The RAG system is working correctly.")
                print("You can now start the Flask server with: python rag_server.py")
            else:
                print("\n❌ RAG response generation test failed.")
        else:
            print("\n❌ Content embeddings initialization test failed.")
    else:
        print("\n❌ Embedding creation test failed.")
        print("Please check your OpenAI API key and internet connection.") 