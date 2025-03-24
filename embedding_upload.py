import os
import json
import glob
from pathlib import Path
from dotenv import load_dotenv
import openai
from tqdm import tqdm  # For progress bars
import time
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Check if the API keys are available
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found. Create a .env file with OPENAI_API_KEY=your_key")
if not PINECONE_API_KEY:
    raise ValueError("Pinecone API key not found. Add PINECONE_API_KEY to your .env file")

# Initialize the OpenAI client with new API pattern
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Define directories
base_dir = Path(".")  # Current directory
chunks_dir = base_dir / "chunks"
embeddings_dir = base_dir / "embeddings"

# Create embeddings directory if it doesn't exist
embeddings_dir.mkdir(exist_ok=True)

# Define the embedding model and parameters
EMBEDDING_MODEL = "text-embedding-ada-002"  # OpenAI's embeddings model
EMBEDDING_DIMENSION = 1536  # Dimension for this model
INDEX_NAME = "youtube-transcript-mountaindog1"  # Custom name for your Pinecone index

def create_pinecone_index():
    """Create a Pinecone index if it doesn't exist."""
    # List all indexes
    indexes = pc.list_indexes()
    index_names = [idx.name for idx in indexes]
    
    # Check if the index already exists
    if INDEX_NAME not in index_names:
        print(f"Creating new Pinecone index: {INDEX_NAME}")
        # Create the index with the appropriate dimension
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",  # Use cosine similarity
            spec=ServerlessSpec(cloud="aws", region="us-east-1")  # Using us-east-1 for free tier
        )
        # Wait for index to be ready
        print("Waiting for index to be ready...")
        time.sleep(30)  # Give it some time to initialize
    else:
        print(f"Using existing Pinecone index: {INDEX_NAME}")
    
    # Connect to the index
    index = pc.Index(INDEX_NAME)
    return index

def generate_embedding(text):
    """Generate an embedding for a text using OpenAI's API."""
    try:
        # Using new OpenAI API pattern (v1.0.0+)
        response = client.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def process_chunks_file(file_path, index):
    """Process a chunks file, generate embeddings, and upload to Pinecone."""
    print(f"Processing: {file_path}")
    
    # Read the chunks file
    with open(file_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"Found {len(chunks_data)} chunks")
    
    # Prepare vectors for batch upload
    vectors_to_upsert = []
    
    # Process each chunk
    for chunk in tqdm(chunks_data, desc="Generating embeddings"):
        # Generate embedding for the chunk content
        embedding = generate_embedding(chunk["content"])
        
        if embedding:
            # Prepare vector with metadata for Pinecone
            vector = {
                "id": chunk["chunk_id"],
                "values": embedding,
                "metadata": {
                    "text": chunk["content"],
                    "url": chunk["metadata"].get("url", ""),
                    "title": chunk["metadata"].get("title", ""),
                    "video_id": chunk["metadata"].get("id", "")
                }
            }
            vectors_to_upsert.append(vector)
            
            # Save to local file (for backup)
            output_path = embeddings_dir / f"{chunk['chunk_id']}.json"
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "chunk_id": chunk["chunk_id"],
                    "embedding": embedding,
                    "metadata": chunk["metadata"],
                    "content": chunk["content"]
                }, f)
    
    # Upload vectors to Pinecone in batches
    batch_size = 100  # Pinecone recommends batches of ~100
    for i in range(0, len(vectors_to_upsert), batch_size):
        batch = vectors_to_upsert[i:i+batch_size]
        try:
            index.upsert(vectors=batch)
            print(f"Uploaded batch {i//batch_size + 1}/{(len(vectors_to_upsert)-1)//batch_size + 1}")
        except Exception as e:
            print(f"Error uploading batch to Pinecone: {e}")
    
    return len(vectors_to_upsert)

def main():
    """Main function to process all chunk files and upload to Pinecone."""
    # Create or connect to Pinecone index
    index = create_pinecone_index()
    
    # Get all chunk files
    chunk_files = list(chunks_dir.glob("*_chunks.json"))
    
    if not chunk_files:
        print(f"No chunk files found in {chunks_dir}. Run the chunking script first.")
        return
    
    print(f"Found {len(chunk_files)} chunk files to process.")
    
    # Process each file
    total_vectors = 0
    for file_path in chunk_files:
        vectors_uploaded = process_chunks_file(file_path, index)
        total_vectors += vectors_uploaded
    
    # Get index stats
    stats = index.describe_index_stats()
    print(f"\nProcess completed!")
    print(f"Total vectors uploaded: {total_vectors}")
    print(f"Total vectors in Pinecone index: {stats.total_vector_count}")
    print(f"Vector dimension: {stats.dimension}")

if __name__ == "__main__":
    main()