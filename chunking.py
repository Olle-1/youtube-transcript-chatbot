import os
import re
import json
from pathlib import Path

# Create directory structure
# Use the current directory as the base directory
base_dir = Path(".")  # Current directory
raw_dir = base_dir / "raw"
processed_dir = base_dir / "processed"
chunks_dir = base_dir / "chunks"

print(f"Looking for transcript files in: {os.path.abspath(raw_dir)}")

# Create directories if they don't exist
raw_dir.mkdir(parents=True, exist_ok=True)
processed_dir.mkdir(exist_ok=True)
chunks_dir.mkdir(exist_ok=True)

def clean_transcript(text):
    """Clean a YouTube transcript text."""
    # Remove counting sequences (e.g., "One, two, three, four...")
    text = re.sub(r'\b([Oo]ne|[Tt]wo|[Tt]hree|[Ff]our|[Ff]ive|[Ss]ix|[Ss]even|[Ee]ight|[Nn]ine|[Tt]en)[,\s]*', '', text)
    
    # Consolidate repeated words (e.g., "Okay. Okay. Okay.")
    text = re.sub(r'(\b\w+[.!?])\s+(\1\s+)+', r'\1 ', text, flags=re.IGNORECASE)
    
    # Remove non-speech elements like (applause), etc.
    text = re.sub(r'\([^)]*\)', '', text)
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', '\n', text)
    
    return text.strip()

def extract_metadata_from_filename(filename):
    """Extract video information from YouTube filename format."""
    base_name = os.path.basename(filename).replace('.txt', '')
    
    # New YouTube filename format: https_www.youtube.com_watch_v_VIDEO-ID_TITLE_VIDEO-ID_YEAR
    parts = base_name.split('_')
    
    if len(parts) >= 7 and parts[0] == 'https' and parts[1] == 'www.youtube.com' and parts[2] == 'watch' and parts[3] == 'v':
        # Extract video ID (the part after "v_")
        video_id = parts[4]
        
        # Last part is now the year
        year = parts[-1]
        
        # Second to last part is the video ID again
        last_id = parts[-2]
        
        # Everything in between is the title
        title_parts = parts[5:-2]  # Exclude both the video ID and year
        title = ' '.join(title_parts).replace('_', ' ')
        
        # Construct proper YouTube URL
        url = f"https://www.youtube.com/watch?v={video_id}"
        
        return {
            "url": url,
            "title": title,
            "id": video_id,
            "year": year,  # Add the year to metadata
            "original_filename": base_name
        }
    else:
        # Fallback for other filename formats
        print(f"  Warning: Couldn't parse filename format for: {filename}")
        return {
            "url": "unknown",
            "title": base_name.replace('_', ' '),
            "id": base_name,
            "year": "unknown",  # Add a default year
            "original_filename": base_name
        }

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    # Print some debug info
    print(f"  Text length: {len(text)} characters")
    
    # Always use sentence-based chunking for more reliable results
    sentences = re.split(r'(?<=[.!?])\s+', text)
    print(f"  Found {len(sentences)} sentences")
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk size and we already have content
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            # Include overlap by keeping some of the previous text
            overlap_point = max(0, len(current_chunk) - overlap)
            current_chunk = current_chunk[overlap_point:] + sentence + " "
        else:
            current_chunk += sentence + " "
    
    # Add the final chunk if not empty
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    print(f"  Created {len(chunks)} chunks")
    return chunks

def process_transcript_file(file_path):
    """Process a single transcript file."""
    print(f"Processing: {file_path}")
    
    # Extract metadata from filename
    metadata = extract_metadata_from_filename(file_path)
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as f:
        transcript_text = f.read()
    
    print(f"  Original file size: {len(transcript_text)} characters")
    
    # Clean the transcript
    cleaned_text = clean_transcript(transcript_text)
    print(f"  After cleaning: {len(cleaned_text)} characters")
    
    # Save the cleaned transcript
    processed_path = processed_dir / os.path.basename(file_path)
    with open(processed_path, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)
    
    # Chunk the transcript
    text_chunks = chunk_text(cleaned_text, chunk_size=1000)  # Increased chunk size
    
    # Save chunks with metadata
    chunk_data = []
    for i, chunk in enumerate(text_chunks):
        chunk_item = {
            "chunk_id": f"{metadata['id']}_{i}",
            "content": chunk,
            "metadata": metadata
        }
        chunk_data.append(chunk_item)
    
    # Save all chunks for this transcript
    chunks_path = chunks_dir / f"{metadata['id']}_chunks.json"
    with open(chunks_path, 'w', encoding='utf-8') as f:
        json.dump(chunk_data, f, indent=2)
    
    print(f"  Created {len(text_chunks)} chunks for this transcript")
    return len(text_chunks)

def main():
    """Process all transcript files in the raw directory."""
    # Get all .txt files in the raw directory
    transcript_files = list(raw_dir.glob('*.txt'))
    
    if not transcript_files:
        print(f"No .txt files found in {raw_dir}. Please add your transcript files there.")
        return
    
    print(f"Found {len(transcript_files)} transcript files to process.")
    
    total_chunks = 0
    for file_path in transcript_files:
        chunks = process_transcript_file(file_path)
        total_chunks += chunks
    
    print(f"Processing complete!")
    print(f"Processed {len(transcript_files)} transcripts into {total_chunks} chunks.")
    print(f"Cleaned transcripts saved to: {processed_dir}")
    print(f"Chunked data saved to: {chunks_dir}")
    print("\nNext steps would be to generate embeddings for these chunks.")

if __name__ == "__main__":
    main()