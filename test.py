"""
Query Qdrant to inspect actual chunk_index values
This will show you what's actually stored in the vector database
"""

# This script requires qdrant-client to be installed
# pip install qdrant-client

from qdrant_client import QdrantClient
import os
from typing import List, Dict


def inspect_qdrant_chunks():
    """
    Connect to Qdrant and inspect chunk_index values
    """
    # You'll need to provide your Qdrant credentials
    QDRANT_URL = os.getenv('QDRANT_URL','https://16b38aab-e1db-42a8-85b8-85b8dedec06b.us-east4-0.gcp.cloud.qdrant.io')
    QDRANT_API_KEY = os.getenv('QDRANT_API_KEY','eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.C-OrjDqbtCP-4FjIqWbYzDTcuXBs7IWCKtR82K9RdNE')
    COLLECTION_NAME = os.getenv('QDRANT_COLLECTION_NAME','one-rag-collection')
    
    print("\n" + "="*80)
    print("CONNECTING TO QDRANT")
    print("="*80)
    print(f"URL: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        
        # Get collection info
        collection_info = client.get_collection(COLLECTION_NAME)
        print(f"\n✓ Connected to collection")
        print(f"  Total points: {collection_info.points_count}")
        
        # Scroll through points
        print("\n" + "="*80)
        print("INSPECTING CHUNKS (first 20)")
        print("="*80)
        
        result = client.scroll(
            collection_name=COLLECTION_NAME,
            limit=20,
            with_payload=True,
            with_vectors=False
        )
        
        points = result[0]
        
        if not points:
            print("⚠️  No points found in collection!")
            return
        
        # Analyze chunk_index values
        chunk_indices = []
        file_groups = {}
        
        print(f"\nFound {len(points)} points:\n")
        
        for i, point in enumerate(points):
            payload = point.payload
            
            chunk_index = payload.get('chunk_index', 'MISSING')
            page_number = payload.get('page_number', 'MISSING')
            file_id = payload.get('file_id', 'unknown')
            chunk_id = payload.get('chunk_id', 'unknown')
            
            # Group by file
            if file_id not in file_groups:
                file_groups[file_id] = []
            file_groups[file_id].append({
                'chunk_index': chunk_index,
                'page_number': page_number,
                'chunk_id': chunk_id
            })
            
            chunk_indices.append(chunk_index)
            
            # Show first 10
            if i < 10:
                print(f"Point {i}:")
                print(f"  file_id: {file_id[:20]}...")
                print(f"  chunk_id: {chunk_id[:30]}...")
                print(f"  chunk_index: {chunk_index}")
                print(f"  page_number: {page_number}")
                print()
        
        # Analysis
        print("\n" + "="*80)
        print("ANALYSIS")
        print("="*80)
        
        unique_indices = set(chunk_indices)
        print(f"\nUnique chunk_index values: {sorted(unique_indices)}")
        print(f"Total unique values: {len(unique_indices)}")
        
        if unique_indices == {0}:
            print("\n❌ CONFIRMED BUG: All chunks have chunk_index = 0!")
        elif len(unique_indices) > 1:
            print(f"\n✅ LOOKS GOOD: Found {len(unique_indices)} different chunk_index values")
        
        # Analyze by file
        print("\n" + "="*80)
        print("ANALYSIS BY FILE")
        print("="*80)
        
        for file_id, chunks in list(file_groups.items())[:3]:  # First 3 files
            print(f"\nFile: {file_id[:40]}...")
            chunk_indices_for_file = [c['chunk_index'] for c in chunks]
            page_numbers_for_file = [c['page_number'] for c in chunks]
            
            print(f"  Chunks: {len(chunks)}")
            print(f"  chunk_index values: {chunk_indices_for_file}")
            print(f"  page_number values: {page_numbers_for_file}")
            
            if len(set(chunk_indices_for_file)) == 1 and chunk_indices_for_file[0] == 0:
                print(f"  ❌ All chunks have chunk_index = 0 (BUG!)")
            else:
                print(f"  ✅ Chunks have different indices")
        
        # Recommendation
        print("\n" + "="*80)
        print("RECOMMENDATION")
        print("="*80)
        
        if unique_indices == {0} or (len(unique_indices) == 1 and 0 in unique_indices):
            print("""
❌ CONFIRMED: All chunks have chunk_index = 0

This means there's a bug in how chunk_index is being stored.

POSSIBLE CAUSES:
1. base_metadata is being shared and mutated
2. chunk_index is being overwritten somewhere
3. There's a version mismatch between deployed files

DEBUGGING STEPS:
1. Add logging in vector_store_service.py line 284:
   logger.debug(f"Processing chunk: chunk_index={metadata.get('chunk_index')}, from chunk dict={chunk.get('chunk_index')}")

2. Add logging in chunking_service.py line 1007:
   logger.debug(f"Created chunk {chunk_index} with metadata chunk_index={metadata['chunk_index']}")

3. Check upload.py to see if chunks are being modified before vector store

4. Verify the deployed version matches your local version
            """)
        else:
            print("""
✅ Chunks have different chunk_index values - system is working correctly!

If you're still seeing issues, check:
1. Are you looking at old data before a recent fix?
2. Are specific files affected?
3. Is the database vs vector store out of sync?
            """)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure to set environment variables:")
        print("  export QDRANT_URL='your_url'")
        print("  export QDRANT_API_KEY='your_key'")
        print("  export QDRANT_COLLECTION_NAME='your_collection'")

if __name__ == "__main__":
    inspect_qdrant_chunks()