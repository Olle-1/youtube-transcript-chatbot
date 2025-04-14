import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pinecone import Pinecone
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# API keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Constants
INDEX_NAME = "youtube-transcript-mountaindog1"
CHATBOT_NAME = "MountainDog1 Assistant"

def initialize_retriever():
    """Connect to Pinecone and set up the retriever."""
    print(f"Connecting to Pinecone index: {INDEX_NAME}")
    
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Initialize embeddings model
    embeddings = OpenAIEmbeddings()
    
    # Connect to existing Pinecone index
    vector_store = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
        text_key="text"  # Field in metadata containing the text content
    )
    
    # Create retriever with MMR (Maximum Marginal Relevance) for better diversity
    retriever = vector_store.as_retriever(
        search_type="mmr",  # Use MMR for diversity in results
        search_kwargs={
            "k": 5,  # Return 5 most relevant chunks
            "fetch_k": 10,  # Fetch 10 candidates first
            "lambda_mult": 0.7  # Balance between relevance (1.0) and diversity (0.0)
        }
    )
    
    return retriever

def create_chain(retriever, system_prompt: str):
    """Create the conversational chain with a specific system prompt."""
    # Use the provided system prompt
    # Combine the dynamic system prompt with the rest of the template structure
    custom_template = f"""{system_prompt}

Answer the question based ONLY on the following context:

{{context}}

If you don't know the answer based on the context, just say "I don't have enough information about that in my knowledge base." Don't make up answers.

Chat History:
{{chat_history}}

Question: {{question}}

Answer:
"""
    
    CUSTOM_PROMPT = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=custom_template
    )
    
    # Initialize conversation memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        input_key="question",
        output_key="answer"
    )
    
    # Initialize LLM
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # Use a cheaper model like gpt-3.5-turbo to save costs
        temperature=0.3  # Lower temperature for more consistent responses
    )
    
    # Create the chain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # Return source docs for citation
        combine_docs_chain_kwargs={"prompt": CUSTOM_PROMPT},
        chain_type="stuff"  # 'stuff' method: simply stuffs all documents into the prompt
    )
    
    return chain

def format_response(response):
    """Format the response with source citations."""
    answer = response["answer"]
    source_docs = response["source_documents"]
    
    # Add source citations
    if source_docs:
        answer += "\n\nSources:"
        seen_titles = set()
        for i, doc in enumerate(source_docs[:3]):  # Limit to first 3 sources
            title = doc.metadata.get("title", "Unknown Video")
            url = doc.metadata.get("url", "")
            
            # Skip duplicates
            if title in seen_titles:
                continue
            seen_titles.add(title)
            
            answer += f"\n- [{title}]({url})"
    
    return answer

def chat_loop():
    """Run an interactive chat loop."""
    print(f"Initializing {CHATBOT_NAME}...")
    retriever = initialize_retriever()
    # TODO: This chat_loop is for standalone testing and needs updating
    # or removal if only used via API. For now, provide a default prompt.
    default_prompt = "You are a helpful AI assistant." # Placeholder default
    chain = create_chain(retriever, default_prompt)
    
    print(f"\n{CHATBOT_NAME} is ready! Type 'exit' to end the conversation.")
    print("Ask any question about MountainDog1's training methods, programs, or advice.")
    
    while True:
        user_input = input("\nYou: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Thank you for using the chatbot!")
            break
        
        if not user_input.strip():
            continue
        
        try:
            # Use invoke method instead of calling the chain directly
            response = chain.invoke({"question": user_input})
            formatted_response = format_response(response)
            print(f"\n{CHATBOT_NAME}: {formatted_response}")
        except Exception as e:
            print(f"Error: {e}")
            print("Sorry, I encountered an error processing your request. Please try again.")

if __name__ == "__main__":
    chat_loop()