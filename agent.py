# agent.py
#
# This script creates a specialized AI agent to retrieve and analyze academic articles.
# It intelligently handles paywalls by first searching for open-access versions
# on PubMed before falling back to scraping publicly available data.
#
# Installation:
# pip install langchain langchain_google_genai langgraph langchain_community beautifulsoup4 requests pubmed-lookup python-dotenv

import os
import re
from typing import TypedDict, Annotated, Optional
import operator

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PubMedLoader
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- Configuration and Setup ---
# Load environment variables from a .env file
load_dotenv()

# IMPORTANT: Set up your Google API Key.
# You can get one from the Google AI Studio and set it in your .env file.
# Example .env file:
# GOOGLE_API_KEY="AIzaSy...your...key"

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

# Initialize the Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=GOOGLE_API_KEY
)


# --- Graph State Definition ---
class AgentState(TypedDict):
    """Defines the state of our agent as it runs."""
    original_question: str
    question: str
    data_context: str
    execution_result: str
    execution_error: Optional[str]
    task_type: str

# --- Helper Functions and Tools ---

def find_doi_in_text(text: str) -> Optional[str]:
    """Finds the first Digital Object Identifier (DOI) in a given string."""
    doi_pattern = r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b'
    match = re.search(doi_pattern, text)
    return match.group(0) if match else None

def scrape_and_extract(url: str) -> tuple[str, None]:
    """
    A simple web scraper to extract the main text from a URL.
    This includes a User-Agent header to avoid being blocked by anti-scraping measures.
    """
    try:
        # --- THIS IS THE KEY ADDITION ---
        # A User-Agent header makes the request look like it's from a real browser.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # This will raise an error for 4xx or 5xx responses
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements to clean the text
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Get text, clean it up, and return it
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        # The abstract and title are usually enough to answer the question
        if "abstract" in text.lower():
            print("   - Successfully scraped page content (including abstract).")
            return text, None 
        else:
            print("   - Warning: Scraped content but could not find an 'abstract'. May be incomplete.")
            return text, None

    except requests.RequestException as e:
        # This will now correctly report the 403 error if it still occurs
        print(f"   - Error during web scraping for {url}: {e}")
        return "", None



def pubmed_loader_tool(query: str) -> str:
    """Fetches documents from PubMed based on a query (ideally a DOI)."""
    try:
        loader = PubMedLoader(query=query, load_max_docs=1)
        documents = loader.load()
        if not documents:
            return "No documents found on PubMed for the given query."
        # Combine content of all fetched documents
        return "\n\n---\n\n".join([doc.page_content for doc in documents])
    except Exception as e:
        return f"An error occurred while fetching from PubMed: {e}"

# --- Agent Node Definitions ---

def task_router_node(state: AgentState):
    """Classifies the user's request to route it to the correct tool."""
    print("--- 1. ROUTING TASK ---")
    question = state['original_question'].lower()
    
    # If the query contains a URL or a DOI, it's an article analysis task.
    is_article_query = 'http' in question or '10.' in question
    
    if is_article_query:
        print("   - Task classified as: academic_article_analysis")
        state['task_type'] = 'academic_article_analysis'
    else:
        print("   - Task classified as: general_qa (unsupported)")
        state['task_type'] = 'error' # This agent only supports article analysis
    return state

def smart_article_retriever_node(state: AgentState):
    """
    Intelligently retrieves academic article content. Handles paywalls by
    searching for open-access versions via PubMed using a DOI.
    """
    print("--- 2. RETRIEVING ARTICLE ---")
    question_text = state['original_question']
    url = ""
    url_match = re.search(r'https?://\S+', question_text)
    if url_match:
        url = url_match.group(0).strip('">')

    # Attempt to find a DOI in the user's query or the URL itself
    doi = find_doi_in_text(f"{question_text} {url}")
    
    final_context = ""
    
    # Priority 1: Use DOI to get full text from PubMed (if open-access)
    if doi:
        print(f"   - Found DOI: {doi}. Querying PubMed...")
        pubmed_content = pubmed_loader_tool(doi)
        if "No documents found" not in pubmed_content and "error occurred" not in pubmed_content:
            print("   - Success: Retrieved full-text content from PubMed.")
            final_context = pubmed_content
    
    # Priority 2: If PubMed fails, fall back to scraping the abstract from the URL
    if not final_context and url:
        print("   - PubMed failed or no DOI found. Falling back to web scraping...")
        scraped_text, _ = scrape_and_extract(url)
        if scraped_text and scraped_text.strip():
             print("   - Success: Scraped public content (likely abstract) from URL.")
             final_context = scraped_text

    if not final_context:
        state['execution_error'] = "Fatal: Could not retrieve any content from the URL or open-access sources."
        state['task_type'] = 'error'
    else:
        state['data_context'] = final_context
        # Pass the original question to the next node for analysis
        state['question'] = state['original_question']
        
    return state

def summarizer_qa_node(state: AgentState):
    """Uses the Gemini LLM to answer the user's question based on the retrieved context."""
    print("--- 3. ANALYZING AND ANSWERING ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Your task is to carefully read the provided scientific article context and answer the user's question based ONLY on the information within the text. Be concise and precise."),
        ("user", "Article Context:\n\n{context}\n\nBased on the text above, please answer the following question:\n\nQuestion: {question}")
    ])
    
    chain = prompt | llm
    result = chain.invoke({
        "context": state['data_context'],
        "question": state['question']
    })
    
    print("--- 4. FINAL ANSWER GENERATED ---")
    state['execution_result'] = result.content
    return state

# --- Graph Assembly ---

workflow = StateGraph(AgentState)

# Add nodes to the graph
workflow.add_node("task_router", task_router_node)
workflow.add_node("smart_article_retriever", smart_article_retriever_node)
workflow.add_node("summarizer_qa", summarizer_qa_node)

# Define the workflow edges
workflow.set_entry_point("task_router")

workflow.add_conditional_edges(
    "task_router",
    lambda state: state.get("task_type"),
    {
        "academic_article_analysis": "smart_article_retriever",
        "error": END
    }
)

workflow.add_conditional_edges(
    "smart_article_retriever",
     lambda state: "error" if state.get("execution_error") else "continue",
    {
        "continue": "summarizer_qa",
        "error": END
    }
)

workflow.add_edge("summarizer_qa", END)

# Compile the final graph
app = workflow.compile()


# --- Main Execution Block ---

if __name__ == "__main__":
    # Example usage of the agent
    
    # This is the query from your example
    user_query = (
        'check this article and give me information on "amino acid Mutation", '
        '"Name of the pathogen" and "Name of the drug"\n'
        'the url is https://journals.asm.org/doi/10.1128/aac.00637-15'
    )
    
    # The initial state for the agent
    initial_state = {"original_question": user_query}

    # Run the agent
    final_state = app.invoke(initial_state)
    
    # Print the final result or any errors
    if final_state.get("execution_error"):
        print("\n--- Agent Run Failed ---")
        print(f"Error: {final_state['execution_error']}")
    else:
        print("\n--- Agent Run Successful ---")
        print("Final Answer:")
        print(final_state['execution_result'])

