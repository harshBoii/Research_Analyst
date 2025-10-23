# agent.py
#
# This script creates a specialized AI agent to retrieve and analyze academic articles.
# It intelligently handles both single and multiple URLs in a query.
#
# Installation:
# pip install langchain langchain_google_genai langgraph langchain_community beautifulsoup4 requests pubmed-lookup python-dotenv

import os
import re
from typing import TypedDict, Optional, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate  
from langgraph.graph import StateGraph, END
from langchain_community.document_loaders import PubMedLoader
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# --- Configuration and Setup ---
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set. Please add it to your .env file.")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
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

# --- Helper Functions and Tools (Unchanged) ---

def find_all_urls(text: str) -> List[str]:
    """Finds all URLs in a given string and returns them as a list."""
    return re.findall(r'https?://\S+', text)

def find_doi_in_text(text: str) -> Optional[str]:
    """Finds the first Digital Object Identifier (DOI) in a given string."""
    doi_pattern = r'\b(10[.][0-9]{4,}(?:[.][0-9]+)*/(?:(?!["&\'<>])\S)+)\b'
    match = re.search(doi_pattern, text)
    return match.group(0) if match else None

def scrape_and_extract(url: str) -> tuple[str, None]:
    """
    A simple web scraper to extract the main text from a URL.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        text = soup.get_text()
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        if "abstract" in text.lower():
            print("   - Successfully scraped page content (including abstract).")
        else:
            print("   - Warning: Scraped content but could not find an 'abstract'. May be incomplete.")
        return text, None
    except requests.RequestException as e:
        print(f"   - Error during web scraping for {url}: {e}")
        return "", None

def pubmed_loader_tool(query: str) -> str:
    """Fetches documents from PubMed based on a query (ideally a DOI)."""
    try:
        loader = PubMedLoader(query=query, load_max_docs=1)
        documents = loader.load()
        if not documents:
            return "No documents found on PubMed for the given query."
        return "\n\n---\n\n".join([doc.page_content for doc in documents])
    except Exception as e:
        return f"An error occurred while fetching from PubMed: {e}"

# --- Agent Node Definitions ---

def task_router_node(state: AgentState):
    """Classifies the user's request to route it to the correct tool."""
    print("--- 1. ROUTING TASK ---")
    question = state['original_question']
    urls = find_all_urls(question)
    
    if len(urls) > 1:
        print(f"   - Found {len(urls)} URLs. Task classified as: multi_url_analysis")
        state['task_type'] = 'multi_url_analysis'
    elif len(urls) == 1 or '10.' in question:
        print("   - Found single URL or DOI. Task classified as: single_url_analysis")
        state['task_type'] = 'single_url_analysis'
    else:
        print("   - No URL or DOI found. Task is unsupported.")
        state['task_type'] = 'error'
    return state

# --- ORIGINAL SINGLE-URL RETRIEVER (UNCHANGED) ---
def smart_article_retriever_node(state: AgentState):
    """
    Intelligently retrieves academic article content for a SINGLE URL.
    """
    print("--- 2a. RETRIEVING SINGLE ARTICLE ---")
    question_text = state['original_question']
    url_match = re.search(r'https?://\S+', question_text)
    url = url_match.group(0).strip('">') if url_match else ""

    text_to_search = f"{question_text} {url}"
    doi = find_doi_in_text(text_to_search)
    
    final_context = ""
    
    if doi:
        print(f"   - Found DOI: {doi}. Querying PubMed...")
        pubmed_content = pubmed_loader_tool(doi)
        if "No documents found" not in pubmed_content and "error occurred" not in pubmed_content:
            final_context = pubmed_content
    
    if not final_context and url:
        print("   - PubMed failed or no DOI found. Falling back to web scraping...")
        scraped_text, _ = scrape_and_extract(url)
        if scraped_text and scraped_text.strip():
             final_context = scraped_text

    if not final_context:
        state['execution_error'] = "Fatal: Could not retrieve any content from the URL or open-access sources."
        state['task_type'] = 'error'
    else:
        state['data_context'] = final_context
        # Logic for default question
        question_without_urls = re.sub(r'https?://\S+', '', question_text).strip()
        if len(question_without_urls) < 15:
            state['question'] = 'Find the "amino acid Mutation", "Name of the pathogen", and "Name of the drug".'
        else:
            state['question'] = state['original_question']
    return state

# --- NEW MULTI-URL RETRIEVER ---
def multi_url_retriever_node(state: AgentState):
    """
    Intelligently retrieves content from MULTIPLE academic articles.
    """
    print("--- 2b. RETRIEVING MULTIPLE ARTICLES (BATCH MODE) ---")
    question_text = state['original_question']
    urls = find_all_urls(question_text)
    all_contexts = []
    
    for i, url in enumerate(urls):
        url = url.strip('">')
        print(f"\n--- Processing URL {i+1}/{len(urls)}: {url} ---")
        
        article_context = "" 
        
        text_to_search_doi = f"{url}" # Search only the URL first for precision
        doi = find_doi_in_text(text_to_search_doi)
        
        # Scrape first to also search for DOI in content
        scraped_text, _ = scrape_and_extract(url)
        if not doi and scraped_text:
            doi = find_doi_in_text(scraped_text)

        if doi:
            print(f"   - Found DOI: {doi}. Querying PubMed...")
            pubmed_content = pubmed_loader_tool(doi)
            if "No documents found" not in pubmed_content and "error occurred" not in pubmed_content:
                article_context = pubmed_content
        
        if not article_context and scraped_text:
            article_context = scraped_text
        
        if article_context:
            all_contexts.append(f"--- START OF ARTICLE FROM {url} ---\n{article_context}\n--- END OF ARTICLE FROM {url} ---")
        else:
            all_contexts.append(f"--- FAILED TO RETRIEVE CONTENT FOR {url} ---")

    state['data_context'] = "\n\n".join(all_contexts)
    # Logic for default question
    question_without_urls = re.sub(r'https?://\S+', '', question_text).strip()
    if len(question_without_urls) < 15:
        state['question'] = 'For each article, find the "amino acid Mutation", "Name of the pathogen", and "Name of the drug".'
    else:
        state['question'] = state['original_question']
        
    return state


def summarizer_qa_node(state: AgentState):
    """Uses the Gemini LLM to answer the user's question based on the retrieved context."""
    print("--- 3. ANALYZING AND ANSWERING ---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research assistant. Your task is to carefully read the provided scientific article context and answer the user's question based ONLY on the information within the text. For multiple articles, provide a separate answer for each. Be concise and precise."),
        ("user", "Article Context(s):\n\n{context}\n\nBased on the text above, please answer the following question:\n\nQuestion: {question}")
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

# Add all nodes
workflow.add_node("task_router", task_router_node)
workflow.add_node("smart_article_retriever_node", smart_article_retriever_node) # Original single URL node
workflow.add_node("multi_url_retriever_node", multi_url_retriever_node) # New multi URL node
workflow.add_node("summarizer_qa", summarizer_qa_node)

# Define the workflow edges
workflow.set_entry_point("task_router")
workflow.add_conditional_edges(
    "task_router",
    lambda state: state.get("task_type"),
    {
        "single_url_analysis": "smart_article_retriever_node",
        "multi_url_analysis": "multi_url_retriever_node",
        "error": END
    }
)

# Connect both retriever nodes to the same summarizer
workflow.add_edge("smart_article_retriever_node", "summarizer_qa")
workflow.add_edge("multi_url_retriever_node", "summarizer_qa")
workflow.add_edge("summarizer_qa", END)

app = workflow.compile()

# --- Main Execution Block (Unchanged) ---
if __name__ == "__main__":
    # Example usage for single URL
    # user_query = 'https://journals.asm.org/doi/10.1128/aac.00637-15'
    
    # Example usage for multiple URLs
    user_query = (
        'For these articles, find the pathogen and drug. '
        'URL 1: https://journals.asm.org/doi/10.1128/aac.00637-15 '
        'URL 2: https://www.nature.com/articles/s41586-020-2202-y'
    )
    
    initial_state = {"original_question": user_query}
    final_state = app.invoke(initial_state)
    
    if final_state.get("execution_error"):
        print("\n--- Agent Run Failed ---")
        print(f"Error: {final_state['execution_error']}")
    else:
        print("\n--- Agent Run Successful ---")
        print("Final Answer:")
        print(final_state['execution_result'])
