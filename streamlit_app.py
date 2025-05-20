import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import WebBaseLoader
from langchain_core.documents import Document
import requests
from bs4 import BeautifulSoup
import re
import time

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# Function to ensure all content is in Document format
def ensure_document(content, source="unknown"):
    if isinstance(content, Document):
        # Make sure source is in metadata
        if "source" not in content.metadata:
            content.metadata["source"] = source
        return content
    elif isinstance(content, str):
        return Document(page_content=content, metadata={"source": source})
    else:
        # Try to convert to string if possible
        try:
            return Document(page_content=str(content), metadata={"source": source})
        except:
            return Document(page_content="Content conversion error", metadata={"source": source})

# Function to scrape a website
def scrape_website(url):
    try:
        # Clean up URL if needed
        url = url.strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            url = 'https://' + url
        
        # Use WebBaseLoader to get content
        loader = WebBaseLoader(url)
        raw_docs = loader.load()
        
        # Process documents to ensure they're in the right format
        docs = []
        for doc in raw_docs:
            docs.append(ensure_document(doc, url))
        
        return docs, None
    except Exception as e:
        return None, str(e)

# Function to extract internal links
def extract_internal_links(docs):
    internal_links_by_domain = {}
    
    for doc in docs:
        url = doc.metadata.get("source", "")
        if not url:
            continue
            
        # Extract domain from URL
        domain_match = re.match(r'https?://(?:www\.)?([^/]+)', url)
        if not domain_match:
            continue
            
        base_domain = domain_match.group(1)
        
        # Try to parse HTML content
        try:
            soup = BeautifulSoup(doc.page_content, 'html.parser')
            links = soup.find_all('a', href=True)
            
            # Collect internal links
            for link in links:
                href = link['href']
                
                # Handle relative URLs
                if href.startswith('/'):
                    if url.endswith('/'):
                        href = url[:-1] + href
                    else:
                        href = url + href
                        
                # Check if it's an internal link
                if base_domain in href:
                    if base_domain not in internal_links_by_domain:
                        internal_links_by_domain[base_domain] = set()
                    internal_links_by_domain[base_domain].add(href)
        except:
            # Skip parsing errors
            continue
    
    return internal_links_by_domain

# Website URL input
website_urls_input = st.text_area("Enter website URLs (one per line):")

if st.button("Load and Process"):
    website_urls = website_urls_input.splitlines()
    st.session_state.loaded_docs = []  # Reset loaded docs
    
    progress_bar = st.progress(0)
    status_container = st.empty()
    
    for i, url in enumerate(website_urls):
        status_container.info(f"Loading: {url}")
        docs, error = scrape_website(url)
        
        if error:
            status_container.error(f"Error loading {url}: {error}")
        else:
            st.session_state.loaded_docs.extend(docs)
            status_container.success(f"Successfully loaded: {url} ({len(docs)} documents)")
        
        # Update progress
        progress = (i + 1) / len(website_urls)
        progress_bar.progress(progress)
        time.sleep(0.1)  # Small delay for UI updates
    
    status_container.empty()
    st.write(f"📄 Loaded content from {len(website_urls)} websites")
    
    # Extract and offer internal links
    if st.session_state.loaded_docs and st.checkbox("Would you like to crawl internal links as well?"):
        internal_links = extract_internal_links(st.session_state.loaded_docs)
        
        if not internal_links:
            st.write("No internal links found.")
        else:
            for domain, links in internal_links.items():
                external_urls = [url for url in website_urls if domain in url]
                link_list = list(links)
                
                # Filter out links that were already in the original URLs
                filtered_links = [link for link in link_list if link not in website_urls]
                
                if filtered_links:
                    st.write(f"Found {len(filtered_links)} internal links for {domain}")
                    selected_links = st.multiselect(f"Select internal links to crawl for {domain}", filtered_links)
                    
                    if selected_links:
                        link_progress = st.progress(0)
                        link_status = st.empty()
                        
                        for j, link in enumerate(selected_links):
                            link_status.info(f"Loading: {link}")
                            additional_docs, error = scrape_website(link)
                            
                            if error:
                                link_status.error(f"Error loading {link}: {error}")
                            else:
                                st.session_state.loaded_docs.extend(additional_docs)
                                link_status.success(f"Successfully loaded: {link}")
                            
                            # Update progress
                            link_progress.progress((j + 1) / len(selected_links))
                            time.sleep(0.1)  # Small delay for UI updates
                        
                        link_status.empty()

# LLM setup if documents are loaded
if st.session_state.loaded_docs:
    st.write(f"Total documents loaded: {len(st.session_state.loaded_docs)}")
    
    # Configure LLM
    with st.expander("Configure LLM Settings"):
        api_key = st.text_input("Enter your Groq API Key:", value="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", type="password")
        model_name = st.selectbox("Select Model:", ["llama-3.1-70b-versatile", "llama-3.1-8b-versatile", "llama-3.3-70b-versatile"], index=2)
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        top_p = st.slider("Top P:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    # Create text splitter and process documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )
    
    try:
        document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)
        st.write(f"Split into {len(document_chunks)} chunks for processing")
    except Exception as e:
        st.error(f"Error splitting documents: {str(e)}")
        document_chunks = st.session_state.loaded_docs  # Fallback to unsplit documents
    
    # Initialize LLM
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature, top_p=top_p)
    
    # Configure prompt template
    with st.expander("Customize Prompt Template"):
        default_prompt = """
        You are a knowledgeable assistant who analyzes websites and provides insights based on their content.
        Please answer questions based solely on the information provided in the context.
        
        <context>
        {context}
        </context>
        
        Question: {input}
        """
        
        prompt_template = st.text_area("Edit Prompt Template:", default_prompt, height=300)
    
    # Create prompt and document chain
    prompt = ChatPromptTemplate.from_template(prompt_template)
    document_chain = create_stuff_documents_chain(llm, prompt)
    st.session_state.retrieval_chain = document_chain

# Query interface
st.markdown("---")
st.subheader("Ask Questions About the Websites")
query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if not st.session_state.loaded_docs:
        st.warning("No documents loaded. Please load and process websites first.")
    elif not query:
        st.warning("Please enter a question to get an answer.")
    else:
        with st.spinner("Generating answer..."):
            try:
                # Create context string from all documents
                context_parts = []
                for doc in st.session_state.loaded_docs:
                    context_parts.append(doc.page_content)
                
                context = "\n\n".join(context_parts)
                
                # Generate response
                response = st.session_state.retrieval_chain.invoke({"input": query, "context": context})
                
                # Display response
                st.markdown("### Answer")
                if isinstance(response, dict) and 'answer' in response:
                    st.markdown(response['answer'])
                else:
                    st.markdown(response)
                
                # Show sources
                with st.expander("View Sources"):
                    sources = set()
                    for doc in st.session_state.loaded_docs:
                        if "source" in doc.metadata:
                            sources.add(doc.metadata["source"])
                    
                    if sources:
                        for source in sources:
                            st.markdown(f"- [{source}]({source})")
                    else:
                        st.write("No source information available.")
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
                st.error("Please check your Groq API key and try again.")
