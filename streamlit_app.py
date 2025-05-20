import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
import re

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

website_urls_input = st.text_area("Enter website URLs (one per line):")

if st.button("Load and Process"):
    website_urls = website_urls_input.splitlines()
    loaded_docs = []
    
    with st.spinner(f"Loading {len(website_urls)} websites..."):
        for url in website_urls:
            try:
                # Clean up URL if needed
                url = url.strip()
                if not (url.startswith('http://') or url.startswith('https://')):
                    url = 'https://' + url
                    
                st.info(f"Loading: {url}")
                
                # Load the website
                loader = WebBaseLoader(url)
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata["source"] = url
                
                loaded_docs.extend(docs)
                st.success(f"Successfully loaded: {url}")
                
            except Exception as e:
                st.error(f"Error loading {url}: {str(e)}")
    
    st.session_state.loaded_docs = loaded_docs
    st.write(f"📄 Loaded {len(loaded_docs)} documents from {len(website_urls)} websites")
    
    # Extract internal links and offer to load them too
    if loaded_docs and st.checkbox("Would you like to crawl internal links as well?"):
        internal_links_by_domain = {}
        
        for doc in loaded_docs:
            url = doc.metadata["source"]
            # Extract domain from URL
            domain_match = re.match(r'https?://(?:www\.)?([^/]+)', url)
            if domain_match:
                base_domain = domain_match.group(1)
                
                # Parse the document content for links
                soup = BeautifulSoup(doc.page_content, 'html.parser')
                links = soup.find_all('a', href=True)
                
                # Collect internal links
                for link in links:
                    href = link['href']
                    # Make absolute URL if relative
                    if href.startswith('/'):
                        if url.endswith('/'):
                            href = url[:-1] + href
                        else:
                            href = url + href
                    # Check if it's an internal link
                    if base_domain in href and href not in website_urls:
                        if base_domain not in internal_links_by_domain:
                            internal_links_by_domain[base_domain] = set()
                        internal_links_by_domain[base_domain].add(href)
        
        # Allow selection of internal links to crawl
        if internal_links_by_domain:
            for domain, links in internal_links_by_domain.items():
                st.write(f"Found {len(links)} internal links for {domain}")
                selected_links = st.multiselect(f"Select internal links to crawl for {domain}", list(links))
                
                if selected_links:
                    with st.spinner(f"Loading {len(selected_links)} additional pages..."):
                        for link in selected_links:
                            try:
                                loader = WebBaseLoader(link)
                                additional_docs = loader.load()
                                
                                for doc in additional_docs:
                                    doc.metadata["source"] = link
                                
                                st.session_state.loaded_docs.extend(additional_docs)
                                st.success(f"Successfully loaded: {link}")
                            except Exception as e:
                                st.error(f"Error loading {link}: {str(e)}")
                    
                    st.write(f"📄 Now loaded {len(st.session_state.loaded_docs)} documents in total")

# LLM and Text Splitting setup
if st.session_state.loaded_docs:
    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        length_function=len,
    )

    document_chunks = text_splitter.split_documents(st.session_state.loaded_docs)
    st.write(f"Split into {len(document_chunks)} chunks for processing")

    # LLM setup
    with st.expander("Configure LLM Settings"):
        api_key = st.text_input("Enter your Groq API Key:", value="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", type="password")
        model_name = st.selectbox("Select Model:", ["llama-3.1-70b-versatile", "llama-3.1-8b-versatile", "llama-3.3-70b-versatile"], index=2)
        temperature = st.slider("Temperature:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
        top_p = st.slider("Top P:", min_value=0.0, max_value=1.0, value=0.2, step=0.1)
    
    llm = ChatGroq(groq_api_key=api_key, model_name=model_name, temperature=temperature, top_p=top_p)

    # Prompt template - can be customized
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

    prompt = ChatPromptTemplate.from_template(prompt_template)

    # Create document chain
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
                # Prepare context from loaded documents
                context = "\n".join([doc.page_content for doc in st.session_state.loaded_docs])
                
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
                    sources = set([doc.metadata["source"] for doc in st.session_state.loaded_docs])
                    for source in sources:
                        st.markdown(f"- [{source}]({source})")
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
