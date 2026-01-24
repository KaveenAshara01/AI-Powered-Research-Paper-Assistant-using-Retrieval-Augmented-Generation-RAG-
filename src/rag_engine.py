import torch
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from src import config

def load_llm():
    """
    Loads the Mistral-7B model with 4-bit quantization for efficient inference.
    """
    print(f"Loading LLM: {config.LLM_MODEL_NAME}...")
    
    # Configure 4-bit quantization to reduce memory usage
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto", # Automatically distributes to GPU/CPU
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading model with quantization: {e}")
        print("Falling back to CPU-friendly loading (might be slow)...")
        tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            config.LLM_MODEL_NAME,
            device_map="cpu",
            trust_remote_code=True
        )

    # Create a generation pipeline
    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        temperature=config.TEMPERATURE,
        do_sample=True,
        top_k=config.TOP_K_RETRIEVAL,
        top_p=0.95,
        repetition_penalty=1.15
    )

    # Wrap in LangChain interface
    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

def get_retrieval_chain():
    """
    Creates the RAG chain using the vector store and LLM.
    """
    # 1. Load Vector Store
    print(f"Loading vector store from {config.VECTOR_DB_DIR}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}, # Keep embeddings on CPU for now
        encode_kwargs={"normalize_embeddings": True}
    )
    
    try:
        vector_store = FAISS.load_local(config.VECTOR_DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return None

    # 2. Load LLM
    llm = load_llm()

    # 3. Define Prompt Template
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    Context: {context}
    
    Question: {question}
    
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)

    # 4. Create Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": config.TOP_K_RETRIEVAL}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return qa_chain

if __name__ == "__main__":
    # Test the chain logic
    chain = get_retrieval_chain()
    if chain:
        res = chain.invoke("What is this research paper about?") # Updated invoke method for newer LangChain
        print(res['result'])
