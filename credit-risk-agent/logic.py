import os
import logging
from datetime import datetime

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, StorageContext, Document, Settings, PromptTemplate
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.pinecone import PineconeVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.vector_stores import MetadataFilters, MetadataFilter
from llama_index.llms.databricks import Databricks


# Pinecone & Utils
from pinecone import Pinecone
from dotenv import load_dotenv

# Your existing utils
from s3_utils import upload_file_to_s3
from parsers import parse_document_local
from databricks_utils import get_customer_history_from_databricks
from custom_llm import CustomDatabricksLLM

# --- 1. CONFIGURATION ---
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Validate Env Vars
required_vars = ["PINECONE_API_KEY", "PINECONE_INDEX_NAME", "AWS_BUCKET_NAME"]
if not all(os.getenv(v) for v in required_vars):
    raise ValueError(f"Missing one of {required_vars} in .env")

# A. Setup Pinecone Connection
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# B. Connect LlamaIndex to Pinecone
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# C. Setup Embedding Model (all-MiniLM-L6-v2)
# We set this globally so we don't reload it constantly
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Instantiate your Custom LLM
databricks_llm = CustomDatabricksLLM(
    endpoint_name="credit-risk-analyst-endpoint", 
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN")
)

# Set it as the global LLM for LlamaIndex
Settings.llm = databricks_llm

# D. Configure Chunking (Specific to MiniLM's 256 limit)
# We use 256 token chunks with some overlap to respect the model's limit
Settings.node_parser = SentenceSplitter(chunk_size=256, chunk_overlap=20)


def process_and_index_files(uploaded_files, customer_id):
    """
    Orchestrates the RAG Ingestion Pipeline:
    1. Upload to S3 (Archival)
    2. Parse Text (Local OCR/PDF)
    3. Index to Pinecone (via LlamaIndex)
    """
    
    documents_buffer = [] # Holds LlamaIndex 'Document' objects
    
    for up_file in uploaded_files:
        try:
            # --- STEP 1: Archive to S3 ---
            logger.info(f"📂 Processing {up_file.name} into S3...")
            
            s3_key = upload_file_to_s3(up_file, customer_id)
            if not s3_key:
                logger.warning(f"⚠️ Skipping {up_file.name}: S3 Upload Failed")
                continue
                
            # --- STEP 2: Parse Document ---
            # Reset pointer because S3 upload might have moved it
            up_file.seek(0)
            
            # Extract raw text (Your existing logic)
            parsed_text = parse_document_local(up_file)
            
            if not parsed_text or len(parsed_text.strip()) == 0:
                logger.warning(f"⚠️ Skipping {up_file.name}: No text extracted")
                continue

            # --- STEP 3: Create LlamaIndex Document ---
            # We wrap the text + metadata into a standard object
            doc = Document(
                text=parsed_text,
                metadata={
                    "customer_id": customer_id,
                    "filename": up_file.name,
                    "s3_path": f"s3://{os.getenv('AWS_BUCKET_NAME')}/{s3_key}",
                    "type": "financial_doc",
                    "ingested_at": datetime.utcnow().isoformat()
                }
            )
            # Exclude text from being stored in metadata (saves Pinecone storage)
            # doc.excluded_embed_metadata_keys = ["s3_path", "ingested_at"]
            # doc.excluded_llm_metadata_keys = ["s3_path", "ingested_at"]
            
            documents_buffer.append(doc)
            logger.info(f"✅ Prepared {up_file.name} for indexing to Pinecone.")

        except Exception as e:
            logger.error(f"❌ Error processing file {up_file.name}: {str(e)}")

    # --- STEP 4: Batch Indexing ---
    if documents_buffer:
        logger.info(f"🚀 Indexing {len(documents_buffer)} documents into Pinecone...")
        
        # This single line handles Chunking -> Embedding -> Upserting
        VectorStoreIndex.from_documents(
            documents_buffer,
            storage_context=storage_context,
            show_progress=True
        )
        logger.info("🎉 All documents indexed successfully!")
        return True
    
    return False

from llama_index.core import PromptTemplate

# Define a default fallback template
DEFAULT_TEMPLATE_STR = (
    "Context information is below.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Given the context information and not prior knowledge, answer the query.\n"
    "Query: {query_str}\n"
    "Answer: "
)

def query_rag_pipeline(customer_id, user_query, custom_instructions=None):
    """
    Retrieves context and generates an answer using LlamaIndex.
    Args:
        customer_id (str): The customer ID to filter by.
        user_query (str): The search keywords (used to find docs in Pinecone).
        custom_instructions (str, optional): The prompt instructions for the LLM.
    """
    try:
        # 1. Load Index from Vector Store
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )

        # 2. Define Filters (Security: Only show this customer's data)
        filters = MetadataFilters(
            filters=[
                MetadataFilter(key="customer_id", value=customer_id)
            ]
        )

        # 3. Handle Prompt Template
        if custom_instructions:
            # Create a template that combines instructions + context
            new_template_str = (
                f"{custom_instructions}\n\n"
                "{context_str}\n"
                "---------------------\n"
                "Proceed with the analysis requested above." # <--- Command instead of "Answer this"
            )
            qa_template = PromptTemplate(new_template_str)
        else:
            qa_template = PromptTemplate(DEFAULT_TEMPLATE_STR)

        # 4. Create Query Engine
        query_engine = index.as_query_engine(
            filters=filters,
            similarity_top_k=5,
            text_qa_template=qa_template
        )

        # 5. Execute Query
        # Note: We pass 'user_query' (keywords) here. 
        # The 'custom_instructions' are already baked into the template above.
        response = query_engine.query(user_query)
        
        return str(response)

    except Exception as e:
        logger.error(f"❌ RAG Query Error: {e}")
        return "Sorry, I encountered an error retrieving your information."

def assess_credit_risk(customer_id, profile_text, uploaded_files=None):
    """
    Aggregates data from 3 sources (UI, Snowflake, Pinecone) 
    and generates a decision.
    """
    logger.info(f"🧠 Generating Credit Risk Assessment for {customer_id}...")

    # 1. Index New Files (Pinecone)
    if uploaded_files:
        process_and_index_files(uploaded_files, customer_id)

    # 2. Fetch Databricks History (SQL)
    snowflake_history = get_customer_history_from_databricks(customer_id)
    logger.info(f"📊 Data from Databricks, {snowflake_history}")

    # 3. Construct the "Master Context" Prompt
    # We combine UI Profile + Snowflake History here.
    # Pinecone Docs will be added by LlamaIndex via {context_str}
    
    master_instructions = (
        f"You are a Senior Credit Risk Analyst.\n"
        "Using the applicant profile, their payment history, and extracted information from uploaded financial documents, make a loan approval decision.\n" "Important: all the currency values in the SOURCES are in Rupees (Rs). Do not consider anything as USD.\n"
        "\n\n"
        "Your Output Report MUST be structured as follows:\n"
        "1. **Risk Category:** (Low / Medium / High)\n"
        "2. **Approval Status:** (Approved / Rejected / Pending Review)\n"
        "3. **Sanctioned Loan Amount: (in Rs)**\n"
        "4. **Reasoning:** Clear reasoning referencing the evidence from all sources.\n\n"
        f"=== SOURCE 1: APPLICANT PROFILE ===\n"
        f"{profile_text}\n\n"
        
        f"=== SOURCE 2: HISTORICAL CREDIT DATA ===\n"
        f"{snowflake_history}\n\n"
        
        f"=== SOURCE 3: FINANCIAL DOCUMENTS ===\n"
    )

    # 4. Define Search Keywords (for Pinecone retrieval)
    search_keywords = "salary payslip deposit rent loan repayment overdraft fee bank statement transaction"
    logger.info("✅ Instructions formatted. Calling RAG Pipeline...")

    # 5. Call Pipeline
    assessment_text = query_rag_pipeline(
        customer_id, 
        user_query=search_keywords, 
        custom_instructions=master_instructions
    )

    logger.info("✅ Analysis Complete.")
    return assessment_text

def extract_risk_category(text: str) -> str:
    """Extracts risk category from LLM response."""
    if "High" in text:
        return "High"
    elif "Medium" in text:
        return "Medium"
    else:
        return "Low"


def extract_approval_status(text: str) -> str:
    """Extracts approval status from LLM response."""
    text_upper = text.upper()
    if "APPROVED" in text_upper:
        return "Approved"
    elif "REJECT" in text_upper:
        return "Rejected"
    else:
        return "Pending Review"


def extract_loan_amount(text: str) -> str:
    """Extracts a loan sanction amount from the LLM response."""
    import re

    patterns = [
        r"approved\s+loan\s+amount[:\s]+\$?([\d,]+(?:\.\d{2})?)",
        r"sanction(?:ed)?\s+loan\s+amount[:\s]+\$?([\d,]+(?:\.\d{2})?)",
        r"sanction(?:ed)?\s+amount[:\s]+\$?([\d,]+(?:\.\d{2})?)",
        r"approved\s+(?:for\s+)?\$?([\d,]+(?:\.\d{2})?)"
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return f"${match.group(1)}"

    decision_section = (
        text.split("**Decision:**")[-1] if "**Decision:**" in text else text
    )
    decision_section = (
        decision_section.split("**Recommendation:**")[0]
        if "**Recommendation:**" in decision_section
        else decision_section
    )

    sanction_line_match = re.search(
        r"Loan Sanction Amount[:\s]+\$?([\d,]+(?:\.\d{2})?)",
        decision_section,
        re.IGNORECASE,
    )
    if sanction_line_match:
        return f"${sanction_line_match.group(1)}"

    all_amounts = re.findall(r"\$[\d,]+(?:\.\d{2})?", decision_section)
    if all_amounts:
        return all_amounts[-1]

    return "Check Reasoning"
