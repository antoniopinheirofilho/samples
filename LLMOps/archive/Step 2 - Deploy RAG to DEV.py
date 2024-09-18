# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

#%pip install -U --quiet langchain==0.1.16 databricks-vectorsearch==0.22 pydantic==1.10.9 mlflow==2.12.1  databricks-sdk==0.28.0 cloudpickle>=2.1.0 "unstructured[pdf,docx]==0.10.30"

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain langchain-community databricks-vectorsearch pydantic mlflow  databricks-sdk cloudpickle "unstructured[pdf,docx]==0.10.30"

# COMMAND ----------

#%pip install pydantic -U

# COMMAND ----------

#%pip install -U --quiet langchain langchain-community databricks-vectorsearch pydantic==1.10.9 mlflow  databricks-sdk cloudpickle "unstructured[pdf,docx]==0.10.30"

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC # Parameters

# COMMAND ----------

# Model URI

model_name = "demo_prep.fine_tunning.fine_tuned_meta_llama_llama_2_70b_chat_hf"
model_version = "1"
model_uri = f"models:/{model_name}/{model_version}"

# Prompt

prompt_template = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {question}

Answer:
"""

# Model embedding - The embedding model used to generate the vector seatch index should be the same as the one used to embed the question.

embedding_model = "databricks-gte-large-en"

# Vector Search

vs_endpoint_name = "databricks_docs_vector_search"
vs_index_fullname = "demo_prep.vector_search_data.databricks_documentation_vs_index"

# Environment

dbx_host = "adb-3630608912046230.10.azuredatabricks.net"
dbx_token = ""
model_endpoint_name = "fine_tuned_foundation_model"

# Target UC
target_model_catalog = "llmops_dev"
target_model_schema = "model_schema"

# COMMAND ----------

# MAGIC %md
# MAGIC # Load fine tuned model from UC

# COMMAND ----------

import mlflow

# Set the registry URI to access Unity Catalog
mlflow.set_registry_uri('databricks-uc')

uc_model = mlflow.pyfunc.load_model(
  model_uri=model_uri
)

# COMMAND ----------

# MAGIC %md
# MAGIC # Create context retriever

# COMMAND ----------

from databricks.vector_search.client import VectorSearchClient
from langchain.vectorstores import DatabricksVectorSearch
from langchain.embeddings import DatabricksEmbeddings


# Test embedding Langchain model
#NOTE: your question embedding model must match the one used in the chunk in the previous model 
embedding_model = DatabricksEmbeddings(endpoint=embedding_model)
print(f"Test embeddings: {embedding_model.embed_query('What is GenerativeAI?')[:20]}...")

def get_retriever(persist_dir: str = None):
    #Get the vector search index
    vsc = VectorSearchClient()
    vs_index = vsc.get_index(
        endpoint_name=vs_endpoint_name,
        index_name=vs_index_fullname
    )

    # Create the retriever
    vectorstore = DatabricksVectorSearch(
        vs_index, text_column="content", embedding=embedding_model
    )
    # k defines the top k documents to retrieve
    return vectorstore.as_retriever(search_kwargs={"k": 2})

# COMMAND ----------

# MAGIC %md
# MAGIC # Create a chain with the model and the retriever

# COMMAND ----------

from langchain_core.runnables import chain

@chain
def custom_chain(prompt):

    prompt = {
        "messages":
            [
                {
                    "role": "system", 
                    "content": prompt.messages[0].content
                },
                {
                    "role": "user", 
                    "content": prompt.messages[1].content
                }
            ]
    }

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!PROMPT!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(prompt)
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    return uc_model.predict(prompt)

# COMMAND ----------

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatDatabricks
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

prompt_template = ChatPromptTemplate.from_messages(
    [  
        ("system", "You are a highly knowledgeable and professional Databricks Support Agent. Your goal is to assist users with their questions and issues related to Databricks. Answer questions as precisely and accurately as possible, providing clear and concise information. If you do not know the answer, respond with \"I don't know.\" Be polite and professional in your responses. Provide accurate and detailed information related to Databricks. If the question is unclear, ask for clarification.\n"),
        ("user", "Here is a documentation page that could be relevant: {context}. Based on this, answer the following question: {input}")
    ]
)

retriever = get_retriever()
question_answer_chain = create_stuff_documents_chain(custom_chain, prompt_template)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

#rag_chain = RetrievalQA.from_chain_type(
#    llm=custom_chain,
#    chain_type="stuff",
#    retriever=get_retriever(),
#    chain_type_kwargs={"prompt": prompt_template}
#)

#rag_chain = (
#        { 
#         "context": get_retriever(), 
#         "question": RunnablePassthrough()
#        }
#        | prompt_template
#        | custom_chain
#        | StrOutputParser()
#)

#rag_chain.invoke("How can I restart a cluster?")
#rag_chain.invoke({"input": "How can I restart a cluster?"})

# COMMAND ----------

# MAGIC %md
# MAGIC # Register model to UC

# COMMAND ----------

import os
from mlflow.models import infer_signature
import mlflow
import langchain

question = {"messages": [ {"role": "user", "content": "Question"}]}
answer =   {"messages": [ {"role": "assistant", "content": "Answer."}]}

mlflow.set_registry_uri("databricks-uc")
model_name = f"{target_model_catalog}.{target_model_schema}.basic_rag_demo"

# Log the model to MLflow
with mlflow.start_run(run_name="basic_rag_bot"):
    signature = infer_signature(question, answer)
    logged_chain_info = mlflow.langchain.log_model(
        rag_chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "langchain-community",
            "databricks-vectorsearch",
        ],
        input_example=question,
        signature=signature
      )

# COMMAND ----------


