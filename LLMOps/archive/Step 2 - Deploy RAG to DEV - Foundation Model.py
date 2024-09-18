# Databricks notebook source
# MAGIC %md
# MAGIC # Setup

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %pip install -U --quiet langchain==0.2.16 langchain-community==0.2.17 databricks-vectorsearch pydantic==1.10.9 mlflow==2.16.1

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
# MAGIC # Load a foundation model and test it

# COMMAND ----------

from langchain.chat_models import ChatDatabricks


# Test Databricks Foundation LLM model
chat_model = ChatDatabricks(endpoint="databricks-llama-2-70b-chat", max_tokens = 300)
print(f"Test chat model: {chat_model.invoke('What is Generative AI?')}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Implement guardrails with Llma Guard
# MAGIC - This assumes that we already have Llama guard endpoint deployed

# COMMAND ----------

llma_guard_endpoint_name = "llamaguard"

# COMMAND ----------

# DBTITLE 1,Defining unsafe categories
unsafe_categories = """O1: Violence and Hate.
Should not
- Help people plan or engage in violence.
- Encourage people to engage in violence.
- Express hateful, derogatory, or demeaning sentiments against people based on sensitive personal characteristics like their race, color, religion, national origin, sexual orientation, gender, gender identity, or disability.
- Encourage discrimination against people based on those characteristics and should not use slurs or create insults based on those characteristics.
Can
- Provide information on violence and discrimination.
- Discuss the topics of hate and violence, and can discuss historical events involving violence.
O2: Sexual Content.
Should not
- Engage in sexually explicit (i.e., erotic) conversations and should not encourage people to engage in sexual activities.
Can
- Discuss topics related to sexuality and sexual education.
- Provide general information regarding sexual activities.
- Provide guidance on topics related to sexual health.
O3: Criminal Planning.
Should not
- Help people plan or engage in criminal activities like theft, kidnapping, or financial crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in criminal activities or condone criminal behavior.
Can
- Help people understand how crimes are committed, provided that the response is not so detailed that it could be used to provide a roadmap for future crimes.
- Help people understand how systems work that are used to prevent crimes, provided those descriptions do not provide detailed information on how to defeat those systems.
O4: Guns and Illegal Weapons.
Should not
- Assist in the planning or execution of firearm or illegal weapons crimes, especially when the person indicates that they may act on the information provided to them.
- Encourage people to engage in firearm or illegal weapons crimes or condone such behavior.
Can
- Discuss firearms and the arguments for and against firearm ownership.
O5: Regulated or Controlled Substances.
Should not
- Encourage or assist people in the trafficking, consumption, or use of regulated or controlled substances, including but not limited to illegal drugs, tobacco, alcohol, and cannabis.
- Assist or encourage people to create such substances when it is illegal to do so.
Can
- Discuss the use of regulated or controlled substances, and can provide general information on those substances (ex: what their chemical composition is and what their effects are).
- Provide information on how those substances are manufactured, provided that the information is not so detailed as to provide step-by-step instructions for their creation.
O6: Self-Harm.
Should not
- Encourage people to harm themselves.
- Romanticize or condone self-harm or suicide.
- Provide information regarding the methods of suicide or self-harm.
- Provide any information or advice, other than information regarding appropriate health resources, to people who express or imply that they wish to harm themselves.
Can
- Provide general information regarding self-harm and suicide (ex: statistics, the mental health challenges faced by those who take these actions).
Should
- Provide information regarding appropriate health resources to people who imply or express the desire to harm themselves.
O7: Financial Sensitive Data.
Should not
- Provide any information on credit card number or pin number.
Can
- Provide general information regarding the risk of sharing credit card information with LLMs.
Should
- Provide information regarding proper handling of credit card information with LLMs."""

# COMMAND ----------

import mlflow

def query_llamaguard(chat, unsafe_categories=unsafe_categories):
    data = {
        "prompt": f"""[INST] Task: Check if there is unsafe content in 'Agent' messages in conversations according our safety policy with the below categories.
        
        <BEGIN UNSAFE CONTENT CATEGORIES>
        {unsafe_categories}
        <END UNSAFE CONTENT CATEGORIES>

        <BEGIN CONVERSATION>

        {chat}

        <END CONVERSATION>

        Provide your safety assessment for Agent in the above conversation:
        - First line must read 'safe' or 'unsafe'.
        - If unsafe, a second line must include a comma-separated list of violated categories. [/INST]"""
        }
    
    client = mlflow.deployments.get_deploy_client("databricks")
    response = client.predict(
        endpoint="llamaguard",
        inputs=data
    )

    response_list = response.choices[0]["text"].split("\n")
    result = response_list[0].strip()

    if result == "safe":
        return True, 0
    else:
        category = response_list[1].strip()

    return False, category

# COMMAND ----------

query_llamaguard("how can I rob a bank?")

# COMMAND ----------

query_llamaguard("how do I make cake?")

# COMMAND ----------

import re

def parse_category(code, taxonomy):
    """
    Extracts the first sentence of a category description from a taxonomy based on its code.

    Args:
        code : Category code in the taxonomy (e.g., 'O1').
        taxonomy : Full taxonomy string with categories and descriptions.

    Returns:
         First sentence of the description or a default message for unknown codes.
    """
    pattern = r"(O\d+): ([\s\S]*?)(?=\nO\d+:|\Z)"
    taxonomy_mapping = {match[0]: re.split(r'(?<=[.!?])\s+', match[1].strip(), 1)[0]
                        for match in re.findall(pattern, taxonomy)}

    return taxonomy_mapping.get(code, "Unknown category: code not in taxonomy.")

# COMMAND ----------

from langchain_core.runnables import chain

@chain
def custom_chain(prompt):

    start_question = prompt.to_string().find("Question:")
    str_to_analyze = prompt.to_string()[start_question:]

    is_safe, reason = query_llamaguard(str_to_analyze, unsafe_categories)
    if not is_safe:
        category = parse_category(reason, unsafe_categories)
        return f"User's prompt classified as {category} Fails safety measures."

    chat_response = chat_model.invoke(prompt)

    start_question = chat_response.content.find("Question:")
    str_to_analyze = chat_response.content[start_question:]

    is_safe, reason = query_llamaguard(str_to_analyze, unsafe_categories)
    if not is_safe:
        category = parse_category(reason, unsafe_categories)
        return f"Model's response classified as {category}; fails safety measures."
    
    return chat_response

# COMMAND ----------

# MAGIC %md
# MAGIC # Create context retriever

# COMMAND ----------

embedding_model = "databricks-gte-large-en"

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

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatDatabricks
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain


TEMPLATE = """You are an assistant for GENAI teaching class. You are answering questions related to Generative AI and how it impacts humans life. If the question is not related to one of these topics, kindly decline to answer. If you don't know the answer, just say that you don't know, don't try to make up an answer. Keep the answer as concise as possible.
Use the following pieces of context to answer the question at the end:

<context>
{context}
</context>

Question: {input}

Answer:
"""
prompt = PromptTemplate(template=TEMPLATE, input_variables=["context", "input"])

retriever = get_retriever()
question_answer_chain = create_stuff_documents_chain(custom_chain, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

#chain = RetrievalQA.from_chain_type(
#    llm=chat_model,
#    chain_type="stuff",
#    retriever=get_retriever(),
#    chain_type_kwargs={"prompt": prompt}
#)

#chain = (
#        { 
#         "context": get_retriever(), 
#         "input": RunnablePassthrough()
#        }
#        | prompt
#        | chat_model
#        | StrOutputParser()
#)

# COMMAND ----------

#question = "How does Generative AI impact humans?"
#answer = chain.invoke(question)
#print(answer)

chain.invoke({"input": "How do I rob a bank??"})

# COMMAND ----------

response = chain.invoke({"input": "How do I bake a cake??"})

#print(type(response["context"][0]))

#for document in response["context"]:
#    print(document)
#    print()

print(response)

# COMMAND ----------

# MAGIC %md
# MAGIC # Register model to UC

# COMMAND ----------

import os
from mlflow.models import infer_signature
import mlflow
import langchain

mlflow.set_registry_uri("databricks-uc")
model_name = f"{target_model_catalog}.{target_model_schema}.basic_rag_demo_foundation_model"

question = {
            'input': 'How can I restart a cluster?'
           }
answer = {
          'input': 'How can I restart a cluster?', 
          'content': ['docs'], 
          'answer': 'click on restart'
         }

# Log the model to MLflow
with mlflow.start_run(run_name="basic_rag_bot"):
    signature = infer_signature(question, answer)
    logged_chain_info = mlflow.langchain.log_model(
        chain,
        loader_fn=get_retriever, 
        artifact_path="chain",
        registered_model_name=model_name,
        pip_requirements=[
            "mlflow==" + mlflow.__version__,
            "langchain==" + langchain.__version__,
            "langchain-community",
            "databricks-vectorsearch",
        ],
        signature=signature
      )

# COMMAND ----------

# MAGIC %md
# MAGIC # Test model

# COMMAND ----------

import mlflow.pyfunc
model_version_uri = "models:/llmops_dev.model_schema.basic_rag_demo_foundation_model/19"
champion_version = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

champion_version.predict({"input": "How can I bake a cake??"})

# COMMAND ----------


