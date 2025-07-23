import os
import asyncio
import json
import requests

from typing import Annotated

from semantic_kernel import Kernel
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from semantic_kernel.agents.chat_completion.chat_completion_agent import ChatCompletionAgent
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, AzureChatPromptExecutionSettings

from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
from semantic_kernel.contents.utils.author_role import AuthorRole

from dotenv import load_dotenv
load_dotenv(".env")

# === ENV ===
deployment = "gpt-4o-mini"
subscription_key = os.environ.get("AZURE_OPENAI_KEY")
endpoint = os.environ.get("AZURE_OPENAI_RESOURCE")
embedding_endpoint = os.environ.get('AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE')
headers = {
    "Content-Type": "application/json",
    "Authorization": os.environ.get('AZURE_OPENAI_EMBEDDING_MODEL_RESOURCE_KEY')
}
search_endpoint = os.environ.get('COG_SEARCH_ENDPOINT')
admin_key = os.environ.get('COG_SEARCH_ADMIN_KEY')


# === Search Plugin ===
class SearchPlugin:
    def __init__(self, text_index_name="pdf-economic-summary", table_index_name="pdf-economic-summary-tables",
                 image_index_name="pdf-economic-summary-images"):
        self.search_client_text = SearchClient(
            endpoint=search_endpoint,
            index_name=text_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.search_client_table = SearchClient(
            endpoint=search_endpoint,
            index_name=table_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.search_client_image = SearchClient(
            endpoint=search_endpoint,
            index_name=image_index_name,
            credential=AzureKeyCredential(admin_key)
        )
        self.embedding_endpoint = embedding_endpoint
        self.headers = headers

    async def get_embedding(self, text):
        def sync_post():
            response = requests.post(
                url=self.embedding_endpoint,
                headers=self.headers,
                json={"input": text}
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]

        return await asyncio.to_thread(sync_post)

    async def _search(self, query, client, select, top_k=10, filter=None):
        vector = await self.get_embedding(query)
        vector_query = VectorizedQuery(vector=vector, k_nearest_neighbors=top_k, fields="contentVector")
        results = client.search(
            search_text=query,
            vector_queries=[vector_query],
            select=select,
            top=top_k,
            filter=filter,
        )
        return json.dumps([
            {
                "page": doc.get("page", "N/A"),
                "filename": doc.get("doc_name", "unknown.txt"),
                "content": doc.get("content", ""),
                "table": doc.get("table", "N/A"),
                "figure": doc.get("figure", "N/A")
            } for doc in results
        ], ensure_ascii=False, indent=2)

    @kernel_function(description="Search document text content")
    async def search_text_content(self, query: Annotated[str, "User query"], filter=None, top_k=10) -> Annotated[
        str, "Search results"]:
        return await self._search(query, self.search_client_text, select=["content", "page", "doc_name"], filter=filter,
                                  top_k=top_k)

    @kernel_function(description="Search table data")
    async def search_table_content(self, query: Annotated[str, "User query"], filter=None, top_k=10) -> Annotated[
        str, "Search results"]:
        return await self._search(query, self.search_client_table, select=["content", "page", "table", "doc_name"],
                                  filter=filter, top_k=top_k)

    @kernel_function(description="Search image data")
    async def search_image_content(self, query: Annotated[str, "User query"], filter=None, top_k=10) -> Annotated[
        str, "Search results"]:
        return await self._search(query, self.search_client_image, select=["content", "page", "figure", "doc_name"],
                                  filter=filter, top_k=top_k)


# === System Prompt ===
system_prompt_RAG = f"""You are a helpful, professional financial assistant. Answer **only** from the provided data — no external knowledge or assumptions.

  Instructions:
  - Use clear, simple English.
  - Be concise: no greetings, filler, or extra commentary.
  - Include all numbers. Cite page (e.g., "หน้า 6") and table (e.g., "ตารางที่ 2") if available.
  - Do not omit any numbers or quantitative details.
  - Combine image and text data only if they add different value.
  - Refer to the document by its `filename`.
  - Treat “อเมริกา”, “สหรัฐฯ”, and “สหรัฐ” as the same.
  - Ignore figure numbers.

  Format must be clean and machine-readable.

  Example:
  From p. 6 Table 4 and image on p. 11 of monthly-summary:
  - Thai GDP in Q1/2025 grew 3.1% YoY, driven by 13.8% export growth
  - Domestic demand remains weak; tourism is slowing"""

system_prompt_KEYWORD = f"""  You are a keyword extraction assistant for a search engine.

  Given a user question, extract the most relevant keywords or key phrases, and return them as a single plain string that can be directly passed into Azure Cognitive Search with query_type='simple'.

  Add keyword from user message of thechat history if relevant to the new question.

  Guidelines:
  - Use lowercase.
  - Separate each keyword or phrase with a space.
  - Use double quotes for phrases only if needed (e.g., "interest rate").
  - Do not return a list or any extra formatting — just the search string.

  Example 1:

  User Query: What’s the latest update on Thailand inflation and interest rates?
  Search String: thailand inflation "interest rates"

  Example 2:

  Prior User Query: US Stock?
  Current User Query: Chinese?
  Search String: chinese "stock"

  Only return the search string."""


# === Agent & Plugin Constructor ===
def get_keyword_extractor_agent(kernel: Kernel) -> ChatCompletionAgent:
    # Add AzureChatCompletion service only if not added yet
    if "keyword_chat_service" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="keyword_chat_service",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )

    settings = AzureChatPromptExecutionSettings(
        service_id="keyword_chat_service",
        temperature=0.3,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="keyword-extractor-agent",
        instructions=system_prompt_KEYWORD,
    )
    return agent


# === Agent & Plugin Constructor ===
def get_mm_rag_agent(kernel: Kernel) -> ChatCompletionAgent:
    # Add AzureChatCompletion service only if not added yet
    if "rag_agent" not in kernel.services:
        kernel.add_service(
            AzureChatCompletion(
                service_id="rag_agent",
                deployment_name=deployment,
                api_key=subscription_key,
                endpoint=endpoint,
            )
        )
    settings = AzureChatPromptExecutionSettings(
        service_id="rag_agent",
        temperature=0.1,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
    )
    agent = ChatCompletionAgent(
        kernel=kernel,
        arguments=KernelArguments(settings=settings),
        name="searchservivce-mm-rag-agent",
        instructions=system_prompt_RAG,
    )
    return agent


def get_mm_search_plugin(text_index_name="pdf-economic-summary", table_index_name="pdf-economic-summary-tables",
                         image_index_name="pdf-economic-summary-images"):
    return SearchPlugin(text_index_name=text_index_name, table_index_name=table_index_name,
                        image_index_name=image_index_name)


import asyncio
from semantic_kernel.contents.chat_message_content import ChatMessageContent


# === Helper to call one sub-agent ===
async def run_mmrag_agent(agents, search, user_query, search_keywords, filter=None, top_k=10):
    context_text, context_table, context_image = await asyncio.gather(
        search.search_text_content(search_keywords, filter=filter, top_k=top_k),
        search.search_table_content(search_keywords, filter=filter, top_k=5),
        search.search_image_content(search_keywords, filter=filter, top_k=4),
    )

    user_prompt = f"""Use the following JSON context to answer the question:

        Context text data:
        {context_text}

        Context table data:
        {context_table}

        Context image data:
        {context_image}

        Question: {user_query}
        """

    user_message = ChatMessageContent(role=AuthorRole.USER, content=user_prompt)

    response_text = ""
    async for response in agents.invoke(messages=[user_message]):
        response_text = str(response)

    return response_text


# === Final Orchestrator ===
async def get_agent_response(user_query: str, pdf_rag_agent, keyword_extractor_agent, pdf_search) -> tuple[str, str]:
    keyword_agent_user_prompt = f"Extract keywords from this query: {user_query}"
    keyword_agent_message = ChatMessageContent(role=AuthorRole.USER,
                                               content=keyword_agent_user_prompt)  # === Token count for input to keyword agent

    search_keywords = user_query
    async for response in keyword_extractor_agent.invoke(messages=[keyword_agent_message]):
        search_keywords = str(response)
    response_1, response_2, response_3 = await asyncio.gather(
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords,
                        filter="key_prefix eq 'monthlystandpoint'", top_k=10),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords, filter="key_prefix eq 'ktm'", top_k=20),
        run_mmrag_agent(pdf_rag_agent, pdf_search, user_query, search_keywords, filter="key_prefix eq 'kcma'",
                        top_k=40),
    )

    # Combine with source labels
    final_text_response = (
        f"from monthlystandpoint: {response_1}\n\n"
        f"from ktm: {response_2}\n\n"
        f"from kcma: {response_3}"
    )

    return final_text_response


async def main(question):
    kernel = Kernel()
    pdf_rag_agent = get_mm_rag_agent(kernel)
    keyword_extractor_agent = get_keyword_extractor_agent(kernel)
    pdf_search = get_mm_search_plugin(
        text_index_name="pdf-economic-summary",
        table_index_name="pdf-economic-summary-tables",
        image_index_name="pdf-economic-summary-images"
    )
    return await get_agent_response(question, pdf_rag_agent, keyword_extractor_agent, pdf_search)


# print(asyncio.run(main()))