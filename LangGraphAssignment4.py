## Import all required modules
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

import operator
from pydantic import BaseModel,Field
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.output_parsers import PydanticOutputParser

from langgraph.graph import StateGraph,END


# Load environment variables from .env file
load_dotenv()

# create model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# create embeddings
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")

# Load documents
loader = DirectoryLoader(
    "../data",
    glob="**/*.txt",
    show_progress=True,
    loader_cls=TextLoader
)
documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

new_docs = text_splitter.split_documents(documents)

docs = [doc.page_content for doc in new_docs]

# Create a vector store
db = Chroma.from_documents(
    documents=new_docs,
    embedding=embeddings
)
# Create a retriever from the vector store
retriever = db.as_retriever(
    search_kwargs={"k": 3}
)

# result = retriever.invoke("what type of cars available?")
# print(result)


# create pydantic model for topic selection
class TopicSelectionParser(BaseModel):
    Topic: str = Field(description = "selected topic")
    Reasoning: str = Field(description = "reasoning behind the topic selection")

parser = PydanticOutputParser(pydantic_object=TopicSelectionParser)

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]


# create a supervisor function

def function_supervisor(state: AgentState):
    question=state["messages"][-1]
    print("Question",question)
    template="""
    Your task is to classify the given user query into one of the following categories: [Maruthi Suzuki,Not Related].
    Only respond with the category name and nothing else.

    User query: {question}
    {format_instructions}
    """
    
    prompt= PromptTemplate(
        template=template,
        input_variable=["question"],
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    
    
    chain= prompt | model | parser
    
    response = chain.invoke({"question":question})
    
    print("Parsed response:", response)
    
    return {"messages": [response.Topic]}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# create a rag function
def rag_function(state: AgentState):
    print("-> RAG Call ->")
    
    question = state["messages"][0]
    
    prompt=PromptTemplate(
        template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:""",
        
        input_variables=['context', 'question']
    )
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    result = rag_chain.invoke(question)
    return  {"messages": [result]}

# create a function for web crawling
def web_crawler(state: AgentState):
    print("-> Web Crawler Call ->")
    
    question = state["messages"][0]
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful web crawler that searches the web for information."),
            ("human", "Search the web for information related to: {question}"),
        ]
    )
    
    chain = prompt | model | StrOutputParser()
    
    result = chain.invoke({"question": question})
    
    return {"messages": [result]}


# LLM Function
def function_llm(state:AgentState):
    print("-> LLM Call ->")
    question = state["messages"][0]
    # Normal LLM call
    complete_query = "Anwer the follow question with you knowledge of the real world. Following is the user question: " + question
    response = model.invoke(complete_query)
    return {"messages": [response.content]}


def router(state:AgentState):
    print("-> ROUTER ->")
    last_message=state["messages"][-1]
    print("last_message:", last_message)
    if "latest" in last_message.lower() or "current" in last_message.lower():
        return "Web Crawler Call"
    elif "maruthi suzuki" in last_message.lower():
        return "RAG Call"
    else:
        return "LLM Call"

# 4. Validation Node
def validate_output(state:AgentState):
    answer = state["messages"][-1]
    # Simple validation logic: e.g., length, keywords, confidence score, etc.
    if not answer or "I don't know" in answer:
        state["valid"] = False
    else:
        state["valid"] = True
    return state

workflow1=StateGraph(AgentState)
workflow1.add_node('Supervisor', function_supervisor)
workflow1.add_node('RAG', rag_function)
workflow1.add_node('Web Crawler', web_crawler)
workflow1.add_node('LLM', function_llm)
workflow1.add_node('Validate', validate_output)
workflow1.set_entry_point('Supervisor')
workflow1.add_conditional_edges(
    "Supervisor",
    router,
    {
        "RAG Call": "RAG",
        "LLM Call": "LLM",
        "Web Crawler Call": "Web Crawler"
    }
)
workflow1.add_edge("RAG", "Validate")
workflow1.add_edge("Web Crawler", "Validate")
workflow1.add_edge("LLM", "Validate")
workflow1.add_conditional_edges(
    "Validate",
    lambda state: "final_output" if state.get("valid", False) else "supervisor",
    {
        "final_output": END,
        "supervisor": "Supervisor"
    }
)


# Run the workflow
if __name__ == "__main__":
    # Example input
    input_data = {"messages": ["What are the latest car models available of Maruthi Suzuki?"]}
    app = workflow1.compile()
    # Run the workflow
    result = app.invoke(input_data)
    
    # Print the final output
    print("Final Output:", result["messages"][-1])



