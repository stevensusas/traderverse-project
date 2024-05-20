# This code defines the AI agent's abilities. It is built on LangChain and retrieves information from the Chroma vector database consructed
# from the CSV file. The agent can answer questions about the existing people in the database and add new entries to the database.

from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub

#Input your API key here!!!
key = ''

#Define the method for getting the Chatbot to respond to user questions, augmented by the vector database's existing information
def ask_question(question):

    # Initialize the chat model
    chat_model = ChatOpenAI(openai_api_key= key, model="gpt-3.5-turbo-0125", temperature=0)

    # Define the prompt template, which feeds the retrieved contexts from the database to the agent
    template_str = """Your job is to find the best people that fits the customer's networking needs. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.

    {context}
    """

    # Define the prompt to the system
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["context"],
            template=template_str,
        )
    )

    #Add the user's question to the prompt
    human_prompt = HumanMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template="{question}",
        )
    )

    # Combine the system and user prompts
    messages = [system_prompt, human_prompt]
    prompt_template = ChatPromptTemplate(
        input_variables=["context", "question"],
        messages=messages,
    )

    #Define the path to the Chroma database
    CHROMA_PATH = "chroma_data/"

    # Initialize the Chroma vector store
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(openai_api_key= key)
    )

    # Retrieve 10 closest matches from the vector store
    retriever  = vector_db.as_retriever(k=10)

    # Define the chain, which contains the retriever, prompt template, chat model, and output parser
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | chat_model
        | StrOutputParser()
    )

    # Invoke the review chain with the user's question
    response = chain.invoke(question)

    return response


#Define the method of adding new information to the vector database, which allows the agent to add new entries to the database
def add_to_DB(question):

    #Chat model initation
    chat_model = ChatOpenAI(openai_api_key= key, model="gpt-3.5-turbo-0125", temperature=0)

    #Define the prompt template for the system, which extracts the furniture and its description from the user's input to add to the database
    template_str = """Your job is to extract two things: the name and the description of the impressive people. The key is the name of the person, and the description is a brief description of the person's work. 
    Given the input, extract the key and the description, and output them as a pair of string splitted by comma. For example, if the input is 'I am Steven Su, and I am combining computer science and biology to accelerate life science discovery', the key is 'Steven Su' and the description is 'combining computer science and biology to accelerate life science discovery' and your output should be Steven Su, combining computer science and biology to accelerate life science discovery.
    {question}
    """
    # Define the prompt to the system with the user's question
    system_prompt = SystemMessagePromptTemplate(
        prompt=PromptTemplate(
            input_variables=["question"],
            template=template_str,
        )
    )

    messages = [system_prompt]

    # Combine the system prompt with the chat model and output parser
    prompt_template = ChatPromptTemplate(
        input_variables=["question"],
        messages=messages,
    )

    # Build the cahin with the prompt template, chat model, and output parser
    chain = (
        prompt_template
        | chat_model
        | StrOutputParser()
    )

    #invoke the chain with the user's question
    response = chain.invoke(question)

    # Extract the name and description of the furniture to add from the response
    name = response.split(',')[0].strip()
    description = response.split(',')[1].strip()

    #retrieves the database from the Chroma vector store
    CHROMA_PATH = "chroma_data/"
    vector_db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=OpenAIEmbeddings(openai_api_key=key)
    )
    
    # Add the new people to the database
    vector_db.add_texts(
        texts=[description], 
        metadatas=[{"product_name": name}]
    )

    #Save the databse with the new added furniture
    vector_db.persist()

#Define the tools that the agent can use from the two functions that we have defined earlier
tools = [
    Tool(
        name="ask_question",
        func= ask_question,
        description="""Useful when you need to answer questions
        about existing people in the databse and their descriptions.
        Not useful for adding new entries to the people database.
        Pass the entire question as input to the tool. For instance,
        if the question is "What are the best people I can find doing work in genomics?",
        the input should be "What are the best people I can find doing work in genomics?"
        """,
    ),
    Tool(
        name="add_to_database",
        func= add_to_DB,
        description="""Use when asked to add new people and their descriptions to the database. 
        This tool can only add new people to the database but cannot ask questions about the existing people.
        When asked to "add Steven Su, who is doing work in computational biology", the input should be key = Steven Su,  description = doing work in computational biology".
        """,
    ),
]

#Use existing agent prompts that instruct the agent on deciding which tool to use in its kit
agent_prompt = hub.pull("hwchase17/openai-functions-agent")

#Create the agent using the OpenAI model and the tools
agent_chat_model = ChatOpenAI(
    model="gpt-3.5-turbo-1106",
    temperature=0,
    openai_api_key = key,
)
alcov_agent = create_openai_functions_agent(
    llm=agent_chat_model,
    prompt=agent_prompt,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=alcov_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=True,
)