import os
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables import RunnableConfig
from langchain_together import Together
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent, create_structured_chat_agent
from langchain.tools import tool
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from sentence_transformers import SentenceTransformer
from typing import List, Any
import streamlit as st

st.set_page_config(page_title="LangChain: Chat with search", page_icon="🦑")
st.title("🦑 Together AI & Langchain: Milvus Vectore Store Agent")

load_dotenv()
# Replace uri and token with your own
client = MilvusClient(
    uri=os.getenv("MILVUS_ENDPOINT"),
    token=os.getenv("MILVUS_API_KEY") # API key or a colon-separated cluster username and password
)

# Connect to cluster
connections.connect(
  alias='default',
  # Public endpoint obtained from Zilliz Cloud
  uri=os.getenv("MILVUS_ENDPOINT"),
  # API key or a colon-separated cluster username and password
  token=os.getenv("MILVUS_API_KEY")
)

collection = Collection("News_2019")      # Get an existing collection.
collection.load()


# Cargar el modelo de Sentence Transformers y moverlo a CUDA si está disponible
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)

@tool('milvus_chilean_consult')
def milvus_chilean_consult(query: dict) -> List[Any]:
    "Gives a list of document found by the query"
    query = query['title']
    vect = model.encode(query)
    results = collection.search(
        data=[vect], 
        anns_field="vector", 
        param={"metric_type": "L2", "params": {"nprobe": 1}}, 
        limit=15, 
        expr=None,
        consistency_level="Strong"
    )
    ids = results[0].ids
    res = collection.query(
        expr =f"""id in {ids}""" , 
        output_fields = ["body"],
        consistency_level="Strong"
    )
    return res

DEFAULT_TEMPLATE = """You are a helpfull AI assistant that helps to makes querys to a vectore store database given a human consult of a specific topic.
The AI gives information and explanation from the results of the query.
The AI is talkative and provides a lot of specific details of its context.
If the AI does not know the answer to the question, it will truthfully sai it does not know.
Both answer and queries should always be in spanish, never in other lenguage.

You have access to the following tools:
{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
    "action": $TOOL_NAME,
    "action_input": $INPUT
}}```

Follow this format:

Question: input question to answer
Thought: consider previous and subsequent steps
Action:
```
$JSON_BLOB
```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
        Action:
        ```
            {{
                "action": "Final Answer",
                "action_input": "Final response to human"
            }}
            
For example, if you want to use a tool to make a query to the vectore store database, your $JSON_BLOB might look like this:

```
{{
    'action': 'vector_store_query_json', 
    'action_input': {{'query': 'Example query'}}
}}```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation'
"""




prompt = hub.pull("hwchase17/structured-chat-agent")
prompt[0].prompt.template = DEFAULT_TEMPLATE

msgs = StreamlitChatMessageHistory()
memory = ConversationBufferMemory(
    chat_memory=msgs, return_messages=True, memory_key="chat_history", output_key="output"
)
if len(msgs.messages) == 0 or st.sidebar.button("Reset chat history"):
    msgs.clear()
    msgs.add_ai_message("How can I help you?")
    st.session_state.steps = {}

avatars = {"human": "user", "ai": "assistant"}
for idx, msg in enumerate(msgs.messages):
    with st.chat_message(avatars[msg.type]):
        # Render intermediate steps if any were saved
        for step in st.session_state.steps.get(str(idx), []):
            if step[0].tool == "_Exception":
                continue
            with st.status(f"**{step[0].tool}**: {step[0].tool_input}", state="complete"):
                st.write(step[0].log)
                st.write(step[1])
        st.write(msg.content)

if instruction := st.chat_input(placeholder="Escribir aqui :D"):
    st.chat_message("user").write(instruction)

    llm = Together(
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        temperature=0,
        max_tokens=256,
        top_k=1,
        together_api_key=os.getenv('TOGETHER_API_KEY')
    )

    tools = [milvus_chilean_consult]
    agent = create_structured_chat_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps = True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        cfg = RunnableConfig()
        cfg["callbacks"] = [st_cb]
        response = agent_executor.invoke({"input": instruction},cfg)
        #response = executor.invoke(prompt + instruction, cfg)
        st.write(response["output"])
        #st.chat_message("assistant").write(response["output"])
        st.session_state.steps[str(len(msgs.messages) - 1)] = response["intermediate_steps"]
        msgs.add_user_message(instruction)
        msgs.add_ai_message(response["output"])