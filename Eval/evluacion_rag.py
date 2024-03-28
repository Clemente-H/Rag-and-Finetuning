import os
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from typing import List, Any
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from sentence_transformers import SentenceTransformer
from langchain_together import Together
import os
from langchain import hub
from dotenv import load_dotenv
from langchain.agents import create_openai_functions_agent, AgentExecutor, create_openai_tools_agent, create_structured_chat_agent

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

@tool('milvus_query_results')
def milvus_query_results(query: dict) -> List[Any]:
    "Gives a list of document found by the query"
    query = query['title']
    vect = model.encode(query)
    results = collection.search(
        data=[vect], 
        anns_field="vector", 
        param={"metric_type": "L2", "params": {"nprobe": 1}}, 
        limit=10, 
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

tools = [milvus_query_results]

llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=1024,
    top_k=1,
    together_api_key=os.getenv("TOGETHER_API_KEY")
)


DEFAULT_TEMPLATE = """You are a helpfull AI assistant that helps to makes querys to a vectore store database given a human consult of a specific topic.
The AI gives information and explanation from the results of the query.
The AI is talkative and provides a lot of specific details of its context.
If the AI does not know the answer to the question, it will truthfully sai it does not know.

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

agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Ruta del archivo CSV con las preguntas y respuestas correctas
csv_file = '../Data/preguntas_noticias_santiago.csv'

# Leer el archivo CSV
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Saltear la primera fila
    data = list(reader)

for row in data:
    pregunta = row[0]
    respuesta_correcta = row[1]
    # Responde con 'Verdadero' , 'Falso' o 'No lo se' a la siguiente afirmación: {pregunta}"""
    input_text = f"""Eres un asistente de IA especializado en verificar hechos sobre la realidad chilena en el año 2023. 
    Tu deber es verificar la veracidad de las afirmaciones presentadas relacionadas con eventos, situaciones o datos sobre Chile durante el año 2023, y responder si son verdaderas o falsas de manera concisa, sin proporcionar explicaciones adicionales. 
    Si no tienes suficiente información en tu base de conocimientos para determinar la veracidad de una afirmación sobre la realidad chilena en 2023, debes responder honestamente que no lo sabes. 
    Tus respuestas deben ser siempre en español.

    Responde unicamente con 'Verdadero', 'Falso' o 'No lo sé' a la siguiente afirmación sobre la realidad chilena en 2023: 
    
    {pregunta}"""
    respuesta_modelo = llm.invoke(input_text).strip().lower()
    agent_executor.invoke({"input": pregunta})
    time.sleep(1)
    print(respuesta_modelo)
    # Verificar si la respuesta es correcta
    if respuesta_modelo == respuesta_correcta.lower():
        correctas += 1
    total += 1

    # Agregar la respuesta del modelo a la lista
    respuestas_modelo.append(respuesta_modelo)

# Crear un nuevo archivo CSV con las respuestas del modelo
output_file = '../Data/respuestas_preguntas_noticias_santiago_base_model.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for i, row in enumerate(data):
        writer.writerow([row[0], row[1], respuestas_modelo[i]])



agent_executor.invoke({"input": "Acontecimientos en Marzo en la region de los rios"})