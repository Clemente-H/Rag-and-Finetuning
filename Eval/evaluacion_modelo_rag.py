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
import csv
import time

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

collection = Collection("News_2023")      # Get an existing collection.
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
        limit=6, 
        expr=None,
        consistency_level="Strong"
    )
    ids = results[0].ids
    print(ids)
    res = collection.query(
        expr =f"""Auto_id in {ids}""" , 
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


DEFAULT_TEMPLATE = """Eres un asistente de IA especializado en verificar hechos sobre la realidad chilena en el año 2023. 
Tu deber es verificar la veracidad de las afirmaciones presentadas relacionadas con eventos, situaciones o datos sobre Chile durante el año 2023, y responder si son verdaderas o falsas de manera concisa, sin proporcionar explicaciones adicionales. 
Responderas unicamente Verdadero, Falso o No lo Sé, sin otorgar explicaciones adicionales.
Para revisar si la afirmacion es correcta, utilizaras tu conocimiento factual y en caso de ser necesario utilizaras una de las herramientas que se te daran a continuacion para hacer una consulta a una base de datos:

Tienes acceso a las siguientes tools:
{tools}

Usa un json blob para especificar una tool dando una action key (nombre de la tool) y una action_input key (input tool)

Valores validos de "action" : "Final Answer or {tool_names}

Da solo UNA action por $JSON_BLOB, como se muestra:

```
{{
    "action": $TOOL_NAME,
    "action_input": $INPUT
}}```

Sigue el siguiente formato:

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

Por ejemplo, si quieres usar una tool que hace una query a una base datos vectore store, tu $JSON_BLOB podría verse así:      

```
{{
    'action': 'vector_store_query_json', 
    'action_input': {{'query': 'Example query'}}
}}```

Comencemos! Recuerda SIEMPRE responder con un json blob valido de una sola action. Usa las tools si es necesario. Responde directamente si es apropiado. El formato de Action: ```$JSON_BLOB```luego Observation'
"""

prompt = hub.pull("hwchase17/structured-chat-agent")
prompt[0].prompt.template = DEFAULT_TEMPLATE

agent = create_structured_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps = True)

# Ruta del archivo CSV con las preguntas y respuestas correctas
csv_file = '../Data/preguntas_noticias_santiago.csv'

# Leer el archivo CSV
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Saltear la primera fila
    data = list(reader)

respuestas_modelo = []
pasos_intermedios = []
correctas = 0
total = 0

for row in data:
    pregunta = row[0]
    respuesta_correcta = row[1]
    response = agent_executor.invoke({"input": "afirmacion: " + pregunta})
    respuesta_modelo = response["output"]
    intermediate_steps = response["intermediate_steps"]
    time.sleep(1)
    print(respuesta_modelo)
    # Verificar si la respuesta es correcta
    if respuesta_modelo == respuesta_correcta.lower():
        correctas += 1
    total += 1

    # Agregar la respuesta del modelo a la lista
    respuestas_modelo.append(respuesta_modelo)
    pasos_intermedios.append(intermediate_steps)

# Crear un nuevo archivo CSV con las respuestas del modelo
output_file = '../Data/respuestas_preguntas_noticias_santiago_rag_model2.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for i, row in enumerate(data):
        writer.writerow([row[0], row[1], respuestas_modelo[i]])