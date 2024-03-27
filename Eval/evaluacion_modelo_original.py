import csv
from langchain_together import Together
import os
from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Cargar el modelo LLM Mixtral-7x8B
llm = Together(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    temperature=0,
    max_tokens=64,
    top_k=1,
    together_api_key=os.getenv('TOGETHER_API_KEY')
)
# Ruta del archivo CSV con las preguntas y respuestas correctas
csv_file = '../Data/preguntas_noticias_santiago.csv'

# Leer el archivo CSV
with open(csv_file, 'r') as file:
    reader = csv.reader(file)
    next(reader)  # Saltear la primera fila
    data = list(reader)

# Iterar sobre las preguntas y obtener las respuestas del modelo
respuestas_modelo = []
correctas = 0
total = 0

for row in data:
    pregunta = row[0]
    respuesta_correcta = row[1]
    #    input_text = """You are an AI assistant specialized in fact-checking. 
    # Your duty is to verify the truthfulness of the presented statements and respond whether they are true or false in a concise manner, without providing additional explanations. 
    # If you don't have enough information to determine the truthfulness of a statement, you must honestly respond that you don't know. 
    # Your responses must always be in Spanish. Let's begin:

    # Responde con 'Verdadero' , 'Falso' o 'No lo se' a la siguiente afirmación: {pregunta}"""
    input_text = f"""Eres un asistente de IA especializado en verificar hechos sobre la realidad chilena en el año 2023. 
    Tu deber es verificar la veracidad de las afirmaciones presentadas relacionadas con eventos, situaciones o datos sobre Chile durante el año 2023, y responder si son verdaderas o falsas de manera concisa, sin proporcionar explicaciones adicionales. 
    Si no tienes suficiente información en tu base de conocimientos para determinar la veracidad de una afirmación sobre la realidad chilena en 2023, debes responder honestamente que no lo sabes. 
    Tus respuestas deben ser siempre en español.

    Responde unicamente con 'Verdadero', 'Falso' o 'No lo sé' a la siguiente afirmación sobre la realidad chilena en 2023: 
    
    {pregunta}"""
    # Invocar al modelo LLM para obtener la respuesta
    #    input_text = f"Responde 
    #    con 'Verdadero' o 'Falso' a la siguiente pregunta: 
    #    {pregunta}"
    print(input_text)
    break
    respuesta_modelo = llm.invoke(input_text).strip().lower()

    # Verificar si la respuesta es correcta
    if respuesta_modelo == respuesta_correcta.lower():
        correctas += 1
    total += 1

    # Agregar la respuesta del modelo a la lista
    respuestas_modelo.append(respuesta_modelo)

# Crear un nuevo archivo CSV con las respuestas del modelo
output_file = '../Data/respuestas_preguntas_noticias_santiago.csv'
with open(output_file, 'w', newline='') as file:
    writer = csv.writer(file)
    for i, row in enumerate(data):
        writer.writerow(row + [respuestas_modelo[i]])

# Calcular el puntaje
puntaje = correctas / total * 100

# Imprimir el puntaje
print(f"Puntaje: {puntaje:.2f}% ({correctas}/{total})")