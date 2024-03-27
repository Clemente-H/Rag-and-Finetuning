import pandas as pd
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import SentenceTransformersTokenTextSplitter
from tqdm.auto import tqdm
import torch
import os
from pymilvus import MilvusClient, connections, FieldSchema, CollectionSchema, DataType, Collection, utility
import json
import re
import argparse

# Configurar el analizador de argumentos
parser = argparse.ArgumentParser(description='Insertar datos en Milvus')
parser.add_argument('--file', type=str, required=True, help='Ruta del archivo CSV a procesar')
args = parser.parse_args()

# Obtener la ruta del archivo desde los argumentos
rute = args.file

df = pd.read_csv(rute)
df = df.drop(['storyUri', 'sentiment', 'lang', 'location', 'isDuplicate', 'authors', 'links', 'dateTime', 'sim',
              'time', 'source', 'concepts', 'date', 'videos', 'categories', 'image', 'eventUri', 'extractedDates'], axis=1)

caracteres_especiales = '³©Ãâ€œ'

df = df[~df['body'].str.contains(f"[{re.escape(caracteres_especiales)}]")]

def cut_2_point(row):
    if ".cl" in row:
        rs = row.split('.cl')
        return ''.join(rs[0:1]) + '.cl'
    elif ".com" in row:
        rs = row.split('.com')
        return ''.join(rs[0:1]) + '.com'
    else:
        return row

print("Lista de noticieros: ")
print(df['url'].apply(lambda row: cut_2_point(row)).unique())

# Most trustworthy news sites according to Chileans: The Reuters Institute Digital News Report 2021
news_to_keep = ["www.biobiochile.cl", "cooperativa.cl", "www.adnradio.cl",
                "www.chvnoticias.cl", "www.24horas.cl", 'https://www.tvn.cl', 'https://www.emol.com',
                'https://www.theclinic.cl']

resized_df = df[df['url'].apply(lambda x: any(news_site in x for news_site in news_to_keep))]

# SplitByToken
splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=10, model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
tqdm.pandas(desc="Procesando texto")

# Función para dividir el texto y expandir el DataFrame
def split_and_expand(row):
    text = row['title'] + '\n' + row['body']
    text_chunks = splitter.split_text(text)

    temp_df = pd.DataFrame([row] * len(text_chunks))
    temp_df['body'] = text_chunks
    return temp_df

# Aplicamos la función a cada fila y utilizamos tqdm para mostrar el progreso
expanded_df = pd.concat(resized_df.progress_apply(split_and_expand, axis=1).tolist(), ignore_index=True)

# GenerateEmbeddings
model_name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_name)
if torch.cuda.is_available():
    model = model.to('cuda')

# Función para codificar los textos utilizando el modelo y manejar manualmente el batch_size
def encode_texts(model, texts, batch_size=32):
    vectors = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch_texts = texts[i:i + batch_size]
        batch_vectors = model.encode(batch_texts, convert_to_tensor=True, show_progress_bar=False)
        vectors.extend(batch_vectors)
    return vectors

# Codificar los textos en vectores
vectors = encode_texts(model, expanded_df['body'].tolist(), batch_size=64)

# Convertir tensores a lista y asignar al DataFrame
expanded_df['vector'] = [vector.cpu().numpy().tolist() for vector in vectors]

# InsertData
# Convertir el DataFrame filtrado a una lista de diccionarios
data_list = expanded_df.to_dict(orient='records')

# Crear el diccionario final con la clave 'rows'
final_structure = {'rows': data_list}

ruta_archivo = './Data/embeddings/2020_embedding_upload.json'
with open(ruta_archivo, 'w', encoding='utf-8') as file:
    json.dump(final_structure, file, ensure_ascii=False, indent=None)

with open(ruta_archivo, 'r') as f:
    data = json.load(f)

# Replace uri and token with your own
client = MilvusClient(
    uri=os.getenv("MILVUS_ENDPOINT"),
    token=os.getenv("MILVUS_API_KEY")  # API key or a colon-separated cluster username and password
)

# Connect to cluster
connections.connect(
    alias='default',
    # Public endpoint obtained from Zilliz Cloud
    uri=os.getenv("MILVUS_ENDPOINT"),
    # API key or a colon-separated cluster username and password
    token=os.getenv("MILVUS_API_KEY")
)

# Definimos el tamaño del bloque
block_size = 10000

# Iteramos sobre los bloques
for i in range(0, len(data['rows']), block_size):
    res = client.insert(
        collection_name='News_2020',
        data=data['rows'][i:i + block_size]
    )
    print('Insertando bloque ' + str(i))

print("Data insertada correctamente")