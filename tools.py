import os
import requests
import chromadb
import hashlib
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from PIL import Image
import simpleaudio
import pyaudio
import wave
import pypdf
from tqdm import tqdm
import json

# Auxillary tools
def record_audio(seconds, out_file):

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    
    p = pyaudio.PyAudio()
    s = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = s.read(CHUNK)
        frames.append(data)

    s.stop_stream()
    s.close()
    p.terminate()

    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()


def speak_text(text, speech_client):

    try:
        with open('greeting.prompt', 'r') as f:
            greeting = f.read().strip()
    except:
        greeting = None

    if text != greeting:
        response = speech_client.audio.speech.create(model="tts-1", voice="onyx", response_format='wav', input=text)
        response.write_to_file("response.wav")
        wave_obj = simpleaudio.WaveObject.from_wave_file("response.wav")
    else:
        if "greeting.wav" not in os.listdir():
            response = speech_client.audio.speech.create(model="tts-1", voice="onyx", response_format='wav', input=text)
            response.write_to_file("greeting.wav")
        wave_obj = simpleaudio.WaveObject.from_wave_file("greeting.wav")
    
    play_obj = wave_obj.play()
    play_obj.wait_done()


# Retriever augmented generation
class RAG:

    def __init__(self, database_client, collection_name="base", emb_method=None, emb_kwargs=None, query_kwargs=None):

        # Set embedding function
        if emb_method is None or emb_method == "default":

            if emb_kwargs is None or emb_kwargs['model_name'] is None:
                self.emb_fun = chromadb.utils.embedding_functions.DefaultEmbeddingFunction()
                self.emb_model_name = self.emb_fun.MODEL_NAME
            else:
                self.emb_model_name = emb_kwargs['model_name']
                self.emb_fun = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(model_name=self.emb_model_name)

        elif emb_method == "openai":
            if emb_kwargs is None or emb_kwargs['model_name'] is None:
                self.emb_model_name = "text-embedding-3-small"
            else:
                self.emb_model_name = emb_kwargs['model_name']
            self.emb_fun = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(api_key=emb_kwargs['api_key'], model_name=self.emb_model_name)

        else:
            raise NotImplementedError(f"Embedding method {self.emb_method} not implemented.")

        # Get collection
        try:
            self.collection = database_client.get_collection(name=collection_name, embedding_function=self.emb_fun)
        except:
            self.collection = database_client.create_collection(name=collection_name, embedding_function=self.emb_fun, metadata={"hnsw:space": "cosine"})

        # Get dictionary of documents embedded in collections
        self.doc_dict_path = 'chroma/chroma.json'
        try:
            with open(self.doc_dict_path, 'r', encoding='utf-8') as f:
                self.doc_dict = json.load(f)
        except:
            self.doc_dict = {}

        if self.collection.name not in self.doc_dict:
            self.doc_dict[self.collection.name] = []

        # Save query keyword arguments
        self.query_kwargs = query_kwargs


    def add_doc(self, doc, chunk_method="sentence", chunk_kwargs={}):

        # Check if document and chunking method already exist in collection
        if not self.collection.get(where={"$and": [{"filename": os.path.basename(doc)}, {"chunk_method": chunk_method}, {"chunk_kwargs": '-'.join(sorted([f"{k}{v}" for k,v in chunk_kwargs.items()]))}]}, limit=1)['ids']:
                            
            chunks, metadatas, ids = self.chunk_pdf(doc, chunk_method, **chunk_kwargs)
            chunk_bs = 10
            
            for i in tqdm(range(len(chunks) // chunk_bs + (len(chunks) % chunk_bs > 0)), desc="Embedding chunks"):
                li = i * chunk_bs
                hi = (i+1) * chunk_bs
                self.collection.add(documents=chunks[li:hi], metadatas=metadatas[li:hi], ids=ids[li:hi])

            # Add document to embedded documents dictionary
            if doc not in self.doc_dict[self.collection.name]:

                self.doc_dict[self.collection.name].append(doc)
                with open(self.doc_dict_path, 'w', encoding='utf-8') as f:
                    json.dump(self.doc_dict, f, ensure_ascii=False, indent=4)


    def chunk_pdf(self, file_name, chunk_method="sentence", **kwargs):

        reader = pypdf.PdfReader(file_name)

        metadata = {
            'filename': os.path.basename(file_name),
            'emb_model_name': self.emb_model_name,
            'chunk_method': chunk_method,
            'chunk_kwargs': '-'.join(sorted([f"{k}{v}" for k,v in kwargs.items()])),
            'author': reader.metadata["/Author"],
            'title': reader.metadata["/Title"],
            'year': reader.metadata["/CreationDate"][2:6],
            'month': reader.metadata["/CreationDate"][6:8],
            }
    
        if chunk_method == "wordSW":
            try:
                window_size = kwargs['window_size']
            except:
                window_size = 500

            try:
                stride = kwargs['stride']
            except:
                stride = 300

            all_words = [word for page in reader.pages for word in page.extract_text().split()]
            chunks = [' '.join(all_words[(i*stride):(i*stride+window_size)]) for i in range(len(all_words) // stride + (len(all_words) % stride > 0))]
        
        elif chunk_method == "sentence":
            try:
                min_sentence_len = kwargs['min_sentence_len']
            except:
                min_sentence_len = 3
            
            all_text = '\n'.join([page.extract_text() for page in reader.pages])
            chunks = [sentence for sentence in all_text.replace('-\n', '').replace('\n', ' ').split('.') if len(sentence.split()) >= min_sentence_len]
            # TODO: More cleaning of the data
        
        else:
            raise NotImplementedError(f"Chunking method {chunk_method} not implemented.")
        
        metadatas = [metadata] * len(chunks)
        chunk_hash = str(hashlib.md5(str.encode('-'.join(list(metadata.values())))).hexdigest())
        ids = [f"{chunk_hash}-{i}" for i in range(len(chunks))]
    
        return chunks, metadatas, ids


    def query_db(self, query):

        try:
            topk = self.query_kwargs['topk']
        except:
            topk = 1

        try:
            query_window = self.query_kwargs['query_window']
        except:
            query_window = (0, 0)

        try:
            query_docs = self.query_kwargs['query_docs']
        except:
            query_docs = None

        if query_docs is None or len(query_docs) == 0:
            doc_filter = None
        else:
            if len(query_docs) > 1:
                doc_filter = {"$or": [{"filename": os.path.basename(doc)} for doc in query_docs]}
            else:
                doc_filter = {"filename": os.path.basename(query_docs[0])}

        query_res = self.collection.query(
            query_texts=[query],
            n_results=topk,
            where=doc_filter,
            # TODO: allow keywords via where_document={"$contains":"search_string"}
        )

        chunks = []
        for t in range(topk):

            chunk_hash, id = query_res['ids'][0][t].split('-')
            metadata = query_res['metadatas'][0][t]

            window_res = self.collection.get(ids=[f"{chunk_hash}-{i}" for i in range(max(0, int(id)+query_window[0]), int(id)+query_window[1]+1)], where={"$and": [{k: v} for k, v in metadata.items()]})
            chunks.append('. '.join(window_res['documents']))
                          
        return chunks
            

# Tools for function calling
class Toolbox:

    def __init__(self, tools, image_client=None, retriever=None, emb_ktop=1):

        if not tools:
            self.tools = list(self.tool_descs.keys())
        else:
            self.tools = [tool for tool in tools if tool in self.tool_descs]

        self.calls = {tool: getattr(self, tool) for tool in self.tools}
        self.descs = [self.tool_descs[tool] for tool in self.tools]

        self.image_client = image_client
        self.retriever = retriever

        self.emb_ktop = emb_ktop

    def run_python(self, code, safe_mode=True):

        if safe_mode:
            print(f"Python code: {code}")

        if not safe_mode or input("Execute code? (y/n)") == 'y':
            stdout = StringIO()
            try:
                with redirect_stdout(stdout):
                    exec(code)
                output = stdout.getvalue()
                return f"Code execution successful. Output: {output}"
            except Exception as e:
                return f"Code execution failed due to following error: {e}"
        else:
            return "Python code not executed by user."
        
    def generate_image(self, prompt):

        try:
            response = self.image_client.images.generate(
                prompt=prompt,
                n=1,
                size="256x256"
            )
            return f"The generated image is located at the following url: {response.data[0].url}"
        
        except Exception as e:
            return f"The following error occurred during image generation: {e}"

    def display_image(self, url):

        try:
            image = Image.open(url)
            image.show()
            return f"Image {url} shown to user."
        except:
            pass

        try:
            image_response = requests.get(url)
            image = Image.open(BytesIO(image_response.content))
            image.show()
            return f"Image {url} shown to user."

        except Exception as e:
            return f"Image {url} not shown to user due to the error: {e}"
        
    def search_documents(self, query):

        res = self.retriever.query_db(query)

        answer = f"The top {len(res)} search results are the following:\n\n"
        answer += '\n\n'.join([f"{i+1}: {res[i]}" for i in range(len(res))])

        print(answer)
        return answer
        
    tool_descs = {
        'run_python': {
            "type": "function",
            "function": {
                "name": "run_python", 
                "description": "Execute given code via python interpreter", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "The python code performing the intended function"
                        }
                    }
                }
            }
        },
        'display_image': {
            "type": "function", 
            "function": {
                "name": "display_image", 
                "description": "Display image at given url to user", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "The url address on the local machine or the internet"
                        }
                    }
                }
            }
        },
        'generate_image': {
            "type": "function", 
            "function": {
                "name": "generate_image", 
                "description": "Generate new image via generative AI according to given prompt", 
                "parameters": {
                    "type": "object",
                    "properties": {
                        "prompt": {
                            "type": "string",
                            "description": "A prompt describing the content of the image"
                        }
                    }
                }
            }
        },
        'search_documents': {
            "type": "function", 
            "function": {
                "name": "search_documents", 
                "description": "Search the available documents for context matching the given query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "An extensive query describing the goal of the search"
                        }
                    }
                }
            }
        },
    }
    