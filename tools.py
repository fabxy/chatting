import os
import numpy as np
import requests
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from PIL import Image
import simpleaudio
import pyaudio
import wave
import pypdf
from tqdm import tqdm

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


def chunk_pdf(file_name, window_size=500, stride=300):

    reader = pypdf.PdfReader(file_name)
    all_words = [word for page in reader.pages for word in page.extract_text().split()]
    chunks = [' '.join(all_words[(i*stride):(i*stride+window_size)]) for i in range(len(all_words) // stride + 1)]

    return chunks


def embed_chunks(chunks, embedding_client):

    res = {}
    for chunk in tqdm(chunks, desc="Chunk embedding"):
        response = embedding_client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        res[chunk] = response.data[0].embedding
    
    return res


def query_embedding(query, emb_dict, embedding_client):

    response = embedding_client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
        )
    q_emb = response.data[0].embedding

    res = []
    for chunk, k_emb in emb_dict.items():
        res.append((np.dot(q_emb, k_emb) / np.linalg.norm(q_emb) / np.linalg.norm(k_emb), chunk))

    return sorted(res, reverse=True)


# Tools for function calling
class Toolbox:

    def __init__(self, tools, image_client=None, embedding_client=None, emb_dict=None, emb_ktop=1):

        if not tools:
            self.tools = list(self.tool_descs.keys())
        else:
            self.tools = [tool for tool in tools if tool in self.tool_descs]

        self.calls = {tool: getattr(self, tool) for tool in self.tools}
        self.descs = [self.tool_descs[tool] for tool in self.tools]

        self.image_client = image_client
        self.embedding_client = embedding_client

        self.emb_dict = emb_dict
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
        
        res = query_embedding(query, self.emb_dict, self.embedding_client)

        answer = f"The top {self.emb_ktop} search results are the following:\n\n"
        answer += '\n\n'.join([f"{i+1}: {res[i][1]}" for i in range(self.emb_ktop)])

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
    