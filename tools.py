import os
import requests
from io import BytesIO, StringIO
from contextlib import redirect_stdout
from PIL import Image
import simpleaudio
import pyaudio
import wave

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


# Tools for function calling
class Toolbox:

    def __init__(self, tools, image_client=None):

        if not tools:
            self.tools = list(self.tool_calls.keys())
        else:
            self.tools = [tool for tool in tools if tool in self.tool_calls]

        self.calls = {tool: self.tool_calls[tool] for tool in self.tools}
        self.descs = [self.tool_descs[tool] for tool in self.tools]

        self.image_client = image_client


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

    def display_image(url):

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
    
    tool_calls = {
        'run_python': run_python,
        'display_image': display_image,
        'generate_image': generate_image,
    }
        
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
    }
    