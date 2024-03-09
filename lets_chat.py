from dotenv import load_dotenv
from openai import OpenAI
import os
import sys
import time
import pyaudio
import wave
import simpleaudio
from PIL import Image
import requests
from io import BytesIO, StringIO
from contextlib import redirect_stdout
import json

# List available local LLMs
models = ['llama2']

# Command line arguments
speak = 'speak' in sys.argv[1:]
memory = 'mem' in sys.argv[1:]
tools = 'tools' in sys.argv[1:]
model = None
for m in models:
    if m in sys.argv[1:]: model = m

# Load API key
load_dotenv()

# Start clients
speech_client = OpenAI()
image_client = speech_client
if model:
    client = OpenAI(base_url = "http://localhost:11434/v1", api_key="ollama")
else:
    model = 'gpt-3.5-turbo'
    client = speech_client

# Initialization
stream = False
stream_delay = 0.1
tokens = 0
max_tokens = 10000
system_name = "Peter"
system_prompt = f"You are {system_name}, a helpful assistant to a researcher who has programming questions or simply needs some companionship. Today's date is {time.strftime('%Y-%m-%d')}."
greeting = "Howdy my friend, how can I help you today?"
end_prompt = "END"
message_log = "log.messages"
safe_mode = True

if tools:
    stream = False # TODO
    tool_list = [
        {"type": "function", 
         "function": {
            "name": "display_image", 
            "description": "Display image at given url to user", 
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The url address on the local machine or the internet"
                    }}}}},
        
        {"type": "function", 
         "function": {
            "name": "generate_image", 
            "description": "Generate new image via generative AI according to given prompt", 
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A prompt describing the content of the image"
                    }}}}},

        {"type": "function", 
         "function": {
            "name": "run_python", 
            "description": "Execute given code via python interpreter", 
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code performing the intended function"
                    }}}}},

                    ]

if memory:
    memory_prompt = "The following is your memory of previous chats:\n\n"

    with open('.memory', 'r') as f:
        memory_prompt += ''.join(f.readlines())

    system_prompt += "\n\n" + memory_prompt

messages = [
    {
        "role": "system", 
         "content": system_prompt,
    },
    {
        "role": "assistant",
         "content": greeting,
    },
]

# Tools
def record_audio(seconds, out_file):

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    CHUNK = 1024
    
    p = pyaudio.PyAudio()
    s = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

    # print(f"Recording for {seconds} seconds:")

    frames = []

    for i in range(0, int(RATE / CHUNK * seconds)):
        data = s.read(CHUNK)
        frames.append(data)

    # print("Done recording.")

    s.stop_stream()
    s.close()
    p.terminate()

    wf = wave.open(out_file, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def speak_text(text):

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

def generate_image(prompt):

    try:
        response = image_client.images.generate(
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
    
def run_python(code):

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
    
tool_calling = {
    'display_image': display_image,
    'generate_image': generate_image,
    'run_python': run_python,
}

# Start chatting
print(f"{system_name}: {greeting}")
if speak:
    stream = False
    # print(f"{system_name}:")
    speak_text(greeting)

while tokens < max_tokens:
    prompt = input(">>> ")
    print("")

    if end_prompt in prompt:
        break

    if prompt == '':

        record_audio(5, "prompt.wav")

        audio_file= open("prompt.wav", "rb")
        prompt = speech_client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text").strip()

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
        tools=tool_list,
    )

    try:
        tokens += response.usage.total_tokens
    except:
        pass

    if stream:
        answer = ""
        print(f"{system_name}: ", end="", flush=True)

        for chunk in response:
            time.sleep(stream_delay)
            phrase = chunk.choices[0].delta.content
            if phrase:
                answer += phrase
                print(phrase, end="", flush=True)
        
        print("")
        messages.append({"role": "assistant", "content": answer})
    
    else:
        while (tool_calls := response.choices[0].message.tool_calls):
            
            answer = response.choices[0].message.content
            if answer:
                print(f"{system_name}: {answer}")        
                if speak:
                    # print(f"{system_name}:")
                    speak_text(answer) # TODO: count tokens of speaking and listening
            
            messages.append(response.choices[0].message)

            for tool_call in tool_calls:
                tool_response = tool_calling[tool_call.function.name](**(json.loads(tool_call.function.arguments)))
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name, "content": tool_response})
                # TODO: does the function result have to be in json?

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                tools=tool_list
            )
            tokens += response.usage.total_tokens

        answer = response.choices[0].message.content

        print(f"{system_name}: {answer}")
        if speak:
            # print(f"{system_name}:")
            speak_text(answer)

        messages.append({"role": "assistant", "content": answer})


if memory:
    memory_update_prompt = "Please write an update to your memory with the categories 'Personal user information', 'Topics covered' and 'Notes for future chats'. Write at most 10 points per category. Start and end the memory with a ``` line."
    
    messages.append({"role": "user", "content": memory_update_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    tokens += response.usage.total_tokens

    answer = response.choices[0].message.content
    
    if answer.count("```") == 2:
        
        memory_text = answer.split("```")[1]
        with open(".memory", 'w') as f:
            f.write(memory_text)

        print(f"Memory updated:\n{memory_text}")

    messages.append({"role": "assistant", "content": answer})

if message_log:
    with open(message_log, 'w') as f:
        for m in messages:
            f.write(str(m)+'\n')

print(f"Time flies when you're having fun. Have a great day, byeee!")
if not stream:
    print(f"Today's total token usage was {tokens}.")