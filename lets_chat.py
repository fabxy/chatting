from dotenv import load_dotenv
from openai import OpenAI
import os
import sys
import time
import pyaudio
import wave
import simpleaudio

# Command line arguments
speak = 'speak' in sys.argv[1:]
memory = 'mem' in sys.argv[1:]

# Load API key
load_dotenv()

# Start OpenAI client
client = OpenAI()

# Settings
model = 'gpt-3.5-turbo'
stream = False
stream_delay = 0.1

# Initialization
tokens = 0
max_tokens = 10000
system_name = "Peter"
system_prompt = f"You are {system_name}, a helpful assistant to a researcher who has programming questions or simply needs some companionship. Today's date is {time.strftime('%Y-%m-%d')}."
greeting = "Howdy my friend, how can I help you today?"
end_prompt = "END"

if memory:
    memory_text = "The following is your memory of previous chats:\n\n"

    with open('.memory', 'r') as f:
        memory_text += ''.join(f.readlines())

    system_prompt += "\n" + memory_text

messages = [
    {
        "role": "system", 
         "content": system_prompt,
    },
    {
        "role": "system", 
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
        response = client.audio.speech.create(model="tts-1", voice="onyx", response_format='wav', input=text)
        response.write_to_file("response.wav")
        wave_obj = simpleaudio.WaveObject.from_wave_file("response.wav")
    else:
        if "greeting.wav" not in os.listdir():
            response = client.audio.speech.create(model="tts-1", voice="onyx", response_format='wav', input=text)
            response.write_to_file("greeting.wav")
        wave_obj = simpleaudio.WaveObject.from_wave_file("greeting.wav")
    
    play_obj = wave_obj.play()
    play_obj.wait_done()

# Start chatting
if speak:
    stream = False
    print(f"{system_name}:")
    speak_text(greeting)
else:
    print(f"{system_name}: {greeting}")

while tokens < max_tokens:
    prompt = input(">>> ")
    print("")

    if end_prompt in prompt:
        break

    if prompt == '':

        record_audio(3, "prompt.wav")


        audio_file= open("prompt.wav", "rb")
        prompt = client.audio.transcriptions.create(model="whisper-1", file=audio_file, response_format="text").strip()

        # print(prompt)

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=stream,
    )

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
    else:
        answer = response.choices[0].message.content
        tokens += response.usage.total_tokens

        if speak:
            print(f"{system_name}:")
            speak_text(answer)
        else:
            print(f"{system_name}: {answer}")
    
    messages.append({"role": "system", "content": answer})


if memory:
    memory_prompt = "Please write an update to your memory with the categories 'Personal user information', 'Topics covered' and 'Notes for future chats'. Write at most 10 points per category. Start and end the memory with a ``` line."
    
    messages.append({"role": "user", "content": memory_prompt})
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    answer = response.choices[0].message.content
    tokens += response.usage.total_tokens

    if answer.count("```") == 2:
        
        memory_text = answer.split("```")[1]
        with open(".memory", 'w') as f:
            f.write(memory_text)

        print(f"Memory updated:\n{memory_text}")

print(f"Time flies when you're having fun. Have a great day, byeee!")
if not stream:
    print(f"Today's total token usage was {tokens}.")