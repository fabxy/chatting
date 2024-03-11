from dotenv import load_dotenv
from openai import OpenAI
import time
import argparse
import json

from tools import Toolbox, speak_text, record_audio

# Identity
system_name = "Peter"

# Set OpenAI chat model and list available local LLMs
openai_model = 'gpt-3.5-turbo'
models = ['llama2']

# Command line arguments
parser = argparse.ArgumentParser(description=f'Options for chatting with {system_name}.')
parser.add_argument('--speak', action='store_true', help=f'Should {system_name} have a voice?')
parser.add_argument('-mem', '--memory', action='store_true', help=f'Allow {system_name} to read and update the memory of previous conversations.')
parser.add_argument('--model', choices=models, default=openai_model, help=f'Specify a local LLM to be used. Otherwise {openai_model} is used.')
parser.add_argument('--tools', nargs='*', help=f'Select specific tools that {system_name} can use.')
parser.add_argument('--stream', action='store_true', help=f'Should text output be streamed? ')

args = parser.parse_args()

speak = args.speak
memory = args.memory
model = args.model
tools = args.tools # TODO: specify list of possible tools
stream = args.stream

# Load API key
load_dotenv()

# Start client
speech_client = OpenAI()
image_client = speech_client

if model == openai_model:
    client = speech_client
else:
    client = OpenAI(base_url = "http://localhost:11434/v1", api_key="ollama")

# Tools
if tools is not None:
    stream = False
    toolbox = Toolbox(tools, image_client)
    tool_descs = toolbox.descs
else:
    tool_descs = None

# Settings
tokens = 0
max_tokens = 10000
stream_delay = 0.1
message_log = "log.messages"
safe_mode = True

# Initialization
with open('system.prompt', 'r') as f:
    system_prompt = f.read().strip().replace("SYSTEM_NAME", system_name).replace("DATE", time.strftime('%Y-%m-%d'))

with open('greeting.prompt', 'r') as f:
    greeting = f.read().strip()
end_prompt = "END"
    
# Memory
if memory:
    memory_prompt = "The following is your memory of previous chats:\n\n"

    with open('.memory', 'r') as f:
        memory_prompt += f.read()

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

# Start chatting
print(f"{system_name}: {greeting}")
if speak:
    stream = False
    speak_text(greeting, speech_client)

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
        tools=tool_descs,
    )

    try:
        tokens += response.usage.total_tokens
    except:
        pass # TODO: use tiktoken

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
                    speak_text(answer, speech_client) # TODO: count tokens of speaking and listening
            
            messages.append(response.choices[0].message)

            for tool_call in tool_calls:
                tool_response = toolbox.calls[tool_call.function.name](**(json.loads(tool_call.function.arguments)))
                messages.append({"tool_call_id": tool_call.id, "role": "tool", "name": tool_call.function.name, "content": tool_response})
                # TODO: does the function result have to be in json?

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                stream=stream,
                tools=tool_descs,
            )
            tokens += response.usage.total_tokens

        answer = response.choices[0].message.content

        print(f"{system_name}: {answer}")
        if speak:
            speak_text(answer, speech_client)

        messages.append({"role": "assistant", "content": answer})


if memory:
    
    with open('memory.prompt', 'r') as f:
        memory_update_prompt = f.read().strip()
    
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