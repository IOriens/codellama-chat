from threading import Thread
from typing import Iterator, Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from flask import Flask, request, jsonify
import torch.distributed as dist
from flask_cors import CORS


# CodeLlama model 
model_id = 'codellama/CodeLlama-7b-Instruct-hf'
quantization_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_compute_dtype=torch.float16
)
if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id) 
    # config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=quantization_config,
        # torch_dtype=torch.float16,
        # load_in_4bit=True,
        device_map='auto',
        # use_safetensors=False,
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Flask API
app = Flask(__name__)
CORS(app)

@app.route("/v1/chat/completions", methods=["POST"]) 
def chat_completions():
    content = request.json
    messages = content["messages"]
    
    # Process messages
    if messages[0]["role"] == "assistant":
        messages[0]["role"] = "system"
    print(content)
    last_role = None
    remove_elements = []
    for i in range(len(messages)):
        if messages[i]["role"] == last_role:
            messages[i-1]["content"] += "\n\n" + messages[i]["content"]
            remove_elements.append(i)
        else:
            last_role = messages[i]["role"]
    
    for element in remove_elements:
         messages.pop(element)
         
    # Run CodeLlama
    response = run_chat_completion(messages)
    response = list(response)
    response = response[-1]



    print(response)

    # Return response
    return jsonify({"choices": [{"delta": {"role": "assistant", "content": response}}]})

def get_prompt(messages: list[dict], system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n'] 
    
    do_strip = False
    for message in messages:
        content = message["content"].strip() if do_strip else message["content"]
        if message["role"] == "user":
            texts.append(f'{content} [/INST] ')
        else:
            texts.append(f' {content.strip()} </s><s>[INST] ')
        do_strip = True

    return ''.join(texts)

def run_chat_completion(
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 50    
) -> str:
    
    system_prompt = "The following is a conversation with an AI assistant. The assistant is helpful, harmless, and honest."
    prompt = get_prompt(messages, system_prompt)
    
    inputs = tokenizer([prompt], return_tensors='pt', add_special_tokens=False).to('cuda')

    streamer = TextIteratorStreamer(
        tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True
    )
    
    generate_kwargs = dict(
        inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=top_p, 
        top_k=top_k,
        temperature=temperature,
        num_beams=1,
    )
    
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    outputs = []
    for text in streamer:
        outputs.append(text)
        yield ''.join(outputs)
        
    return outputs[-1]

if __name__ == "__main__":
   app.run()