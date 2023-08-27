from threading import Thread
from typing import Iterator, Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer,
    BitsAndBytesConfig,
)
from flask import Flask, request, jsonify, Response, stream_with_context
import torch.distributed as dist
from flask_cors import CORS
import json


# CodeLlama model
model_id = "codellama/CodeLlama-7b-Instruct-hf"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    # llm_int8_enable_fp32_cpu_offload=True
)
if torch.cuda.is_available():
    config = AutoConfig.from_pretrained(model_id)
    config.pretraining_tp = 1
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        config=config,
        quantization_config=quantization_config,
        device_map="auto",
        # use_safetensors=False,
    )
else:
    model = None
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Flask API
app = Flask(__name__)
CORS(app)


@app.route("/v1/completions", methods=["POST"])
def completions():
    content = request.json
    
    # Is used by Continue to generate a relevant title corresponding to the
    # model's response, however, the current prompt passed by Continue is not
    # good at obtaining a title from Code Llama's completion feature so we
    # use chat completion instead.
    messages = [
        {
            "role": "user",
            "content": content["prompt"]
        }
    ]

    print("-------------------")
    print(content["prompt"])
    print("-------------------")
    
    # Perform Code Llama chat completion.
    response = run_chat_completion(messages)

        # get outputs
    outputs = []
    if not response is None:
        for text in response:
            outputs.append(text)
    else:
        print("response is None")
    
    # Send back the response.
    return jsonify({"choices": [{"text":  "".join(outputs)}]})


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    content = request.json
    messages = content["messages"]
    temperature = content.get("temperature", 0.1)
    top_p = content.get("top_p", 0.9)
    top_k = content.get("top_k", 10)
    stream = content.get("stream", False)
    max_new_tokens = content.get("max_tokens", 1024)

    # Process messages
    if messages[0]["role"] == "assistant":
        messages[0]["role"] = "system"

    last_role = None
    remove_elements = []
    for i in range(len(messages)):
        if messages[i]["role"] == last_role:
            messages[i - 1]["content"] += "\n\n" + messages[i]["content"]
            remove_elements.append(i)
        else:
            last_role = messages[i]["role"]

    # remove messages in remove_elements
    finalMessages = []
    for i in range(len(messages)):
        if not i in remove_elements:
            finalMessages.append(messages[i])

    response = run_chat_completion(
        finalMessages, max_new_tokens, temperature, top_p, top_k
    )

    if stream:
        def generate():
            outputs = []
            for text in response:
                outputs.append(text)
                yield "data: " + json.dumps(
                    {"choices": [{"delta": {"role": "assistant", "content": text}}]}
                ) + "\n\n"
        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    # get outputs
    outputs = []
    if not response is None:
        for text in response:
            outputs.append(text)
    else:
        print("response is None")

    # Return response
    return jsonify(
        {"choices": [{"delta": {"role": "assistant", "content": "".join(outputs)}}]}
    )


def get_prompt(messages: list[dict], system_prompt: str) -> str:
    texts = [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]

    do_strip = False
    for message in messages:
        messageContent = message["content"].strip() if do_strip else message["content"]
        if message["role"] == "user":
            texts.append(f"{messageContent} [/INST] ")
        else:
            texts.append(f" {messageContent.strip()} </s><s>[INST] ")
        do_strip = True
    print("-------------------")
    print("".join(texts))
    print("-------------------")
    return "".join(texts)


def run_chat_completion(
    messages: list[dict],
    max_new_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.9,
    top_k: int = 10,
) -> str:
    system_prompt: str = (
        "The following is a conversation with an AI assistant. The assistant is helpful, harmless, and honest.",
    )

    # get system prompt from messages
    system_prompt = ""
    for message in messages:
        if message["role"] == "system":
            system_prompt = message["content"]
            messages.remove(message)
            break

    prompt = get_prompt(messages, system_prompt)

    inputs = tokenizer([prompt], return_tensors="pt", add_special_tokens=False).to(
        "cuda"
    )

    streamer = TextIteratorStreamer(
        tokenizer, timeout=1000.0, skip_prompt=True, skip_special_tokens=True
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
        eos_token_id=2, 
        pad_token_id=2
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()

    return streamer


if __name__ == "__main__":
    app.run()
