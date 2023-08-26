# CodeLlama Chat Assistant

The CodeLlama Chat Assistant is a project built on Flask and the CodeLlama AI model, designed to facilitate real-time chat interactions with an AI assistant. This project enables users to send chat messages and receive responses from the AI assistant.

## Features

- Real-Time Chat Interaction: Engage in real-time chat interactions with the AI assistant by sending chat messages to the API.
- Response Streaming: Responses can be streamed as event flows, providing an efficient real-time chat experience if desired.
- Powered by CodeLlama AI: The project leverages the CodeLlama AI model to generate responses from the assistant, delivering an intelligent chat experience.

## Getting Started

The following steps will guide you through setting up and running the project in your local environment.

### 1. Environment Setup

Ensure your environment meets the following requirements:

- Python 3.6 or higher
- Flask and other required Python libraries

You can install the necessary dependencies using the following command:

```bash
pip install -r requirements.txt
```

### 2. Install Prerequisite

Before running the project, you need to install the `bitsandbytes` library on Windows. You can install it using the following command:

```shell
python -m pip install bitsandbytes --prefer-binary --extra-index-url=https://jllllll.github.io/bitsandbytes-windows-webui
```

### 3. Configure the Model

Within the project, you'll need to configure the CodeLlama AI model. In the code, locate the following section to configure the model:

```python
model_id = "codellama/CodeLlama-7b-Instruct-hf"
# ...
```

### 4. Launch the Project

In your terminal, use the following command to launch the Flask application:

```bash
python main.py
```

The application will run on the default host (usually `localhost`) and port (typically `5000`). You can interact with the AI assistant by accessing `http://localhost:5000`.

## API Endpoints

### POST `/v1/chat/completions`

Send a JSON request to this endpoint containing chat messages to interact with the AI assistant. The request body should include the following fields:

- `messages`: A list containing chat messages, each with `role` and `content` fields specifying the message's role and content.
- `stream`: A boolean indicating whether to return the response as an event stream.

The response will be returned in JSON format, containing the AI assistant's response.

## Contribution

Feel free to raise issues, provide suggestions, and contribute code. If you encounter any issues or have suggestions for improvements, create an Issue to let us know.

## License

This project is licensed under the [MIT License](LICENSE).