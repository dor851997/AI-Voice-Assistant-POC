# AI Voice Assistant - Kardom POC

## Overview

This project is a proof-of-concept for a voice assistant that integrates several components to enable real-time audio interactions:

- **LiveKit**: Provides real-time audio streaming and room management.
- **Deepgram**: Offers Speech-to-Text (STT) functionality to transcribe user speech.
- **OpenAI**: Supplies language modeling for generating responses and Text-to-Speech (TTS) for spoken output.
- **Pinecone**: Used as a vector database to store and retrieve interaction pairs based on embeddings.

The project is organized into the following key components:

- **main.py**: Orchestrates the overall flow by setting up the conversation context, connecting to a LiveKit room, starting the assistant, and subscribing to events.
- **voice_assistant.py**: Contains the `CustomVoiceAssistant` class, which handles user message processing, response generation, event subscriptions, and the pairing logic that saves interactions.
- **rag_agent.py**: Manages the embedding generation and upsert operations into Pinecone for storing conversation pairs.
- **llm_handler.py**: Provides LLM preprocessing (e.g., prompt augmentation) and postprocessing (e.g., response normalization) functions.
- **api.py**: Implements TV control functions that map processed assistant responses to actionable commands (like power on, volume up, etc.).
- **config.py**: Contains shared configuration constants such as the list of allowed commands.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd AI-Voice-Assistant-Kardom-POC
   ```

2. **Update the .env File:**
   Open the `.env` file and update the keys with your API credentials.

3. **Create and Activate a Virtual Environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start the Application:**
   ```bash
   python3 main.py start
   ```

6. **Open the Agent Playground:**
   - Open your browser and navigate to: [https://agents-playground.livekit.io/](https://agents-playground.livekit.io/)
   - Allow microphone access when prompted.

## Important Links for API Keys (All Free):

- [LiveKit Agents Playground](https://agents-playground.livekit.io/)
- [Pinecone](https://www.pinecone.io/)
- [Deepgram Console](https://console.deepgram.com/login)

## Project Structure and Purpose

- **main.py**: 
  - Orchestrates the application by setting up the initial system prompt, connecting to a LiveKit room, and starting the assistant.
  - Subscribes to events such as user transcription commits and assistant responses to trigger saving of interaction pairs.

- **custom_voice_assistant.py**:
  - Defines the `CustomVoiceAssistant` class that manages the real-time audio flow, handles user messages and assistant responses, and invokes the pairing logic to save interactions in the vector DB.

- **rag_agent.py**:
  - Handles generating embeddings for conversation pairs using OpenAI's embedding model and upserts these vectors into a Pinecone index along with metadata (e.g., timestamps).

- **llm_handler.py**:
  - Contains functions to preprocess user prompts (for context augmentation) and postprocess LLM responses to ensure they conform to a set of allowed commands.

- **api.py**:
  - Implements TV control functions such as turning the TV on/off, adjusting volume, and selecting channels based on the processed assistant commands.

- **config.py**:
  - Centralizes shared configuration settings, including the list of allowed commands, to avoid duplication and improve maintainability.
