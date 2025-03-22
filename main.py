import asyncio
import logging
import os
import re
import uuid
from typing import AsyncIterable
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm, voice_assistant
from livekit.plugins import deepgram, openai, silero
from api import TVController  # Import your tools from api.py

logging.basicConfig(level=logging.INFO)
load_dotenv()

# Global constant for allowed commands.
ALLOWED_COMMANDS = [
    "power on",
    "power off",
    "volume up",
    "volume down",
    "mute",
    "unmute",
    "next channel",
    "previous channel",
    "direct channel selection",
    "no comment"
]

def process_response(text: str) -> str:
    """
    Process the raw response text and force it to be one of the allowed commands.
    Special handling for 'direct channel selection' is applied.
    """
    normalized = text.strip().lower()
    if normalized.startswith("direct channel selection"):
        m = re.match(r"direct channel selection\s+(\d+)$", normalized)
        if m:
            return f"direct channel selection {m.group(1)}"
        else:
            return "no comment"
    if normalized not in ALLOWED_COMMANDS:
        return "no comment"
    return normalized

# -------------------------------
# RAGAgent: Saves interactions for retrieval-augmented generation using Pinecone
# -------------------------------
class RAGAgent:
    def __init__(self) -> None:
        # Initialize a list to store interactions for reference.
        self.interactions: list[dict[str, str]] = []

        # Use OpenAI's embedding model.
        from openai import OpenAI
        
        client = OpenAI()
        self.embedding_model_name = "text-embedding-ada-002"
        # text-embedding-ada-002 returns embeddings of dimension 1536.
        self.dim = 1536

        from pinecone import Pinecone, ServerlessSpec

        pinecone_api_key = os.getenv("PINECONE_API_KEY")
        pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
        # Normalize the environment: if it ends with '-aws', remove that suffix so the region is correct
        if pinecone_environment.endswith("-aws"):
            normalized_region = pinecone_environment.replace("-aws", "")
        else:
            normalized_region = pinecone_environment

        # Create an instance of Pinecone using the new API
        pc = Pinecone(api_key=pinecone_api_key)

        self.index_name = "tv-interactions"

        # Check if the index exists using the new API; create it if it does not
        if self.index_name not in pc.list_indexes().names():
            pc.create_index(
                name=self.index_name,
                dimension=self.dim,
                metric='euclidean',
                spec=ServerlessSpec(cloud='aws', region=normalized_region)
            )

        # Obtain the index instance
        self.pinecone_index = pc.Index(self.index_name)

        # List to store interaction texts corresponding to embeddings.
        self.interaction_texts = []

        logging.info(f"Pinecone index '{self.index_name}' initialized with dimension: {self.dim}")

    def add_interaction(self, user_message: str, assistant_message: str) -> None:
        stats = self.pinecone_index.describe_index_stats()
        logging.info(f"Index stats after upsert: {stats}")
        interaction = {"user": user_message, "assistant": assistant_message}
        self.interactions.append(interaction)

        # Concatenate messages.
        combined_text = user_message + " " + assistant_message

        # Compute embedding using OpenAI.
        from openai import OpenAI
        
        client = OpenAI()
        try:
            embedding_response = client.embeddings.create(model=self.embedding_model_name,
            input=combined_text)
            embedding = embedding_response.data[0].embedding
        except Exception as e:
            logging.error(f"Error obtaining embedding for interaction: {e}")
            return

        # Use a unique ID for the vector.
        vector_id = str(uuid.uuid4())
        import datetime
        timestamp = datetime.datetime.utcnow().isoformat()
        metadata = {
            "user": user_message,
            "assistant": assistant_message,
            "text": combined_text,
            "timestamp": timestamp
        }        
        upsert_response = self.pinecone_index.upsert(vectors=[{"id": vector_id, "values": embedding, "metadata": metadata}])
        logging.info(f"Upsert response: {upsert_response}")
        self.interaction_texts.append(combined_text)

        logging.info(f"RAG stored interaction in Pinecone with id {vector_id}: {interaction}")

    async def augment(self, prompt: str) -> str:
        """
        Augment the prompt using external context.
        For now, returns the prompt unchanged.
        """
        return prompt

    def search_interactions(self, query: str, top_k: int = 5) -> list[str]:
        """
        Search for similar interactions based on a query.
        
        Args:
            query (str): The input query to search for.
            top_k (int): The number of top matching interactions to return.
            
        Returns:
            list[str]: A list of interaction texts that closely match the query.
        """
        from openai import OpenAI
        
        client = OpenAI()
        try:
            embedding_response = client.embeddings.create(model=self.embedding_model_name,
            input=query)
            query_embedding = embedding_response.data[0].embedding
        except Exception as e:
            logging.error(f"Error obtaining embedding for search query: {e}")
            return []

        # Query Pinecone.
        result = self.pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        matches = result.get("matches", [])
        return [match["metadata"]["text"] for match in matches]


# -------------------------------
# Custom before_llm_cb callback
# -------------------------------
async def custom_before_llm_cb(assistant: voice_assistant.VoiceAssistant, chat_ctx: llm.ChatContext) -> None:
    """
    Invoked before the LLM call.
    Augments the user prompt (last message) using the RAGAgent.
    """
    if chat_ctx.messages and chat_ctx.messages[-1].role == "user":
        prompt = chat_ctx.messages[-1].content
        logging.info("Before LLM callback - original prompt: " + prompt)
        augmented_prompt = await assistant.rag_agent.augment(prompt)
        logging.info("Before LLM callback - augmented prompt: " + augmented_prompt)
        chat_ctx.messages[-1].content = augmented_prompt
    return None

# -------------------------------
# CustomVoiceAssistant: Subclass of VoiceAssistant with audio wiring, post-processing,
# and built-in interaction pairing logic.
# -------------------------------
class CustomVoiceAssistant(voice_assistant.VoiceAssistant):
    def __init__(self, *args, after_llm_cb=None, rag_agent: RAGAgent | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.after_llm_cb = after_llm_cb  # External callback (optional)
        self.rag_agent = rag_agent
        self.last_saved_index = 0  # Tracks the index in chat context of the last saved message

    @classmethod
    def create(cls, before_llm_cb, after_llm_cb, chat_ctx: llm.ChatContext, rag_agent: RAGAgent):
        """
        Create an instance with the required audio components and callbacks.
        """
        vad = silero.VAD.load()
        stt = deepgram.STT()   # Deepgram STT
        tts = openai.TTS()      # OpenAI TTS
        llm_model = openai.LLM()  # LLM
        fnc_ctx = TVController()  # Tools from api.py
        instance = cls(
            vad=vad,
            stt=stt,
            tts=tts,
            llm=llm_model,
            chat_ctx=chat_ctx,
            before_llm_cb=before_llm_cb,
            fnc_ctx=fnc_ctx,
            after_llm_cb=after_llm_cb,
            rag_agent=rag_agent
        )
        # Set the callback for final transcription, so that when STT produces a final transcript,
        # it calls our handler which in turn appends the message.
        stt.on_final_transcription = instance.handle_final_transcription
        return instance

    async def on_user_message(self, text):
        """
        Override on_user_message to ensure user messages are appended to the chat context.
        Immediately trigger saving of any complete interaction pairs.
        """
        # If text is not a string, try to extract its 'content' attribute
        if not isinstance(text, str):
            try:
                text = text.content
            except Exception as e:
                logging.error(f"Error extracting content from message: {e}")
                return
        logging.info("User message received: " + text)
        self.chat_ctx.append(role="user", text=text)
        logging.info(f"After appending user message, chat context length: {len(self.chat_ctx.messages)}")
        self.save_new_interactions()
        return

    def save_new_interactions(self) -> None:
        """
        Iterate over the entire chat context and save every user–assistant pair
        that hasn't been saved yet. This method now treats messages with role "assistant" or "tool"
        as valid assistant responses.
        """
        messages = self.chat_ctx.messages
        logging.info(f"save_new_interactions triggered. Chat context length: {len(messages)}")
        for idx, msg in enumerate(messages):
            logging.info(f"Message {idx}: role={msg.role}, content={msg.content}")

        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user" and messages[i+1].role in ["assistant", "tool"]:
                pair = {"user": messages[i].content, "assistant": messages[i+1].content}
                if pair not in self.rag_agent.interactions:
                    logging.info(f"Saving interaction pair - User: {pair['user']} | Assistant: {pair['assistant']}")
                    self.rag_agent.add_interaction(pair["user"], pair["assistant"])
                i += 2  # Skip this pair since we've processed it.
            else:
                i += 1

    async def say(self, source: str | llm.LLMStream | AsyncIterable[str], *,
                  allow_interruptions=True, add_to_chat_ctx=True) -> any:
        """
        Override say() to log raw and post-processed responses.
        After calling the base say(), it saves new interaction pairs.
        """
        if not isinstance(source, str):
            text = str(source)
        else:
            text = source
        logging.info("Raw say() text: " + text)
        final_response = process_response(text)
        logging.info("Final post-processed say() response: " + final_response)
        result = await super().say(final_response, allow_interruptions=allow_interruptions,
                           add_to_chat_ctx=add_to_chat_ctx)
        if add_to_chat_ctx:
            self.chat_ctx.append(role="assistant", text=final_response)
            logging.info(f"After appending assistant message, chat context length: {len(self.chat_ctx.messages)}")
        self.save_new_interactions()
        if self.after_llm_cb:
            await self.after_llm_cb(self, self.chat_ctx)
        return result
    
    async def handle_final_transcription(self, transcript: str):
        """
        This method is called when the STT engine produces a final transcript.
        It logs the transcript and forwards it to on_user_message.
        """
        logging.info("Final transcription received: " + transcript)
        await self.on_user_message(transcript)

# -------------------------------
# Orchestrator (entrypoint)
# -------------------------------
async def entrypoint(ctx: JobContext) -> None:
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=f"You are a TV voice assistant. Your responses must be exactly one of the following: {', '.join(ALLOWED_COMMANDS)}. "
             "If it is unclear what command to execute, respond with 'no comment' and do not include any extra text."
    )
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    rag_agent = RAGAgent()

    # For testing, simulate a user message after the greeting.
    assistant = CustomVoiceAssistant.create(custom_before_llm_cb, None, initial_ctx, rag_agent)
    participant = await ctx.wait_for_participant()
    logging.info("Participant connected: " + participant.identity)
    assistant.start(ctx.room, participant)
    async def handle_committed_transcript(transcript: str):
        logging.info(f"Committed transcript event received: {transcript}")
        await assistant.on_user_message(transcript)

    # Subscribe to the event that indicates the user's speech has been finalized
    def handle_committed_transcript_wrapper(transcript: str):
        asyncio.create_task(handle_committed_transcript(transcript))
    assistant.on("user_speech_committed", handle_committed_transcript_wrapper)
    # # Initial greeting
    # await assistant.say("Hello, how can I help you today?", allow_interruptions=True)

    # # Simulate a user message
    # await assistant.on_user_message("test1, Turn on the TV.")
    # # Simulate the assistant's response to the user message
    # await assistant.say("power on", allow_interruptions=True)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))