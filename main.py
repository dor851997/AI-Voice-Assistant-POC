import asyncio
import logging
from dotenv import load_dotenv
from livekit.agents import AutoSubscribe, JobContext, WorkerOptions, cli, llm
from rag_agent import RAGAgent
from voice_assistant import CustomVoiceAssistant
from config import ALLOWED_COMMANDS

# Load configuration and initialize logging
logging.basicConfig(level=logging.INFO)
load_dotenv()


# -------------------------------
# Event Subscription Helpers
# -------------------------------
async def handle_committed_transcript(transcript: str, assistant: CustomVoiceAssistant):
    logging.info(f"Committed transcript event received: {transcript}")
    await assistant.on_user_message(transcript)


def subscribe_to_committed_transcript(assistant: CustomVoiceAssistant):
    def committed_transcript_wrapper(transcript: str):
        asyncio.create_task(handle_committed_transcript(transcript, assistant))
    assistant.on("user_speech_committed", committed_transcript_wrapper)


async def handle_assistant_response(response: str):
    from llm_handler import postprocess  # Import postprocess from llm_handler
    processed_response = postprocess(response)
    logging.info(f"Post-processed assistant response: {processed_response}")
    # Optionally, update chat context or trigger further actions here.


def subscribe_to_assistant_response(assistant: CustomVoiceAssistant):
    def assistant_response_wrapper(response: str):
        asyncio.create_task(handle_assistant_response(response))
    assistant.on("assistant_response", assistant_response_wrapper)


# -------------------------------
# Orchestrator (Entry Point)
# -------------------------------
async def entrypoint(ctx: JobContext) -> None:
    # Set up the initial conversation context with a system prompt
    initial_ctx = llm.ChatContext().append(
        role="system",
        text=(
            f"You are a TV voice assistant. Your responses must be exactly one of the following: {', '.join(ALLOWED_COMMANDS)}. "
            "If it is unclear what command to execute, respond with 'no comment' and do not include any extra text. "
            "If your response is 'direct channel selection' also specify the channel number."
        )
    )

    # Connect to the room with audio only
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)

    # Instantiate the RAG agent and assistant
    rag_agent = RAGAgent()
    assistant = CustomVoiceAssistant.create(initial_ctx, rag_agent)
    
    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    logging.info("Participant connected: " + participant.identity)
    
    # Start the assistant in the room with the participant
    assistant.start(ctx.room, participant)

    # Subscribe to assistant events for transcript and response handling
    subscribe_to_committed_transcript(assistant)
    subscribe_to_assistant_response(assistant)


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))