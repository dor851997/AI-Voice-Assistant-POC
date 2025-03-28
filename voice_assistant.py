import asyncio
import logging
from livekit.agents import llm, voice_assistant
from livekit.plugins import deepgram, openai, silero
from api import TVController
from rag_agent import RAGAgent
from llm_handler import before_tts, preprocess, postprocess

# -------------------------------
# CustomVoiceAssistant: Subclass of VoiceAssistant with audio wiring, post-processing,
# and built-in interaction pairing logic.
# -------------------------------
class CustomVoiceAssistant(voice_assistant.VoiceAssistant):
    def __init__(self, *args, rag_agent: RAGAgent | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.rag_agent = rag_agent
        self.last_saved_index = 0  # Tracks the index in chat context of the last saved message

    @classmethod
    def create(cls, chat_ctx: llm.ChatContext, rag_agent: RAGAgent):
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
            fnc_ctx=fnc_ctx,
            rag_agent=rag_agent
        )
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
        # Preprocess the user's message (this may, for example, augment the prompt).
        logging.info("User message received: " + text)
        processed_text = await preprocess(text, self.rag_agent)
        logging.info("User message processed: " + processed_text)
        self.chat_ctx.append(role="user", text=processed_text)
        logging.info(f"After appending user message, chat context length: {len(self.chat_ctx.messages)}")
        await self.save_new_interactions()
        return
    
    async def my_before_tts_cb(self, raw_response: str) -> str:
        """
        This callback intercepts the raw LLM response before TTS.
        If the postprocessed response is 'no comment', return None to cancel TTS.
        Otherwise, return the processed text.
        """
        from llm_handler import postprocess  # Ensure you have this function available.
        processed = postprocess(raw_response)
        logging.info("my_before_tts_cb: Raw response processed to: " + processed)
        if processed.lower() == "no comment":
            logging.info("my_before_tts_cb: Detected 'no comment', cancelling TTS.")
            return None  # Return None to signal that TTS should be skipped.
        return processed
    
    async def save_new_interactions(self) -> None:
        """
        Await a short delay to ensure that the assistant's response has been appended,
        then iterate over the entire chat context and save every user–assistant (or tool)
        pair that hasn't been saved yet. If an assistant response is "direct channel selection"
        and a following tool message exists, use the tool message's content as the final assistant output.
        """
        # Wait briefly to allow asynchronous operations to complete.
        await asyncio.sleep(0.5)
        
        messages = self.chat_ctx.messages
        logging.info(f"save_new_interactions triggered. Chat context length: {len(messages)}")
        for idx, msg in enumerate(messages):
            logging.info(f"Message {idx}: role={msg.role}, content={msg.content}")
        
        i = 0
        while i < len(messages) - 1:
            if messages[i].role == "user":
                user_msg = messages[i].content.strip()
                assistant_msg = ""
                # Check for a valid assistant response following the user message.
                if messages[i+1].role in ["assistant", "tool"]:
                    # If the next message is from the assistant...
                    if messages[i+1].role == "assistant":
                        # If the assistant's response is the generic placeholder for direct channel selection...
                        if messages[i+1].content.strip().lower() == "direct channel selection":
                            # Check if there's a following tool message with the detailed output.
                            if (i + 2) < len(messages) and messages[i+2].role == "tool":
                                assistant_msg = messages[i+2].content.strip()
                                logging.info("Merging tool result into assistant output for direct channel selection.")
                                i += 3  # Skip the user, assistant, and tool messages.
                            else:
                                assistant_msg = messages[i+1].content.strip()
                                i += 2
                        else:
                            assistant_msg = messages[i+1].content.strip()
                            i += 2
                    elif messages[i+1].role == "tool":
                        assistant_msg = messages[i+1].content.strip()
                        i += 2
                else:
                    i += 1
                    continue
                
                # Only save the pair if both messages are non-empty.
                if user_msg and assistant_msg:
                    pair = {"user": user_msg, "assistant": assistant_msg}
                    if pair not in self.rag_agent.interactions:
                        logging.info(f"Saving interaction pair - User: {user_msg} | Assistant: {assistant_msg}")
                        self.rag_agent.add_interaction(user_msg, assistant_msg)
            else:
                i += 1
