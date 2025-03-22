import logging
from config import ALLOWED_COMMANDS

async def preprocess(prompt: str, rag_agent) -> str:
    """
    Preprocess the user prompt using the provided RAGAgent.
    This function can be extended to include additional NLP preprocessing.
    """
    logging.info("LLM Handler: Preprocessing prompt: " + prompt)
    # For now, we simply use the rag_agent's augment method
    augmented_prompt = await rag_agent.augment(prompt)
    logging.info("LLM Handler: Augmented prompt: " + augmented_prompt)
    return augmented_prompt

def postprocess(response: str) -> str:
    """
    Postprocess the LLM response ensuring it is one of the allowed commands.
    """
    normalized = response.strip().lower()
    if normalized not in ALLOWED_COMMANDS:
        return "no comment"
    return normalized

def before_tts(text: str) -> str:
    processed_text = postprocess(text)
    # Log the processed text for debugging
    import logging
    logging.info("LLM response after before_tts processing: " + processed_text)
    return processed_text