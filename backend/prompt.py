def generate_prompt(user_message, sentiment, emotion):
    """
    Generate a flowing, natural-sounding prompt for the Gemini API based on user's sentiment and emotion.

    The assistant should respond like a close friend: warm, relaxed, and conversational—no repeating the user's message.
    """
    prompt = (
        f"You're a caring, chill mental health companion. "
        f"Don't repeat what the user said—just respond naturally like a close friend would. "
        f"Keep it short, warm, and human. "
        f"User's mood: {sentiment}, feeling: {emotion}. "
        f"User: '{user_message}' "
        f"Respond with a message that keeps the conversation flowing."
    )

    return prompt
