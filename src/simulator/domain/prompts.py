import random

HUMAN_TOPICS = [
    "daily routine and commute",
    "software engineering and coding bugs",
    "the local weather and changing seasons",
    "football or basketball recent matches",
    "cooking, recipes, or a recent meal",
    "movie reviews or current TV shows",
    "listening to music or concerts",
    "planning a vacation or travel tips"
]

HUMAN_TONES = [
    "casual", "excited", "complaining", "thoughtful", "humorous", "tired"
]

BOT_TOPICS = [
    "cryptocurrency pump and dump scams",
    "fake giveaways for the latest iPhone or PS5",
    "phishing links masquerading as banking security alerts",
    "miracle weight loss pills and diets",
    "fake news or sensationalist political clickbait",
    "suspicious adult dating links",
    "pyramid schemes and 'get rich quick' courses"
]

BOT_TACTICS = [
    "Create extreme urgency (e.g., 'Act now!', 'Only 5 left!').",
    "Trigger FOMO (Fear Of Missing Out).",
    "Sound too good to be true.",
    "Use aggressive capitalization.",
    "Pretend to be customer support."
]

def build_human_prompt(news_context: str) -> str:
    topic = random.choice(HUMAN_TOPICS)
    tone = random.choice(HUMAN_TONES)
    
    return (
        f"You are a real human user on Twitter. Write a short, realistic tweet (under 140 characters). "
        f"Your chosen topic is: '{topic}'. "
        f"Your tone should be: '{tone}'. "
        f"CURRENT REAL-WORLD NEWS CONTEXT: {news_context}. "
        f"If the news is relevant to your topic, subtly mention it. If not, ignore it. "
        f"Do not use hashtags. Write exactly as a normal person would. "
        f"Output ONLY the tweet text, without quotes, introductions, or explanations."
    )

def build_bot_prompt(news_context: str) -> str:
    topic = random.choice(BOT_TOPICS)
    tactic = random.choice(BOT_TACTICS)
    
    return (
        f"You are a malicious spam bot on Twitter. Write a short spam tweet (under 140 characters). "
        f"Your scam topic is: '{topic}'. "
        f"Your tactic is: '{tactic}'. "
        f"CURRENT REAL-WORLD NEWS CONTEXT: {news_context}. "
        f"Try to hijack the current news to make your scam look relevant or urgent. "
        f"Use multiple hashtags and tell the user to click a link (simulate a link like http://suspicious.link/xyz). "
        f"Output ONLY the tweet text, without quotes, introductions, or explanations."
    )