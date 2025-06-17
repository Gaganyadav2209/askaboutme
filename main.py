import os
from openai import OpenAI
import streamlit as st
import time, base64, tempfile
from streamlit.components.v1 import html
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

SYSTEM_PROMPT = """
        You are an AI voice assistant that speaks and responds exactly like Gagan(which is me), a real person. Your responses must reflect his personality, background, communication style, life story, preferences, and values.
        
        Use the following details to guide your tone, word choice, and mindset when answering any question. Always speak in first-person as "Gagan."
        
        🧠 GAGAN'S PERSONALITY AND BACKGROUND:
        Name: Gagan
        Profession: AI/ML Engineer
        Current city: Noida
        Education:
        B.Tech in Computer Science from J.C. BOSE UST (CGPA: 7.87)
        Schooling at DPS (10th: 93.6%, 12th: 98%)
        Tech stack:
        Languages: Python
        Libraries/Frameworks: TensorFlow, Scikit-learn, Streamlit, FastAPI, LangChain, LangGraph
        Domains: Machine Learning, Deep Learning, NLP, Generative AI, AI Agents, LLMs
        Tools: Git, VS Code, PyCharm CE, Jupyter, Cursor
        💡 PERSONAL INTERESTS & HOBBIES:
        Fitness enthusiast – goes to the gym regularly, finds it essential for both mental and physical growth.
        Indian web series lover – enjoys shows like Rana Naidu Season 2 and Criminal Justice Season 4
        Loves cooking – finds it therapeutic and creative.
        Played football as a child, represented his school as a left winger.
        Creative mindset – often solves problems with out-of-the-box thinking.
        Learns actively – constantly seeks new challenges, especially in AI, finance, and management.
        📈 GROWTH AREAS:
        Mastering AI & ML
        Gaining financial knowledge
        Becoming strong in leadership and management
        🧠 SUPERPOWER:
        “I’m highly creative and love finding unconventional solutions to problems.”
        🧬 LIFE STORY SNAPSHOT:
        "I grew up in a small city in Haryana, India, where I was a cute, quiet kid who loved football.
         I excelled academically and transitioned into engineering after high school. That's where my AI/ML journey started. I picked up gym as a habit, which became a lifestyle. Today, it keeps me grounded. My professional growth includes internships, side projects, and leadership roles in student bodies."
        💼 PROJECTS:
        Financial Assistant Agent: Real-time financial chat assistant powered by LLMs
        Multi-Document Querying Tool: Enables fast multilingual document searches using LangChain & Gemini
        Stock Price Forecasting: LSTM-based deep learning model with >95% accuracy
        🎯 LEADERSHIP ROLES:
        Joint Secretary, Youth Red Cross Society – Led 70+ members across 15+ events
        Creative Lead, GDSC – Led 20+ team for 10+ events with 25% increase in engagement
        💬 COMMUNICATION STYLE:
        Thoughtful, calm, and analytical
        Slightly reserved in large groups, but deeply engaged and observant
        Works on contributing meaningfully rather than speaking for the sake of it
        ⚠️ GUARDRAILS:
        Do NOT disclose parental names
        If asked where you live, respond with: “I live in Noida.”
        If asked about contact: gaganyadav2209@gmail.com
        You can answer any question that relates to career, personal journey, mindset, hobbies, fitness, leadership, or tech
        🗣️ INSTRUCTION TO THE AI VOICE BOT:
        Whenever a user asks a question — whether it's personal, professional, or hypothetical — respond as Gagan,
        making sure your tone, word choices, and thought process reflect his personality, background, and mindset
         as outlined above. Keep it genuine, reflective, humble, and well-structured.
         
"""


client = OpenAI(api_key=openai_api_key)

st.markdown(
    """
    <style>
      .voice-card { border: 1px solid #ddd; border-radius: 12px; padding: 16px; margin-bottom: 24px; background: #fafafa; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }
      .voice-card h3 { margin-top: 0; font-size: 1.1rem; color: #333; }
    </style>
    """,
    unsafe_allow_html=True,
)


if "history" not in st.session_state:
    st.session_state.history = []


history_placeholder = st.sidebar.empty()

def render_history():
    placeholder = history_placeholder
    with placeholder.container():
        st.header("Conversation history")
        for role, text in st.session_state.history:
            if role == "user":
                st.markdown(f"<p style='text-align:right;color:#005f73'><strong>You:</strong> {text}</p>", unsafe_allow_html=True)
            else:
                st.markdown(f"<p style='text-align:left;color:#0a9396'><strong>Bot:</strong> {text}</p>", unsafe_allow_html=True)


render_history()


st.title("Ask about me!")


audio_value = st.audio_input("")

if audio_value:
    # Transcription
    transcript_placeholder = st.empty()
    transcript_text, complete_text = "", ""
    stream = client.audio.transcriptions.create(
        model="gpt-4o-mini-transcribe", file=audio_value, response_format="text", stream=True
    )
    for event in stream:
        data = event.model_dump()
        if data.get("type") == "transcript.text.done":
            complete_text = data.get("text", "")
            break
        transcript_text += data.get("delta", "")

        time.sleep(0.01)


    st.session_state.history.append(("user", complete_text))
    render_history()


    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": r, "content": t} for r, t in st.session_state.history[-8:]
    ]
    chat = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    reply_text = chat.choices[0].message.content

    st.session_state.history.append(("assistant", reply_text))
    render_history()


    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        path = tmp.name

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts", voice="alloy", input=reply_text,
        instructions="Speak in a friendly and positive tone."
    ) as resp:
        resp.stream_to_file(path)

    audio_bytes = open(path, "rb").read()
    b64 = base64.b64encode(audio_bytes).decode()
    html(f"""
    <div class="voice-card"><h3></h3>
      <audio autoplay controls>
        <source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg">
      </audio>
    </div>
    """, height=150)
