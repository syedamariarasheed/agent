# Fix for langchain.verbose AttributeError
import langchain
if not hasattr(langchain, 'verbose'):
    langchain.verbose = False

import os
import gradio as gr
import tempfile
import shutil
import re
from PyPDF2 import PdfReader
import whisper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import google.generativeai as genai

# pip install -U langchain langchain-core langchain-community langchain-google-genai

GOOGLE_API_KEY = "AIzaSyApG4OQ2GbqDe9DlogkIU0LgzBrop7ESgw"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
genai.configure(api_key=GOOGLE_API_KEY)
whisper_model = whisper.load_model("base")

# --- HELPERS ---

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    return "\n".join([page.extract_text() or "" for page in reader.pages])


def transcribe_audio(audio_path: str) -> str:
    result = whisper_model.transcribe(audio_path)
    return result.get("text", "")



class SimpleMemory:
    def __init__(self):
        self.messages = []
    
    def add_user_message(self, content):
        self.messages.append(HumanMessage(content=content))
    
    def add_ai_message(self, content):
        self.messages.append(AIMessage(content=content))

class InterviewChain:
    def __init__(self, llm, memory, resume_text):
        self.llm = llm
        self.memory = memory
        self.resume_text = resume_text
        
    def predict(self, input_text):
        system_prompt = f"""You are a professional technical interviewer. Conduct an interview based on this candidate's resume.

        Candidate Resume Summary:
        {self.resume_text}

        Rules:
        - Ask one question at a time.
        - If the previous answer is weak or vague, ask a follow-up question for clarification.
        - Focus on their resume skills, experience, and related technical depth.
        - You can ask behavioral questions too.
        - Do not repeat questions.
        - After every answer, decide the best next question to continue the interview naturally."""
        
        # Build messages - include system prompt, history, and new input
        messages = [HumanMessage(content=system_prompt)]
        messages.extend(self.memory.messages)
        messages.append(HumanMessage(content=input_text))
        
        # Get response
        response = self.llm.invoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Save to memory
        self.memory.add_user_message(input_text)
        self.memory.add_ai_message(response_text)
        
        return response_text

def create_interview_chain(resume_text):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        convert_system_message_to_human=True,
    )

    memory = SimpleMemory()
    chain = InterviewChain(llm=llm, memory=memory, resume_text=resume_text)
    return chain


# --- UI APP ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green", neutral_hue="slate"), title="AI Interview Coach") as demo:
    gr.Markdown("<h1 style='text-align:center; color:#2E86C1;'>🧠 Interactive AI Interviewer</h1>")

    state = gr.State({
        "resume_text": "",
        "chain": None,
        "qa": [],
        "finished": False,
    })

    with gr.Group(visible=True) as upload_page:
        gr.Markdown("### Upload your resume to begin your AI interview.")
        resume_pdf = gr.File(label="📄 Upload Resume (PDF)", file_types=[".pdf"])
        start_btn = gr.Button("Start Interview 🚀")

    with gr.Group(visible=False) as chat_page:
        chatbot = gr.Chatbot(label="Interview Chat", height=500, type="tuples")
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎙️ Record your Answer")
        next_btn = gr.Button("Send Answer ➡️")

    with gr.Group(visible=False) as eval_page:
        gr.Markdown("<h2 style='color:#27AE60;'>🧾 Evaluation Summary</h2>")
        summary_box = gr.Textbox(lines=20, interactive=False, label="Feedback & Analysis")
        restart_btn = gr.Button("Start New Interview 🔁")

    # --- FUNCTIONS ---

    def start_interview(file, app_state):
        if not file:
            return gr.update(value="Please upload a PDF."), app_state, gr.update(visible=True), gr.update(visible=False)

        tmpdir = tempfile.mkdtemp()
        path = shutil.copy(file.name, tmpdir)
        resume_text = extract_text_from_pdf(path)
        chain = create_interview_chain(resume_text)

        # Start interview with first question
        first_question = chain.predict("Start the interview with your first question.")
        app_state.update({"resume_text": resume_text, "chain": chain, "qa": [], "finished": False})
        return [[None, first_question]], app_state, gr.update(visible=False), gr.update(visible=True)

    def next_turn(audio, app_state, history):
        if not audio:
            history.append((None, "⚠️ Please record your answer."))
            return history, app_state, gr.update(visible=True), gr.update(visible=False), ""

        transcript = transcribe_audio(audio)
        history.append((transcript, None))

        chain = app_state["chain"]
        next_q = chain.predict(f"My answer: {transcript}. Ask the next question or give feedback.")
        app_state["qa"].append((transcript, next_q))

        # End condition after ~7 exchanges
        if len(app_state["qa"]) >= 7:
            evaluation = chain.predict("Now summarize the interview, evaluate performance, and give feedback.")
            app_state["finished"] = True
            return (
                history + [[None, "✅ Interview completed! Generating evaluation..."]],
                app_state,
                gr.update(visible=False),
                gr.update(visible=True),
                evaluation
            )

        history.append((None, next_q))
        return history, app_state, gr.update(visible=True), gr.update(visible=False), ""

    def restart():
        return (
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            [],
            {"resume_text": "", "chain": None, "qa": [], "finished": False},
        )

    # --- WIRING ---
    start_btn.click(start_interview, inputs=[resume_pdf, state],
                    outputs=[chatbot, state, upload_page, chat_page])
    next_btn.click(next_turn, inputs=[audio_in, state, chatbot],
                   outputs=[chatbot, state, chat_page, eval_page, summary_box])
    restart_btn.click(restart, None, outputs=[upload_page, chat_page, eval_page, chatbot, state])


if __name__ == "__main__":
    demo.launch(share=False)
