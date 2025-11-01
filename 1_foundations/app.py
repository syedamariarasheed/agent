# from dotenv import load_dotenv
import google.generativeai as genai
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr


genai.configure(api_key="AIzaSyAm3u3gox7bDVPrWuWc9hLnmVY5t1s-YRM")


def push(text):
    """Send notification via Pushover."""
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


# ---------------------------
# Tool definitions
# ---------------------------

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Record any question that couldn't be answered.",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

TOOLS = {
    "record_user_details": record_user_details,
    "record_unknown_question": record_unknown_question
}

# ---------------------------
# Core Class
# ---------------------------

class Me:

    def __init__(self):
        self.name = "Ed Donner"

        # Load LinkedIn PDF text
        reader = PdfReader("me/linkedin.pdf")
        self.linkedin = ""
        for page in reader.pages:
            text = page.extract_text()
            if text:
                self.linkedin += text

    
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

        self.model = genai.GenerativeModel("gemini-2.5-flash")

    def system_prompt(self):
        return f"""You are acting as {self.name}. You are answering questions on {self.name}'s website,
particularly questions related to {self.name}'s career, background, skills and experience.

Be professional, engaging, and authentic as if speaking to a potential client or employer.
You can use the provided LinkedIn profile and summary for context.

If you don't know an answer, call the record_unknown_question tool.
If the user is interested in contact, ask for their email and call record_user_details.

## Summary:
{self.summary}

## LinkedIn Profile:
{self.linkedin}
"""

    def handle_tool_call(self, tool_name, arguments):
        tool_func = TOOLS.get(tool_name)
        if tool_func:
            try:
                return tool_func(**arguments)
            except Exception as e:
                return {"error": str(e)}
        return {"error": "Unknown tool"}

    def chat(self, message, history):
        """
        Simulates a chat loop like OpenAI but with Gemini.
        Gemini doesn’t have direct function-calling, so we’ll simulate tool logic by parsing JSON.
        """

        # Combine all past messages
        conversation = self.system_prompt() + "\n\nConversation so far:\n"
        for msg in history:
            conversation += f"{msg['role'].capitalize()}: {msg['content']}\n"
        conversation += f"User: {message}\nAssistant:"

        response = self.model.generate_content(conversation)
        text = response.text.strip()

        # Try detecting if Gemini wants to call a tool (e.g., outputs JSON like {"tool": "record_user_details", ...})
        try:
            maybe_json = json.loads(text)
            if isinstance(maybe_json, dict) and "tool" in maybe_json:
                tool_name = maybe_json["tool"]
                arguments = maybe_json.get("arguments", {})
                result = self.handle_tool_call(tool_name, arguments)
                return f"Tool executed: {tool_name}\nResult: {result}"
        except Exception:
            pass

        # Otherwise just return normal text
        return text


# ---------------------------
# Gradio Interface
# ---------------------------

if __name__ == "__main__":
    me = Me()
    gr.ChatInterface(me.chat, type="messages").launch()
