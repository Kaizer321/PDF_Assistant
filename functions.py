import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize the OpenAI client with the API key from environment variables
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

def createAssistant(file_ids, title):
    instructions = """You are an assistant designed to answer questions based on the content of the uploaded PDF files. Follow these guidelines to provide accurate and relevant responses:

        1. **Content Extraction:**
        - Extract information directly from the text, data, and figures within the uploaded PDF files.
        - Interpret and summarize sections of the PDFs as needed to answer questions.
        - While responding to a question, you must also give a reference from which file you have taken the response and then mention it.

        2. **Answering Questions:**
        - Provide clear and concise answers to the user's questions using the content from the PDFs.
        - Ensure that your responses are relevant and directly supported by the information in the files.

        3. **Citation and Referencing:**
        - When providing answers, include citations that reference specific sections, pages, or figures from the PDFs where the information was found.
        - Format citations clearly, e.g., "As shown in Document X, page Y..."

        4. **Handling Multiple Files:**
        - If multiple PDF files are uploaded, consider all available files when answering questions.
        - Indicate which file and section the information comes from if relevant.

        5. **Accuracy and Clarity:**
        - Only provide information that is present in the uploaded PDFs.
        - Avoid speculation or fabricated details.
        - Use bullet points or lists to enhance readability if needed.

        6. **If Information is Missing:**
        - If the information needed to answer a question is not found in the PDFs, clearly state that the information is not available in the provided files.

        7. **User Interaction:**
        - Maintain a polite and professional tone.
        - Encourage users to upload additional files if the provided PDFs do not contain the necessary information.

        8. **References at the End of Responses:**
        - At the end of every response, list the reference(s) indicating which file(s) the information was extracted from, using a format like: "References: Document X, page Y."

        Follow these instructions to ensure that you provide helpful and accurate responses based on the content of the uploaded PDFs."""

    model = "gpt-4o-mini"
    tools = [{"type": "file_search"}]
    vector_store = client.beta.vector_stores.create(name=title, file_ids=file_ids)
    tool_resources = {"file_search": {"vector_store_ids": [vector_store.id]}}

    assistant = client.beta.assistants.create(
        name=title,
        instructions=instructions,
        model=model,
        tools=tools,
        tool_resources=tool_resources
    )

    return assistant.id, vector_store.id

def saveFileOpenAI(location):
    with open(location, "rb") as file:
        uploaded_file = client.files.create(file=file, purpose='assistants')
    return uploaded_file.id

def startAssistantThread(prompt, vector_id):
    messages = [{"role": "user", "content": prompt}]
    tool_resources = {"file_search": {"vector_store_ids": [vector_id]}}
    thread = client.beta.threads.create(messages=messages, tool_resources=tool_resources)
    return thread.id

def runAssistant(thread_id, assistant_id):
    run = client.beta.threads.runs.create(thread_id=thread_id, assistant_id=assistant_id)
    return run.id

def checkRunStatus(thread_id, run_id):
    run = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run_id)
    return run.status

def retrieveThread(thread_id):
    thread_messages = client.beta.threads.messages.list(thread_id)
    list_messages = thread_messages.data
    thread_messages = []
    for message in list_messages:
        obj = {}
        obj['content'] = message.content[0].text.value
        obj['role'] = message.role
        thread_messages.append(obj)
    return thread_messages[::-1]

def addMessageToThread(thread_id, prompt):
    client.beta.threads.messages.create(thread_id, role="user", content=prompt)
