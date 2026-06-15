from dotenv import load_dotenv
from langchain_groq import ChatGroq
from pypdf import PdfReader
from langchain.tools import tool
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage
)
import os
from pathlib import Path
from rich import print







load_dotenv()

# modle======================================


llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# pdf tool==================================
@tool 
def extract_pdf(pdf_path):
    """Read a PDF and return its text content."""

    if not Path(pdf_path).exists():
        return f"Error: PDF '{pdf_path}' was not found."
    
    reader = PdfReader(pdf_path)


    text = ""


    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

tools={
    "extract_pdf": extract_pdf
}
messages = []
llm_with_tool = llm.bind_tools([extract_pdf])




while True:
    prompt = input("You: ")

    if prompt.lower() == "exit":
        break

    messages.append(HumanMessage(content=prompt))

    result = llm_with_tool.invoke(messages)

    if result.tool_calls:

     for tool_call in result.tool_calls:

        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        print(f"\nTool wants to run: {tool_name}")
        print(f"Arguments: {tool_args}")

        approval = input("Approve tool call? (y/n): ")

        if approval.lower() != "y":
            print("Tool execution cancelled.")
            continue

        tool_result = tools[tool_name].invoke(tool_args)
        messages.append(ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]))

        print(messages)



# respose= llm.invoke("hello")
# print(respose.content)