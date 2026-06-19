from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain_core.messages import HumanMessage
from pypdf import PdfReader
from pathlib import Path
from langchain.tools import tool
from fpdf import FPDF
from rich import print
from langchain_core.messages import HumanMessage, AIMessage
import os

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)


@tool
def extract_pdf(pdf_name: str) -> str:
    """
    Find and read a PDF from the notes folder.
    Pass a keyword from the filename like 'DBMS', 'math'.
    If you don't know the filename, pass 'list' to see all available PDFs.
    Returns: page count and full text content of the PDF.
    """
    notes_path = Path("notes")
    available = [p.name for p in notes_path.glob("*.pdf")]

    if not available:
        return "No PDFs found in notes folder."

    if pdf_name.lower() == "list":
        return f"Available PDFs: {available}"

    for pdf in notes_path.glob("*.pdf"):
        if pdf_name.lower() in pdf.name.lower():
            reader = PdfReader(pdf)
            page_count = len(reader.pages)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return f"File: {pdf.name} | Pages: {page_count}\n\n{text[:8000]}"

    return f"No PDF matching '{pdf_name}' found. Available PDFs: {available}"


@tool
def save_as_txt(filename: str, content: str) -> str:
    """
    Save content to a text file.
    """

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / f"{filename}.txt"

    filepath.write_text(content, encoding="utf-8")

    return f"Saved text file: {filepath}"

@tool
def save_as_pdf(filename: str, content: str) -> str:
    """
    Save content to a PDF file.
    """

    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    filepath = output_dir / f"{filename}.pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 10, content)

    pdf.output(str(filepath))

    return f"Saved PDF file: {filepath}"

# toolcall

@wrap_tool_call
def human_approval(request, handler):

    tool_name = request.tool_call["name"]

    if tool_name in [
        "save_as_txt",
        "save_as_pdf"
    ]:
        confirm = input(
            f"Agent wants to create a file using '{tool_name}'. Approve? (y/n): "
        )

        if confirm.lower() not in ["y", "yes"]:
            return ToolMessage(
                content="Tool call denied by user.",
                tool_call_id=request.tool_call["id"]
            )

    return handler(request)

print("MODEL:", llm.model_name)

# create_agent
agent = create_agent(
    model=llm,
    tools=[
        extract_pdf,
        save_as_txt,
        save_as_pdf],
    middleware=[human_approval],
    system_prompt="""
You are a helpful student assistant.

Only use the extract_pdf tool when the user explicitly asks about a PDF, notes, study material, or document contents.

For greetings like 'hi', 'hello', or general questions, answer normally without using tools.
"""
)

print("how can i help you?")

messages = []

while True:
    user_input = input("you: ")

    if user_input == "":
        print("type anything")
        break
    messages.append(
        HumanMessage(content=user_input)
    )
    if user_input.lower() == "exit":
        break

    print("Calling agent...")

    try:
        result = agent.invoke({
            "messages": messages
        })


        
        messages = result["messages"]

        print("Bot:", messages[-1].content) 

    except Exception as e:
        print("ERROR:", repr(e))

































































# tools_map = {"extract_pdf": extract_pdf}
# llm_with_tool = llm.bind_tools([extract_pdf])
# messages = []

# while True:
#     prompt = input("You: ")
#     if prompt.lower() == "exit":
#         break

#     messages.append(HumanMessage(content=prompt))

#     result = llm_with_tool.invoke(messages)
#     messages.append(result)

#     if result.tool_calls:
#         # LLM wants to use a tool — ask for approval first
#         for tool_call in result.tool_calls:
#             print(f"\nTool: {tool_call['name']} → {tool_call['args']}")
#             approval = input("Approve? (y/n): ")

#             if approval.lower() == "y":
#                 tool_fn = tools_map.get(tool_call["name"])
#                 tool_result = tool_fn.invoke(tool_call["args"]) if tool_fn else "Unknown tool."
#             else:
#                 tool_result = "User denied tool execution."

#             messages.append(ToolMessage(
#                 content=str(tool_result),
#                 tool_call_id=tool_call["id"]
#             ))

#         # LLM sees tool result and gives final answer
#         final = llm_with_tool.invoke(messages)
#         messages.append(final)
#         print("\nAI:", final.content)

#     else:
#         # LLM answered directly, no tool needed
#         print("\nAI:", result.content)