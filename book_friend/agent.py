import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Load environment
load_dotenv()

# --- Pydantic Models ---
class Book(BaseModel):
    title: str
    pages: dict[int, str] or dict
    book_desc: str

# --- Book Functions (Tools) ---
def get_book_instance(title: str, book_desc: str) -> Book:
    """Get a new Book instance"""
    return Book(title=title, book_desc=book_desc, pages={})

def get_book() -> dict:
    """Retrieve the current book from file."""
    if not os.path.exists("book_data.json"):
        return {"error": "No book found."}
    with open("book_data.json", "r") as f:
        return json.load(f)

def update_page(book: Book, page_number: int, page_content: str) -> Book:
    """Update a specific page in the book."""
    book.pages[page_number] = page_content
    save_book(book)
    return book

def save_book(book: Book) -> str:
    """Save the updated book to file."""
    with open("book_data.json", "w") as f:
        json.dump(book.model_dump(), f, indent=2)
    return "Book saved successfully."

# --- Function Declarations for Tools ---
function_declarations = [
    types.FunctionDeclaration(
        name="get_book_instance",
        description="Create a new book instance.",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "book_desc": {"type": "string"}
            },
            "required": ["title", "book_desc"]
        }
    ),
    types.FunctionDeclaration(
        name="get_book",
        description="Fetch the existing saved book.",
        parameters={"type": "object", "properties": {}}
    ),
    types.FunctionDeclaration(
        name="update_page",
        description="Update a specific page in the book.",
        parameters={
            "type": "object",
            "properties": {
                "book": {"type": "object"},
                "page_number": {"type": "integer"},
                "page_content": {"type": "string"}
            },
            "required": ["book", "page_number", "page_content"]
        }
    ),
    types.FunctionDeclaration(
        name="save_book",
        description="Save the book to file storage.",
        parameters={
            "type": "object",
            "properties": {
                "book": {"type": "object"}
            },
            "required": ["book"]
        }
    )
]

# --- Initialize Gemini Client ---




def run_agent():

    while True:

        user_input = str(input("👦🏻:")).strip()
        if user_input == "exit":
            break
        
        print("")
        user_prompt = user_input
        history.append({"role": "user", "parts": [{"text": user_prompt}]})
    # --- Generate Content ---
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=tools
            ),
        )

        # --- Handle Tool Calls ---
        if hasattr(response, "candidates") and response.candidates:
            tool_calls = response.candidates[0].content.parts
            for part in tool_calls:
                if part.function_call:
                    fn_name = part.function_call.name
                    args = part.function_call.args
                    print("🌐: ", fn_name, " 🖥️: ", args)
                    print(f"🔧 Gemini requested tool: {fn_name} with args: {args}")

                    if fn_name == "get_book_instance":
                        result = get_book_instance(**args)
                    elif fn_name == "get_book":
                        result = get_book()
                    elif fn_name == "update_page":
                        b = Book(**args["book"])
                        result = update_page(b, args["page_number"], args["page_content"])
                    elif fn_name == "save_book":
                        b = Book(**args["book"])
                        result = save_book(b)

                    print("🧩 Tool Result:", result)
                    history.append({
                            "role": "model",
                            "parts": [part]  # model's tool call
                        })
                    history.append({
                        "role": "user",
                        "parts": [{"function_response": {
                            "name": fn_name,
                            "response": result.model_dump() if isinstance(result, BaseModel) else result
                        }}]
                    })
        print("🤖", response.text)
        print("")


if __name__ == "__main__":
    client = genai.Client()
    tools = [types.Tool(function_declarations=function_declarations)]
    history = []
    system_instruction = (
        "You are an expert AI writing assistant that helps users craft, edit, "
        "and improve books. You can call tools like get_book, update_page, save_book "
        "to manage book content. Always follow the Book model structure."
    )
    run_agent()
