import math
import os
from datetime import datetime

from langchain_core.tools import tool
from serpapi import GoogleSearch

# First, create the output formatting function
def format_weather_data(text_blocks):
    """Format the weather data from text_blocks into a readable string"""
    formatted_text = ""
    
    for block in text_blocks:
        if block["type"] == "paragraph":
            formatted_text += f"{block['snippet']}\n\n"
        elif block["type"] == "heading":
            formatted_text += f"--- {block['snippet']} ---\n"
        elif block["type"] == "list" and "list" in block:
            for item in block["list"]:
                formatted_text += f"• {item['snippet']}\n"
            formatted_text += "\n"
    
    return formatted_text.strip()

# Define the tool using the langchain decorator
@tool
def get_weather(location: str) -> str:
    """Get current weather information for a specific location.
    Example: "portland oregon", "new york", "london uk"
    """
    try:
        params = {
            "engine": "google_ai_mode",
            "q": f"{location} weather",
            "api_key": os.environ["SERPAPI_KEY"],
            "location": "Portland, OR"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "text_blocks" in results and len(results["text_blocks"]) > 0:
            # Extract and format weather information from the text blocks
            weather_info = format_weather_data(results["text_blocks"])
            return f"Weather in {location}:\n{weather_info}"
        else:
            return f"Could not find weather information for {location}"
            
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

@tool
def web_search(query: str) -> str:
    """Search the web using Google. Pass in the query you want to search. You can use anything that you would use in a regular Google search. e.g. inurl:, site:, intitle:. 
    """
    try:
        params = {
            "engine": "google_ai_mode",
            "q": f"{query}",
            "api_key": os.environ["SERPAPI_KEY"],
            "location": "Portland, OR"
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "text_blocks" in results and len(results["text_blocks"]) > 0:
            # Extract and format weather information from the text blocks
            output = results["text_blocks"]
            return f"{query} results:\n{output}"
        else:
            return f"Could not find search Google for {query}"
            
    except Exception as e:
        return f"Error fetching weather data: {str(e)}"

@tool
def clock():
    """Get the datetime. Returns datetime.now().strftime("%Y-%m-%d %I:%M %p")"""    
    return datetime.now().strftime("%Y-%m-%d %I:%M %p")

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression. You can use basic operators (+, -, *, /, ^) and functions like sqrt, sin, cos, etc.
    Example expressions: 
    - "2 + 3 * 4" 
    - "sqrt(16)" 
    - "sin(30) + cos(60)"
    """
    try:
        # Replace ^ with ** for exponentiation
        expression = expression.replace('^', '**')
        
        # Add math functions to the evaluation context
        safe_dict = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sqrt': math.sqrt, 'sin': math.sin, 'cos': math.cos, 
            'tan': math.tan, 'log': math.log, 'log10': math.log10,
            'pi': math.pi, 'e': math.e
        }
        
        # Evaluate the expression safely
        result = eval(expression, {"__builtins__": None}, safe_dict)
        return f"The result of {expression} is {result}"
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"