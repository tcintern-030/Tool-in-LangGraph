from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        result = eval(expression)
        return str(result)

    except Exception:
        return "Unable to calculate the expression."

@tool
def get_weather(city: str) -> str:
    """Get the weather information for a city."""
    weather_data = {
        "lahore": "Lahore: 32°C, Sunny",
        "islamabad": "Islamabad: 27°C, Partly Cloudy",
        "karachi": "Karachi: 30°C, Humid",
        "peshawar": "Peshawar: 31°C, Sunny"
    }

    city = city.lower()

    if city in weather_data:
        return weather_data[city]

    return f"Weather information for {city} is not available."

tools = [
    calculator,
    get_weather
]