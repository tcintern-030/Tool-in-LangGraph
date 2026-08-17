import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import ToolMessage
from tools import tools

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

llm_with_tools = llm.bind_tools(tools)

tool_dictionary = {
    tool.name: tool
    for tool in tools
}


def run_agent(user_question):

    print("\nUser:")
    print(user_question)

    response = llm_with_tools.invoke(user_question)

    if not response.tool_calls:

        print("\nAgent:")
        print(response.content)

        return response.content

    tool_messages = []

    for tool_call in response.tool_calls:

        tool_name = tool_call["name"]
        tool_arguments = tool_call["args"]

        print("\nTool Selected:")
        print(tool_name)

        print("Arguments:")
        print(tool_arguments)

        selected_tool = tool_dictionary[tool_name]

        try:

            result = selected_tool.invoke(tool_arguments)

        except Exception as e:

            result = f"Tool failed: {str(e)}"

        print("\nTool Result:")
        print(result)

        tool_message = ToolMessage(
            content=result,
            tool_call_id=tool_call["id"]
        )

        tool_messages.append(tool_message)

    messages = [
        response,
        *tool_messages
    ]

    final_response = llm_with_tools.invoke(messages)

    print("\nFinal Answer:")
    print(final_response.content)

    return final_response.content

if __name__ == "__main__":

    while True:
        question = input("\nAsk something (type exit to stop): ")

        if question.lower() == "exit":
            break

        run_agent(question)