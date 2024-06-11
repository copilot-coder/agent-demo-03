import json
import os
import requests
import urllib.parse
from typing import List, Iterable, Union

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall
from openai.types.chat.chat_completion_user_message_param import ChatCompletionUserMessageParam
from openai.types.chat.chat_completion_tool_message_param import ChatCompletionToolMessageParam
from openai.types.chat.chat_completion_assistant_message_param import ChatCompletionAssistantMessageParam


api_key = os.getenv('OPENAI_API_KEY')
base_url = os.getenv("OPENAI_API_BASE")
model = "gpt-3.5-turbo"
stream = True
# 如果你没有openai的api key，可以使用阿里云的通义千问(https://dashscope.console.aliyun.com)
# 并去掉下面三行的注释
# base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
# model = "qwen-plus"
# stream = False # 通义千Function Calling不支持流式输出

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

tools = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "perform mathematical operation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "the mathematical expression"
                    }
                },
                "required": [
                    "expression"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    }
                },
                "required": [
                    "location"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "retrieve the up to date information",
            "parameters": {
                "type": "object",
                "properties": {
                    "keyword": {
                        "type": "string",
                        "description": "the keyword to search"
                    }
                },
                "required": [
                    "keyword"
                ]
            }
        }
    }
]


def calculator(expression: str) -> str:
    data = {'result': eval(expression)}
    return str(data)


def get_current_weather(location: str) -> str:
    url = "https://weather.cma.cn/api/autocomplete?q=" + \
        urllib.parse.quote(location)
    response = requests.get(url)
    data = response.json()
    if data["code"] != 0:
        return "没找到该位置的信息"
    location_code = ""
    for item in data["data"]:
        str_array = item.split("|")
        if str_array[1] == location or str_array[2] == location:
            location_code = str_array[0]
            break
    if location_code == "":
        return "没找到该位置的信息"
    url = f'https://weather.cma.cn/api/now/{location_code}'
    return requests.get(url).text


def search_web(keyword: str) -> str:
    data = {
        'api_key': os.getenv('TAVILY_API_KEY'),
        'query': keyword,
    }
    return requests.post('https://api.tavily.com/search', json=data).text


def invoke_tool(tool_call: Union[ChatCompletionMessageToolCall, ChoiceDeltaToolCall]) -> ChatCompletionToolMessageParam:
    result = ChatCompletionToolMessageParam(
        role="tool", tool_call_id=tool_call.id)
    func_name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)

    if func_name == "calculator":
        result["content"] = calculator(args['expression'])
    elif func_name == "get_current_weather":
        result["content"] = get_current_weather(args['location'])
    elif func_name == "search_web":
        result["content"] = search_web(args['keyword'])
    else:
        result["content"] = "函数未定义"
    return result


def merge_too_calls(tool_calls: List[ChoiceDeltaToolCall], delta_tool_calls: List[ChoiceDeltaToolCall]):
    for index, delta_tool_call in enumerate(delta_tool_calls):
        if len(tool_calls) <= index:
            if delta_tool_call.function.arguments is None:
                delta_tool_call.function.arguments = ''
            tool_calls.append(delta_tool_call)
            continue
        ref = tool_calls[index]
        ref.function.arguments += delta_tool_call.function.arguments


def main():
    MAX_MESSAGES_NUM = 20
    messages: Iterable[ChatCompletionMessageParam] = list()
    needInput = True
    while True:
        # 只保留20条消息作为上下文
        if len(messages) > MAX_MESSAGES_NUM:
            messages = messages[-MAX_MESSAGES_NUM:]
            while len(messages) > 0:
                role = messages[0]['role']
                if role == 'system' or role == 'user':
                    break
                messages = messages[1:]

        # 等待用户输入
        if needInput:
            query = input("\n>>>> 请输入问题:").strip()
            if query == "":
                continue
            messages.append(ChatCompletionUserMessageParam(
                role="user", content=query))
        # 向LLM发起查询（除了用户的query外，还需要带上tools定义）
        chat_completion = client.chat.completions.create(
            messages=messages,
            tools=tools,
            model=model,
            stream=stream
        )

        tool_calls = None
        content = None
        if stream:
            # 处理流式输出
            for chunk in chat_completion:
                if len(chunk.choices) == 0:
                    continue
                delta = chunk.choices[0].delta
                if isinstance(delta.tool_calls, list):
                    if tool_calls is None:
                        tool_calls = []
                    merge_too_calls(tool_calls, delta.tool_calls)
                elif isinstance(delta.content, str):
                    if content is None:
                        content = ""
                    content += delta.content
                    print(delta.content, end='', flush=True)
        else:
            # 非流式输出
            tool_calls = chat_completion.choices[0].message.tool_calls
            content = chat_completion.choices[0].message.content
        if isinstance(tool_calls, list) and len(tool_calls) > 0:  # LLM的响应信息有tool_calls信息
            needInput = False
            messages.append(ChatCompletionAssistantMessageParam(
                role="assistant", tool_calls=tool_calls, content=''))
            # 注意：LLM的响应可能包括多个tool_call
            for tool_call in tool_calls:
                result = invoke_tool(tool_call)
                messages.append(result)
        else:
            needInput = True
            if isinstance(content, str) and len(content) > 0:
                if not stream:
                    print(content)
                messages.append(ChatCompletionAssistantMessageParam(
                    role="assistant", content=content))


main()
