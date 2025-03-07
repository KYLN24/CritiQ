import json
import os
import random
import time
from time import sleep
from typing import Any

from openai import OpenAI, RateLimitError

RATE_LIMIT_RETRY_DELAY = 60
RATE_LIMIT_RETRY_ATTEMPTS = 10
WORKFLOW_AGENT_LOGFILE = os.getenv("WORKFLOW_AGENT_LOGFILE", None)

class Agent:
    def __init__(
        self,
        system: str | None = None,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_keys: str | list[str] | None = None,
        request_kwargs: dict[str, Any] = None,
        tensor_parallel_size: int | None = None,
    ):
        self.system = system
        if self.system is None:
            self.history = []
        else:
            self.history = [{"role": "system", "content": self.system}]
        self.model = model
        self.base_url = base_url

        if api_keys is not None:
            if isinstance(api_keys, str):
                api_keys = [api_keys]
        else:
            api_keys = [os.getenv("OPENAI_API_KEY", "EMPTY")]
        self.api_keys = api_keys

        self.request_kwargs = {}
        if request_kwargs is not None:
            self.request_kwargs.update(request_kwargs)

        self.client = OpenAI(
            api_key=random.choice(self.api_keys), base_url=self.base_url
        )
    def chat_completion_openai(
        self, messages, stream: bool = True, ttl: int = RATE_LIMIT_RETRY_ATTEMPTS
    ):
        response = ""
        if ttl >= 0:
            if stream:
                chunk_stream = self.client.chat.completions.create(
                    stream=True,
                    model=self.model,
                    messages=messages,
                    **self.request_kwargs,
                )
                for chunk in chunk_stream:  # pylint: disable=E1133:not-an-iterable
                    if (
                        not chunk.choices[0].finish_reason
                        and chunk.choices[0].delta.content
                    ):
                        print(chunk.choices[0].delta.content, end="", flush=True)
                        response += chunk.choices[0].delta.content
                print()
            else:
                try:
                    response = (
                        self.client.chat.completions.create(
                            model=self.model, messages=messages, **self.request_kwargs
                        )
                        .choices[0]
                        .message.content
                    )
                except RateLimitError as e:
                    print(
                        f"Rate limit exceeded, waiting for {RATE_LIMIT_RETRY_DELAY} seconds and retrying... {ttl=}",
                        e,
                    )
                    sleep(RATE_LIMIT_RETRY_DELAY)
                    return self.chat_completion_openai(
                        messages, stream=False, ttl=ttl - 1
                    )
        return response

    def chat_completion(
        self, messages, stream: bool = True, ttl: int = RATE_LIMIT_RETRY_ATTEMPTS
    ):
        return self.chat_completion_openai(messages, stream=stream, ttl=ttl)

    def __call__(self, prompt, stream: bool = True) -> str | None:
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.chat_completion(self.history, stream=stream)
            assert response is not None
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            self.history.pop()
            print(e)
            return None
        self.history.append({"role": "assistant", "content": response})
        if WORKFLOW_AGENT_LOGFILE:
            with open(WORKFLOW_AGENT_LOGFILE, "a") as f:
                f.write(
                    json.dumps(
                        {
                            "time": time.time(),
                            "model": self.model,
                            "prompt": prompt,
                            "response": response,
                            "history": self.history,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        return response

    def get_last_reply(self):
        if self.history[-1]["role"] == "assistant":
            return self.history[-1]["content"]
        return None

    def forget_last_turn(self):
        while self.history[-1]["role"] != "user":
            self.history.pop()
        if self.history[-1]["role"] == "user":
            self.history.pop()

    def fork(self) -> "Agent":
        forked = Agent(
            system=None,
            model=self.model,
            base_url=self.base_url,
            api_keys=self.api_keys,
            request_kwargs=self.request_kwargs,
        )
        for turn in self.history:
            forked.history.append(turn)
        return forked
