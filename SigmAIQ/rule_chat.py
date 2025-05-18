from __future__ import annotations

import asyncio
import sys

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage

# Factory that returns the Sigma agent with MitreSearchTool included
from sigmaiq.llm.toolkits.base import create_sigma_agent
from sigmaiq.llm.base import SigmaLLM
from langchain_openai import OpenAIEmbeddings


class RuleChatSession:
    """Interactive session that preserves conversation context."""

    def __init__(self, temperature: float = 0.1):
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        sigma_llm = SigmaLLM(embedding_model=OpenAIEmbeddings(model="text-embedding-3-large", chunk_size=250))
        try:
            sigma_llm.load_sigma_vectordb()
        except FileNotFoundError:
            sigma_llm.create_sigma_vectordb(save=True)
        sigma_llm.load_mitre_vectordb()
        self.memory = memory
        self.agent = create_sigma_agent(
            verbose=True,
            sigma_vectorstore=sigma_llm.sigmadb,
            mitre_vectorstore=sigma_llm.mitre_db,
            #memory=self.memory,
            agent_executor_kwargs={"memory": memory},
        )

    async def aask(self, user_input: str) -> str:
        """Send the user input to the agent and return its answer."""
       
        result = await self.agent.ainvoke({"input": user_input})
        return result["output"]    # agent already logged turn


async def chat_loop() -> None:
    """Async command‑line REPL."""
    session = RuleChatSession()
    print("💬  Interactive Sigma Rule Chat – type 'exit' to quit.\n")
    loop = asyncio.get_event_loop()
    while True:
        try:
            user_in = await loop.run_in_executor(None, sys.stdin.readline)
        except (KeyboardInterrupt, EOFError):
            print("\nGood‑bye!")
            break
        if not user_in:
            continue
        user_in = user_in.rstrip("\n")
        if user_in.lower() in {"exit", "quit"}:
            print("Good‑bye!")
            break
        answer = await session.aask(user_in)
        print("\n🛠️  Assistant:\n" + answer + "\n")


if __name__ == "__main__":
    asyncio.run(chat_loop())
