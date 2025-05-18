from __future__ import annotations

import asyncio
from typing import Any, List, Sequence, Type, Union

from langchain.pydantic_v1 import BaseModel, Field, Extra
from langchain.prompts import ChatPromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.vectorstore import VectorStore
from langchain.tools import BaseTool

# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class MitreSearchInput(BaseModel):
    """Arguments accepted by :class:`MitreSearchTool`."""

    query: Union[str, dict] = Field(
        ...,
        description=(
            "A natural‑language description of behaviour, artefacts, commands, "
            "etc., to match against MITRE ATT&CK Enterprise techniques."
        ),
    )
    k: int = Field(5, description="How many nearest neighbours to retrieve (default 5).")

    class Config:
        extra = Extra.forbid

class MitreReferencesInput(BaseModel):
    """Arguments for :class:`MitreReferencesTool`."""

    references: Sequence[str] = Field(
        ...,
        description="External reference URLs to summarise (max ~10).",
    )
    request_context: str | None = Field(
        None,
        description=(
            "Optional free‑text describing the detection goal or user query. "
            "Summaries will highlight how each reference helps with that goal."
        ),
    )

    class Config:
        extra = Extra.forbid



# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------

class MitreSearchTool(BaseTool):
    """Semantic search over a MITRE ATT&CK Enterprise ``VectorStore``.

    The tool retrieves the *k* most similar technique documents and lets the
    provided LLM decide which one is the single best match and how to format
    the answer.
    """

    name: str = "mitre_technique_search"
    description: str = (
        "Searches MITRE ATT&CK Enterprise techniques using semantic similarity "
        "and returns the single best‑matching technique ID, name, short "
        "description, and top references."
    )
    args_schema: Type[BaseModel] = MitreSearchInput

    # Injected at construction time
    llm: BaseLanguageModel
    mitredb: VectorStore
    k: int = 5
    verbose: bool = False

    class Config:
        extra = Extra.forbid

    # ----------------------- sync/async bridge ---------------------------
    def _run(self, query: Union[str, dict], k: int | None = None):  # noqa: D401
        """Synchronous wrapper for blocking environments."""
        return asyncio.run(self._arun(query, k or self.k))

    async def _arun(self, query: Union[str, dict], k: int | None = None):  # noqa: D401
        """Async execution (preferred by LangChain)."""
        k = k or self.k

        # Build retrieval‑augmented prompt chain
        template = """You are a cybersecurity **detection‑engineering** assistant who
specialises in writing *Sigma rules*.

The *Vectorstore Search Results* below contain MITRE ATT&CK Enterprise
technique fragments (title, description, detection guidance, references).

Your job:
1. Pick the 3 **most relevant** techniques to the *User Question*.
2. From that technique, derive succinct **Sigma detection ideas** – concrete
   field/value pairs, event IDs, log sources, command keywords, etc. that could
   populate a Sigma `detection:` clause. Separate ideas with semicolons.
3. Respond in **exactly** this format for each technique(no extra lines):
   <technique_id> – <name>
   <one‑sentence summary>
   Sigma detection ideas: <semicolon‑separated list>
   Top references: <comma‑separated list (max 3)>
4. If nothing is relevant, answer exactly: No relevant technique found.
5. If less than 3 are relevant return only those that are, be strict with your choices
------
Vectorstore Search Results:
{context}
------
User Question:
{question}
"""
        prompt = ChatPromptTemplate.from_template(template)
        retriever = self.mitredb.as_retriever(search_kwargs={"k": k})
        chain = (
            {
                "context": retriever,
                "question": RunnablePassthrough(),
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return await chain.ainvoke(query)

# ---------------------------------------------------------------------------
# MitreReferencesTool – Summarize external references
# ---------------------------------------------------------------------------

class MitreReferencesTool(BaseTool):
    """Summarise a list of ATT&CK external references.

    For each supplied URL the tool fetches the page, extracts visible text,
    and asks the LLM for a **rule‑aware** two‑sentence summary.
    """

    name: str = "mitre_external_references"
    description: str = (
        "Given ATT&CK external reference URLs, fetch each page and return a "
        "short summary focused on how it can help with the current detection "
        "goal (if provided)."
    )
    args_schema: Type[BaseModel] = MitreReferencesInput

    llm: BaseLanguageModel  # injected at construction

    class Config:
        extra = Extra.forbid

    # ------------- internal helpers -------------
    async def _fetch(self, session, url: str) -> str:
        """Download up to 50 kB of the URL body (UTF‑8)."""
        import aiohttp

        try:
            async with session.get(url, timeout=8) as resp:
                if resp.status != 200:
                    return ""
                return await resp.text(encoding="utf-8", errors="ignore")[:50_000]
        except Exception:  # noqa: BLE001
            return ""

    async def _summarise(self, url: str, html: str, ctx: str | None) -> str:
        """Summarise *html* with optional rule context *ctx*."""
        from bs4 import BeautifulSoup

        text = BeautifulSoup(html, "html.parser").get_text(" ", strip=True)[:2000]
        if not text:
            return "(no readable text)"

        
        if ctx:
            prompt = f"""You are assisting with Sigma rule creation. The user's current detection goal is:
                    {ctx}
                    ---
                    {text}
                    ---
                    In 2 sentences, explain how the information at {url} can help with that goal."""
        else:
            prompt = f"""Summarise in 2 sentences the information useful for detection engineering found at {url}:
                    ---
                    {text}
                    ---
                    Summary:"""
        return (await self.llm.apredict(prompt)).strip()

    # ------------- sync/async entrypoints -------------
    def _run(self, references: Sequence[str], request_context: str | None = None):  # type: ignore[override]
        return asyncio.run(self._arun(references, request_context))

    async def _arun(
        self, references: Sequence[str], request_context: str | None = None
    ) -> str:  # type: ignore[override]
        import aiohttp

        if not references:
            return "No reference URLs provided."

        async with aiohttp.ClientSession() as session:
            bodies = await asyncio.gather(*[self._fetch(session, url) for url in references])

        lines: List[str] = ["External references:"]
        for idx, (url, body) in enumerate(zip(references, bodies, strict=False), start=1):
            if not body:
                lines.append(f"{idx}. (unreachable) {url}")
                continue
            summary = await self._summarise(url, body, request_context)
            lines.append(f"{idx}. {summary} ({url})")
        return "".join(lines)