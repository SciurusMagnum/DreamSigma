
from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Type, List
from urllib.request import urlopen

# sigmaiq
from sigmaiq.utils.sigma.rule_updater import SigmaRuleUpdater
from sigmaiq.globals import DEFAULT_DIRS

# langchain / langchain‑community
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import CharacterTextSplitter

# langchain typing
from langchain.schema.embeddings import Embeddings
from langchain.schema.vectorstore import VectorStore
from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from langchain.schema.document import BaseDocumentTransformer


class SigmaLLM(SigmaRuleUpdater):
    """Base class for Sigma rules with LLMs.

    Provides helpers for keeping the local Sigma rule set up‑to‑date,
    creating a vector store from those rules, and doing basic similarity
    search over it.
    """

    # ---------------- Initialisation ----------------

    def __init__(
        self,
        rule_dir: str | None = None,
        vector_store_dir: str | None = None,
        *,
        embedding_model: OpenAIEmbeddings | None = None,
        embedding_function: Type[Embeddings] = OpenAIEmbeddings,
        vector_store: Type[VectorStore] = FAISS,
        rule_loader: Type[BaseLoader] = DirectoryLoader,
        rule_splitter: Type[BaseDocumentTransformer] = CharacterTextSplitter,
    ) -> None:
        super().__init__(rule_dir=rule_dir)  # may download/update Sigma rules

        self.vector_store_dir = self._setup_vector_store_dir(vector_store_dir)

        self.embedding_function = embedding_model or embedding_function()

        self.vector_store: Type[VectorStore] = vector_store
        self.sigmadb: VectorStore | None = None

        self.rule_loader = rule_loader
        self.rule_splitter = rule_splitter

        self.mitre_db: VectorStore | None = None
        self._mitre_store_path = Path(self.vector_store_dir) / "mitre_vectordb"
        self._mitre_store_path.mkdir(parents=True, exist_ok=True)

    # ---------------- Vector‑store plumbing ----------------

    def _setup_vector_store_dir(self, vector_store_dir: str | None) -> str:
        """Ensures directory exists and returns its absolute path."""
        root = vector_store_dir or DEFAULT_DIRS.VECTOR_STORE_DIR
        os.makedirs(root, exist_ok=True)
        return root

    # ----- Sigma rule vector‑store -----

    def load_sigma_vectordb(self) -> None:
        """Loads an on‑disk Sigma FAISS index into ``self.sigmadb``."""
        if self.sigmadb:
            return
        
        if not os.path.exists(self.vector_store_dir):
            raise FileNotFoundError(
                f"VectorStore not found at {self.vector_store_dir}. Run ``create_sigma_vectordb()`` first."
            )
        self.sigmadb = self.vector_store.load_local(
            folder_path=self.vector_store_dir,
            embeddings=self.embedding_function,
            allow_dangerous_deserialization=True,
        )

    def create_sigma_vectordb(self, *, save: bool = True) -> None:
        """(Re)builds the Sigma vector‑store from the local rule set."""
        if not self.installed_tag:
            self.update_sigma_rules()

        sigma_docs = self.create_sigma_rule_docs()
        sigma_docs = self.split_sigma_docs(sigma_docs)
        self.create_vectordb(sigma_docs)
        if save:
            self.save_vectordb()

    # --- helper sub‑steps --------------------------------------------------

    def create_sigma_rule_docs(self) -> List[Document]:
        """Loads every ``*.yml`` rule from the Sigma repository into Docs."""
        loader = self.rule_loader(self.rule_dir, glob="**/*.yml", loader_cls=TextLoader)
        return loader.load()

    def split_sigma_docs(self, sigma_docs: List[Document]) -> List[Document]:
        """Splits gigantic rules; by default we keep each rule whole."""
        splitter = self.rule_splitter(chunk_size=99999)
        return splitter.split_documents(sigma_docs)

    def create_vectordb(self, sigma_docs: List[Document]) -> None:
        """Embeds documents and stores them in ``self.sigmadb``."""
        self.sigmadb = self.vector_store.from_documents(sigma_docs, self.embedding_function)

    def save_vectordb(self) -> None:
        """Persists ``self.sigmadb`` to :pyattr:`vector_store_dir`."""
        if self.sigmadb is None:
            raise RuntimeError("Vector DB has not been created yet.")
        self.sigmadb.save_local(self.vector_store_dir)

    # Public Sigma search helper
    def simple_search(self, query: str, k: int = 3) -> List[Document]:
        """Returns *k* most similar Sigma rules for *query*."""
        if not self.sigmadb:
            self.load_sigma_vectordb()
        return self.sigmadb.similarity_search(query, k)

    # MITRE ATT&CK EXTENSION 
    _ATTACK_ENTERPRISE_JSON = (
        "https://raw.githubusercontent.com/mitre/cti/master/enterprise-attack/enterprise-attack.json"
    )


    def _download_attack_bundle(self) -> Path:
        """Fetches the Enterprise ATT&CK STIX bundle JSON to the cache dir."""
        bundle_path = self._mitre_store_path / "enterprise-attack.json"
        if not bundle_path.exists():
            with urlopen(self._ATTACK_ENTERPRISE_JSON) as resp, open(bundle_path, "wb") as fh:
                fh.write(resp.read())
        return bundle_path

    def _attack_docs(self) -> List[Document]:
        """Parses STIX and returns a list of LangChain documents."""
        data = json.loads(self._download_attack_bundle().read_text("utf-8"))
        docs: List[Document] = []
        for obj in data.get("objects", []):
            if obj.get("type") != "attack-pattern" or obj.get("revoked") is True:
                continue
            tech_id = next(
                (
                    ref.get("external_id")
                    for ref in obj.get("external_references", [])
                    if ref.get("source_name") == "mitre-attack"
                ),
                None,
            )
            if not tech_id:
                continue
            name = obj.get("name", "")
            description = obj.get("description", "").replace("\n", " ")
            detection = obj.get("x_mitre_detection", "").replace("\n", " ")
            docs.append(
                Document(
                    page_content=f"{tech_id}: {name}\n{description}\nknown detection method: {detection}",
                    metadata={"technique_id": tech_id, "name": name},
                )
            )
        return docs


    def build_mitre_vectordb(self, *, save: bool = True) -> None:
        """Creates a vector‑store of ATT&CK techniques using current embeddings."""
        docs = self._attack_docs()
        self.mitre_db = self.vector_store.from_documents(docs, self.embedding_function)
        if save:
            self.mitre_db.save_local(str(self._mitre_store_path))

    def load_mitre_vectordb(self) -> None:
        """Loads ``mitre_db`` lazily, falling back to on‑the‑fly build."""
        if self.mitre_db:
            return
        index_file = self._mitre_store_path / "index.faiss"
        if index_file.exists():
            self.mitre_db = self.vector_store.load_local(
                folder_path=str(self._mitre_store_path),
                embeddings=self.embedding_function,
                allow_dangerous_deserialization=True,
            )
        else:
            self.build_mitre_vectordb(save=True)
