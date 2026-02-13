import os
from typing import Any, Dict, Iterator, List

from openai import OpenAI
import pinecone

from spoon_ai.tools.base import BaseTool, ToolFailure, ToolResult


class ToolManager:
    def __init__(self, tools: List[BaseTool]):
        self.tools = tools
        self.tool_map = {tool.name: tool for tool in tools}
        self.indexed = False
        # Schema cache: avoids re-serializing every step
        self._params_cache: List[Dict[str, Any]] | None = None
        self._params_cache_count: int = -1

    def _invalidate_params_cache(self) -> None:
        """Invalidate the cached tool schemas."""
        self._params_cache = None
        self._params_cache_count = -1

    def reindex(self) -> None:
        """Rebuild the internal name->tool mapping. Useful if tools have been renamed dynamically."""
        self.tool_map = {tool.name: tool for tool in self.tools}
        self._invalidate_params_cache()


    def _lazy_init_pinecone(self):
        if not hasattr(self, "pc"):
            pinecone.init(api_key=os.getenv("PINECONE_API_KEY"))
            index_name = "dex-tools"

            if index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=index_name,
                    dimension=3072,
                    metric="cosine"
                )

            self.index = pinecone.Index(index_name)
            self.embedding_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def __getitem__(self, name: str) -> BaseTool:
        return self.tool_map[name]

    def __iter__(self) -> Iterator[BaseTool]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    def to_params(self) -> List[Dict[str, Any]]:
        if self._params_cache is not None and self._params_cache_count == len(self.tools):
            return self._params_cache
        self._params_cache = [tool.to_param() for tool in self.tools]
        self._params_cache_count = len(self.tools)
        return self._params_cache

    async def execute(self, * ,name: str, tool_input: Dict[str, Any] =None) -> ToolResult:
        tool = self.tool_map[name]
        if not tool:
            return ToolFailure(f"Tool '{name}' not found. Available tools: {list(self.tool_map.keys())}")

        # Ensure tool_input is not None
        if tool_input is None:
            tool_input = {}

        try:
            result = await tool(**tool_input)
            return result
        except Exception as e:
            # Provide better error context
            return ToolFailure(f"Tool '{name}' execution failed: {str(e)}")

    def get_tool(self, name: str) -> BaseTool:
        tool = self.tool_map.get(name)
        if not tool:
            raise ValueError(f"Tool {name} not found")
        return tool

    def add_tool(self, tool: BaseTool) -> None:
        self.tools.append(tool)
        self.tool_map[tool.name] = tool
        self._invalidate_params_cache()

    def add_tools(self, *tools: BaseTool) -> None:
        for tool in tools:
            self.tools.append(tool)
            self.tool_map[tool.name] = tool
        self._invalidate_params_cache()

    def remove_tool(self, name: str) -> None:
        if name not in self.tool_map:
            return
        self.tools = [tool for tool in self.tools if tool.name != name]
        del self.tool_map[name]
        self._invalidate_params_cache()

    def index_tools(self):
        self._lazy_init_pinecone()
        vectors = []
        for tool in self.tools:
            embedding_vector = self.embedding_client.embeddings.create(
                input=tool.description,
                model="text-embedding-3-large"
            )
            vectors.append(
                {
                    "id": tool.name,
                    "values": embedding_vector.data[0].embedding,
                    "metadata": {
                        "name": tool.name,
                        "description": tool.description
                    }
                }
            )
        self.index.upsert(vectors=vectors, namespace="dex-tools-test")
        self.indexed = True

    def query_tools(self, query: str, top_k: int = 5, rerank_k: int = 20) -> List[str]:
        if not self.indexed:
            self.index_tools()
        embedding_vector = self.embedding_client.embeddings.create(
            input=query,
            model="text-embedding-3-large"
        )
        results = self.index.query(
            namespace="dex-tools-test",
            top_k=rerank_k,
            include_metadata=True,
            include_values=False,
            vector=embedding_vector.data[0].embedding
        )
        print(results)
        doc_to_tool = {}
        for match in results["matches"]:
            doc_to_tool[match["metadata"]["description"]] = match["id"]

        result_tool_names = []
        for match in results["matches"][:top_k]:
            result_tool_names.append(match["id"])
        return result_tool_names
