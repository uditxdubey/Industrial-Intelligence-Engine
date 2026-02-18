from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.selectors import LLMMultiSelector
from llama_index.core.query_engine import RouterQueryEngine

class AgenticRouter:
    def __init__(self, llm):
        self.llm = llm

    def create_router_engine(self, siemens_index, rockwell_index):
        # Tool 1: Siemens
        siemens_tool = QueryEngineTool(
            query_engine=siemens_index.as_query_engine(similarity_top_k=5),
            metadata=ToolMetadata(
                name="siemens_manual_tool",
                description="Useful for questions about Siemens S7-1200 hardware/wiring."
            ),
        )

        # Tool 2: Rockwell
        rockwell_tool = QueryEngineTool(
            query_engine=rockwell_index.as_query_engine(similarity_top_k=5),
            metadata=ToolMetadata(
                name="rockwell_manual_tool",
                description="Useful for questions about Rockwell, Allen-Bradley, and CompactLogix hardware."
            ),
        )

        # AI model that decides which tool to use based on the question
        selector = LLMMultiSelector.from_defaults(llm=self.llm)
        
        return RouterQueryEngine(
            selector=selector,
            query_engine_tools=[siemens_tool, rockwell_tool],
            verbose=True
        )