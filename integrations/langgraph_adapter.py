"""LangGraph Integration Adapter

Provides bidirectional integration between ResearchCrew (ADK) and LangGraph:
- Pattern A: Use LangGraph agents/graphs as ADK tools
- Pattern B: Use ADK agents as LangGraph nodes

Usage:
    # Pattern A: LangGraph as ADK tool
    from integrations.langgraph_adapter import langgraph_to_adk_tool

    @langgraph_to_adk_tool("analyzer", description="Analyze data")
    def my_langgraph_agent(input: str) -> str:
        return langgraph_graph.invoke({"input": input})["output"]

    # Pattern B: ADK as LangGraph node
    from integrations.langgraph_adapter import adk_to_langgraph_node

    researcher_node = adk_to_langgraph_node(researcher_agent)
    graph.add_node("research", researcher_node)
"""

import asyncio
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, Union

from integrations.base import (
    AdapterConfig,
    AdapterResult,
    AdapterError,
    ExternalAgentAdapter,
    register_adapter,
)
from utils.logging_config import get_logger
from utils.tracing import trace_span

logger = get_logger(__name__)

T = TypeVar("T")
R = TypeVar("R")


# ============================================================================
# LangGraph to ADK Integration (Pattern A)
# ============================================================================


class LangGraphToADKAdapter(ExternalAgentAdapter[dict[str, Any], dict[str, Any]]):
    """Adapter for using LangGraph agents/graphs as ADK tools.

    Wraps a LangGraph graph or runnable to be used as an ADK tool.
    """

    def __init__(
        self,
        name: str,
        langgraph_runnable: Any,
        config: Optional[AdapterConfig] = None,
        input_key: str = "input",
        output_key: str = "output",
    ):
        """Initialize the adapter.

        Args:
            name: Adapter/tool name.
            langgraph_runnable: LangGraph graph or runnable.
            config: Optional adapter configuration.
            input_key: Key for input in LangGraph state.
            output_key: Key for output in LangGraph state.
        """
        config = config or AdapterConfig(name=name)
        super().__init__(config)

        self._runnable = langgraph_runnable
        self._input_key = input_key
        self._output_key = output_key

    @property
    def source_framework(self) -> str:
        return "ADK"

    @property
    def target_framework(self) -> str:
        return "LangGraph"

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the LangGraph runnable.

        Args:
            input_data: Input state for LangGraph.

        Returns:
            Output state from LangGraph.
        """
        # Check if runnable has ainvoke (async) or invoke (sync)
        if hasattr(self._runnable, "ainvoke"):
            result = await self._runnable.ainvoke(input_data)
        elif hasattr(self._runnable, "invoke"):
            # Run sync invoke in thread pool
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._runnable.invoke(input_data),
            )
        else:
            # Assume it's a callable
            if asyncio.iscoroutinefunction(self._runnable):
                result = await self._runnable(input_data)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._runnable(input_data),
                )

        return result


def langgraph_to_adk_tool(
    name: str,
    description: str,
    input_key: str = "input",
    output_key: str = "output",
    timeout: float = 60.0,
) -> Callable:
    """Decorator to convert a LangGraph function/runnable to an ADK tool.

    The decorated function should accept and return LangGraph-compatible
    state dictionaries.

    Args:
        name: Tool name.
        description: Tool description.
        input_key: Key for input in state.
        output_key: Key for output in state.
        timeout: Timeout in seconds.

    Returns:
        Decorator function.

    Example:
        @langgraph_to_adk_tool("analyzer", description="Analyze text")
        def analyze(state: dict) -> dict:
            # LangGraph-style function
            result = process(state["input"])
            return {"output": result}

        # Now usable as ADK tool
        result = await analyze.adk_invoke("some text")
    """

    def decorator(func: Callable) -> Callable:
        # Create adapter
        config = AdapterConfig(
            name=name,
            timeout=timeout,
        )
        adapter = LangGraphToADKAdapter(
            name=name,
            langgraph_runnable=func,
            config=config,
            input_key=input_key,
            output_key=output_key,
        )

        # Register adapter
        register_adapter(adapter)

        @wraps(func)
        async def wrapper(*args, **kwargs) -> Any:
            """Wrapper that handles both direct calls and ADK-style calls."""
            # If called with state dict, execute normally
            if args and isinstance(args[0], dict):
                return func(*args, **kwargs)

            # Otherwise, wrap input for LangGraph
            input_data = {input_key: args[0] if args else kwargs.get("input", "")}
            result = await adapter.execute(input_data)
            return result.value.get(output_key)

        # Add ADK-specific invoke method
        async def adk_invoke(input_text: str) -> str:
            """ADK-style invoke with simple string input/output."""
            input_data = {input_key: input_text}
            result = await adapter.execute(input_data)
            return result.value.get(output_key, "")

        wrapper.adk_invoke = adk_invoke
        wrapper.__doc__ = description
        wrapper._adapter = adapter

        return wrapper

    return decorator


# ============================================================================
# ADK to LangGraph Integration (Pattern B)
# ============================================================================


class ADKToLangGraphAdapter(ExternalAgentAdapter[dict[str, Any], dict[str, Any]]):
    """Adapter for using ADK agents as LangGraph nodes.

    Wraps an ADK agent to be used as a node in a LangGraph workflow.
    """

    def __init__(
        self,
        name: str,
        adk_agent: Any,
        config: Optional[AdapterConfig] = None,
        state_key: str = "query",
        result_key: str = "result",
    ):
        """Initialize the adapter.

        Args:
            name: Adapter/node name.
            adk_agent: ADK agent instance.
            config: Optional adapter configuration.
            state_key: Key to extract query from LangGraph state.
            result_key: Key to store result in LangGraph state.
        """
        config = config or AdapterConfig(name=name)
        super().__init__(config)

        self._agent = adk_agent
        self._state_key = state_key
        self._result_key = result_key

    @property
    def source_framework(self) -> str:
        return "LangGraph"

    @property
    def target_framework(self) -> str:
        return "ADK"

    async def _execute(self, input_data: dict[str, Any]) -> dict[str, Any]:
        """Execute the ADK agent.

        Args:
            input_data: LangGraph state.

        Returns:
            Updated LangGraph state with result.
        """
        # Extract query from state
        query = input_data.get(self._state_key, "")

        # Invoke ADK agent
        if hasattr(self._agent, "ainvoke"):
            result = await self._agent.ainvoke(query)
        elif hasattr(self._agent, "invoke"):
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: self._agent.invoke(query),
            )
        else:
            # Assume callable
            if asyncio.iscoroutinefunction(self._agent):
                result = await self._agent(query)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    lambda: self._agent(query),
                )

        # Return updated state
        return {
            **input_data,
            self._result_key: result,
        }


def adk_to_langgraph_node(
    adk_agent: Any,
    name: Optional[str] = None,
    state_key: str = "query",
    result_key: str = "result",
    timeout: float = 120.0,
) -> Callable:
    """Convert an ADK agent to a LangGraph node function.

    Args:
        adk_agent: ADK agent instance.
        name: Node name (defaults to agent name).
        state_key: Key to extract query from state.
        result_key: Key to store result in state.
        timeout: Timeout in seconds.

    Returns:
        Function compatible with LangGraph add_node().

    Example:
        from google.adk import Agent
        from langgraph.graph import StateGraph

        researcher = Agent(name="researcher", ...)
        research_node = adk_to_langgraph_node(researcher)

        graph = StateGraph(dict)
        graph.add_node("research", research_node)
    """
    # Get agent name
    agent_name = name or getattr(adk_agent, "name", "adk_agent")

    # Create adapter
    config = AdapterConfig(
        name=agent_name,
        timeout=timeout,
    )
    adapter = ADKToLangGraphAdapter(
        name=agent_name,
        adk_agent=adk_agent,
        config=config,
        state_key=state_key,
        result_key=result_key,
    )

    # Register adapter
    register_adapter(adapter)

    async def node_function(state: dict[str, Any]) -> dict[str, Any]:
        """LangGraph node function wrapping ADK agent."""
        result = await adapter.execute(state)
        return result.value

    # For sync LangGraph, provide sync version
    def sync_node_function(state: dict[str, Any]) -> dict[str, Any]:
        """Sync version for LangGraph workflows."""
        return asyncio.run(node_function(state))

    # Attach both versions
    node_function.sync = sync_node_function
    node_function._adapter = adapter
    node_function.__name__ = f"{agent_name}_node"

    return node_function


# ============================================================================
# LangGraph Adapter Class
# ============================================================================


class LangGraphAdapter:
    """High-level adapter for LangGraph integration.

    Provides utility methods for common integration patterns.
    """

    def __init__(self, name: str = "langgraph"):
        """Initialize the adapter.

        Args:
            name: Adapter name prefix.
        """
        self.name = name
        self._tools: dict[str, Callable] = {}
        self._nodes: dict[str, Callable] = {}

    def register_tool(
        self,
        langgraph_runnable: Any,
        tool_name: str,
        description: str,
        **kwargs,
    ) -> Callable:
        """Register a LangGraph runnable as an ADK tool.

        Args:
            langgraph_runnable: LangGraph graph or runnable.
            tool_name: Name for the tool.
            description: Tool description.
            **kwargs: Additional arguments for langgraph_to_adk_tool.

        Returns:
            The wrapped tool function.
        """
        # Create decorated function
        @langgraph_to_adk_tool(tool_name, description, **kwargs)
        async def tool_wrapper(state: dict) -> dict:
            if hasattr(langgraph_runnable, "ainvoke"):
                return await langgraph_runnable.ainvoke(state)
            elif hasattr(langgraph_runnable, "invoke"):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None,
                    lambda: langgraph_runnable.invoke(state),
                )
            else:
                return langgraph_runnable(state)

        self._tools[tool_name] = tool_wrapper
        logger.info(f"Registered LangGraph tool: {tool_name}")
        return tool_wrapper

    def register_node(
        self,
        adk_agent: Any,
        node_name: Optional[str] = None,
        **kwargs,
    ) -> Callable:
        """Register an ADK agent as a LangGraph node.

        Args:
            adk_agent: ADK agent instance.
            node_name: Name for the node.
            **kwargs: Additional arguments for adk_to_langgraph_node.

        Returns:
            The node function.
        """
        node = adk_to_langgraph_node(adk_agent, name=node_name, **kwargs)
        name = node_name or getattr(adk_agent, "name", "adk_agent")
        self._nodes[name] = node
        logger.info(f"Registered ADK node: {name}")
        return node

    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a registered tool."""
        return self._tools.get(name)

    def get_node(self, name: str) -> Optional[Callable]:
        """Get a registered node."""
        return self._nodes.get(name)

    def list_tools(self) -> list[str]:
        """List registered tool names."""
        return list(self._tools.keys())

    def list_nodes(self) -> list[str]:
        """List registered node names."""
        return list(self._nodes.keys())


# ============================================================================
# Example Hybrid Workflow
# ============================================================================


async def example_hybrid_workflow():
    """Example demonstrating bidirectional LangGraph integration.

    This shows how to:
    1. Use a LangGraph analysis pipeline as an ADK tool
    2. Use an ADK research agent as a LangGraph node
    """

    # Example LangGraph-style analysis function
    @langgraph_to_adk_tool(
        "analyze_data",
        description="Analyze research data using LangGraph pipeline",
    )
    async def langgraph_analyzer(state: dict) -> dict:
        """Simulated LangGraph analysis pipeline."""
        input_text = state.get("input", "")

        # Simulate multi-step analysis
        analysis = {
            "summary": f"Analysis of: {input_text[:100]}...",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "confidence": 0.85,
        }

        return {"output": str(analysis)}

    # Example usage as ADK tool
    print("=== Using LangGraph as ADK Tool ===")
    result = await langgraph_analyzer.adk_invoke("Research topic about AI")
    print(f"Result: {result}")

    # Example ADK agent mock
    class MockADKAgent:
        """Mock ADK agent for demonstration."""

        name = "researcher"

        async def ainvoke(self, query: str) -> str:
            return f"Research findings for: {query}"

    # Convert to LangGraph node
    researcher = MockADKAgent()
    research_node = adk_to_langgraph_node(
        researcher,
        state_key="query",
        result_key="research",
    )

    # Example usage as LangGraph node
    print("\n=== Using ADK as LangGraph Node ===")
    state = {"query": "What is quantum computing?"}
    result_state = await research_node(state)
    print(f"Result state: {result_state}")


if __name__ == "__main__":
    asyncio.run(example_hybrid_workflow())
