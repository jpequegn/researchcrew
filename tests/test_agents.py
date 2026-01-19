"""Tests for ResearchCrew Agent Instructions

Tests agent instruction definitions and content.
Note: Agent objects are mocked since google.adk is not available in CI.
"""


class TestResearcherInstructions:
    """Tests for the Researcher agent instructions."""

    def test_instructions_defined(self):
        """Test that researcher instructions are defined."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS

        assert RESEARCHER_INSTRUCTIONS is not None
        assert len(RESEARCHER_INSTRUCTIONS) > 100

    def test_instructions_mention_knowledge_base(self):
        """Test that instructions include knowledge base workflow."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS

        assert "knowledge base" in RESEARCHER_INSTRUCTIONS.lower()

    def test_instructions_mention_search(self):
        """Test that instructions mention search capabilities."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS

        assert "search" in RESEARCHER_INSTRUCTIONS.lower()

    def test_instructions_mention_citations(self):
        """Test that instructions require source citations."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS

        assert "cite" in RESEARCHER_INSTRUCTIONS.lower() or "source" in RESEARCHER_INSTRUCTIONS.lower()

    def test_instructions_have_constraints(self):
        """Test that instructions include constraints."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS

        assert "constraint" in RESEARCHER_INSTRUCTIONS.lower() or "maximum" in RESEARCHER_INSTRUCTIONS.lower()

    def test_instructions_mention_confidence(self):
        """Test that instructions mention confidence scores."""
        from agents.researcher import RESEARCHER_INSTRUCTIONS

        assert "confidence" in RESEARCHER_INSTRUCTIONS.lower()


class TestSynthesizerInstructions:
    """Tests for the Synthesizer agent instructions."""

    def test_instructions_defined(self):
        """Test that synthesizer instructions are defined."""
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS

        assert SYNTHESIZER_INSTRUCTIONS is not None
        assert len(SYNTHESIZER_INSTRUCTIONS) > 100

    def test_instructions_mention_synthesis(self):
        """Test that instructions describe synthesis responsibilities."""
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS

        instructions_lower = SYNTHESIZER_INSTRUCTIONS.lower()
        assert any(word in instructions_lower for word in ["synthesize", "combine", "integrate"])

    def test_instructions_mention_sources(self):
        """Test that instructions mention working with sources."""
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS

        assert "source" in SYNTHESIZER_INSTRUCTIONS.lower()


class TestFactCheckerInstructions:
    """Tests for the Fact Checker agent instructions."""

    def test_instructions_defined(self):
        """Test that fact checker instructions are defined."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS

        assert FACT_CHECKER_INSTRUCTIONS is not None
        assert len(FACT_CHECKER_INSTRUCTIONS) > 100

    def test_instructions_mention_verification(self):
        """Test that instructions describe verification responsibilities."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS

        instructions_lower = FACT_CHECKER_INSTRUCTIONS.lower()
        assert "verify" in instructions_lower or "check" in instructions_lower or "validate" in instructions_lower

    def test_instructions_mention_claims(self):
        """Test that instructions mention claims or facts."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS

        instructions_lower = FACT_CHECKER_INSTRUCTIONS.lower()
        assert "claim" in instructions_lower or "fact" in instructions_lower


class TestWriterInstructions:
    """Tests for the Writer agent instructions."""

    def test_instructions_defined(self):
        """Test that writer instructions are defined."""
        from agents.writer import WRITER_INSTRUCTIONS

        assert WRITER_INSTRUCTIONS is not None
        assert len(WRITER_INSTRUCTIONS) > 100

    def test_instructions_mention_writing(self):
        """Test that instructions describe writing responsibilities."""
        from agents.writer import WRITER_INSTRUCTIONS

        instructions_lower = WRITER_INSTRUCTIONS.lower()
        assert "write" in instructions_lower or "report" in instructions_lower or "document" in instructions_lower

    def test_instructions_mention_structure(self):
        """Test that instructions mention document structure."""
        from agents.writer import WRITER_INSTRUCTIONS

        instructions_lower = WRITER_INSTRUCTIONS.lower()
        assert "structure" in instructions_lower or "format" in instructions_lower or "section" in instructions_lower


class TestOrchestratorInstructions:
    """Tests for the Orchestrator agent instructions."""

    def test_instructions_defined(self):
        """Test that orchestrator instructions are defined."""
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS

        assert ORCHESTRATOR_INSTRUCTIONS is not None
        assert len(ORCHESTRATOR_INSTRUCTIONS) > 100

    def test_instructions_mention_coordination(self):
        """Test that instructions describe coordination responsibilities."""
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS

        instructions_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        assert (
            "coordinate" in instructions_lower
            or "delegate" in instructions_lower
            or "orchestrate" in instructions_lower
            or "workflow" in instructions_lower
        )

    def test_instructions_mention_agents(self):
        """Test that instructions mention working with other agents."""
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS

        instructions_lower = ORCHESTRATOR_INSTRUCTIONS.lower()
        assert "agent" in instructions_lower


class TestInstructionQuality:
    """Tests for overall instruction quality."""

    def test_all_instructions_non_empty(self):
        """Test that all instructions are non-empty strings."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS
        from agents.researcher import RESEARCHER_INSTRUCTIONS
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS
        from agents.writer import WRITER_INSTRUCTIONS

        instructions = [
            RESEARCHER_INSTRUCTIONS,
            SYNTHESIZER_INSTRUCTIONS,
            FACT_CHECKER_INSTRUCTIONS,
            WRITER_INSTRUCTIONS,
            ORCHESTRATOR_INSTRUCTIONS,
        ]

        for instruction in instructions:
            assert isinstance(instruction, str)
            assert len(instruction.strip()) > 50

    def test_instructions_are_unique(self):
        """Test that each agent has unique instructions."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS
        from agents.researcher import RESEARCHER_INSTRUCTIONS
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS
        from agents.writer import WRITER_INSTRUCTIONS

        instructions = [
            RESEARCHER_INSTRUCTIONS,
            SYNTHESIZER_INSTRUCTIONS,
            FACT_CHECKER_INSTRUCTIONS,
            WRITER_INSTRUCTIONS,
            ORCHESTRATOR_INSTRUCTIONS,
        ]

        # Each instruction should be unique
        assert len(instructions) == len(set(instructions)), "Agent instructions should be unique"

    def test_instructions_have_role_definition(self):
        """Test that instructions include role definitions."""
        from agents.fact_checker import FACT_CHECKER_INSTRUCTIONS
        from agents.orchestrator import ORCHESTRATOR_INSTRUCTIONS
        from agents.researcher import RESEARCHER_INSTRUCTIONS
        from agents.synthesizer import SYNTHESIZER_INSTRUCTIONS
        from agents.writer import WRITER_INSTRUCTIONS

        instructions = {
            "researcher": RESEARCHER_INSTRUCTIONS,
            "synthesizer": SYNTHESIZER_INSTRUCTIONS,
            "fact_checker": FACT_CHECKER_INSTRUCTIONS,
            "writer": WRITER_INSTRUCTIONS,
            "orchestrator": ORCHESTRATOR_INSTRUCTIONS,
        }

        for name, instruction in instructions.items():
            # Instructions should mention "you are" or similar role definition
            assert "you are" in instruction.lower() or "your" in instruction.lower(), (
                f"{name} instructions should define the agent's role"
            )
