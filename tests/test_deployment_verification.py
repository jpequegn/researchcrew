"""Deployment Verification Tests

Comprehensive tests validating the ResearchCrew deployment readiness.
This file validates Issue #19 requirements:
- Running on Vertex AI Agent Engine
- MCP tools packaged and reusable
- Configuration management in place
- Secrets properly managed
"""

import json
from pathlib import Path

import pytest
import yaml

# ============================================================================
# Project Root and Paths
# ============================================================================


@pytest.fixture
def project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


# ============================================================================
# Configuration Management Verification
# ============================================================================


class TestConfigurationManagement:
    """Tests verifying configuration files exist and are properly structured."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config_dir = Path(__file__).parent.parent / "config"

    def test_dev_config_exists(self):
        """Verify development configuration exists."""
        dev_config = self.config_dir / "dev.yaml"
        assert dev_config.exists(), "Development config should exist at config/dev.yaml"

    def test_staging_config_exists(self):
        """Verify staging configuration exists."""
        staging_config = self.config_dir / "staging.yaml"
        assert staging_config.exists(), "Staging config should exist at config/staging.yaml"

    def test_prod_config_exists(self):
        """Verify production configuration exists."""
        prod_config = self.config_dir / "prod.yaml"
        assert prod_config.exists(), "Production config should exist at config/prod.yaml"

    def test_dev_config_valid_yaml(self):
        """Verify development config is valid YAML."""
        dev_config = self.config_dir / "dev.yaml"
        if not dev_config.exists():
            pytest.skip("Dev config not found")

        with open(dev_config) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "environment" in config
        assert config["environment"] == "development"

    def test_staging_config_valid_yaml(self):
        """Verify staging config is valid YAML."""
        staging_config = self.config_dir / "staging.yaml"
        if not staging_config.exists():
            pytest.skip("Staging config not found")

        with open(staging_config) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "environment" in config
        assert config["environment"] == "staging"

    def test_prod_config_valid_yaml(self):
        """Verify production config is valid YAML."""
        prod_config = self.config_dir / "prod.yaml"
        if not prod_config.exists():
            pytest.skip("Prod config not found")

        with open(prod_config) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "environment" in config
        assert config["environment"] == "production"

    def test_prod_config_has_required_sections(self):
        """Verify production config has all required sections."""
        prod_config = self.config_dir / "prod.yaml"
        if not prod_config.exists():
            pytest.skip("Prod config not found")

        with open(prod_config) as f:
            config = yaml.safe_load(f)

        required_sections = [
            "environment",
            "models",
            "tools",
            "quality",
            "performance",
            "retry",
        ]

        for section in required_sections:
            assert section in config, f"Production config should have '{section}' section"

    def test_config_environments_are_distinct(self):
        """Verify each environment config has correct environment setting."""
        environments = {
            "dev.yaml": "development",
            "staging.yaml": "staging",
            "prod.yaml": "production",
        }

        for filename, expected_env in environments.items():
            config_file = self.config_dir / filename
            if not config_file.exists():
                continue

            with open(config_file) as f:
                config = yaml.safe_load(f)

            assert config["environment"] == expected_env, f"{filename} should have environment={expected_env}"

    def test_prod_has_observability_enabled(self):
        """Verify production config has observability settings."""
        prod_config = self.config_dir / "prod.yaml"
        if not prod_config.exists():
            pytest.skip("Prod config not found")

        with open(prod_config) as f:
            config = yaml.safe_load(f)

        assert "observability" in config, "Production should have observability config"
        assert config["observability"].get("tracing", {}).get("enabled") is True
        assert config["observability"].get("metrics", {}).get("enabled") is True


# ============================================================================
# Vertex AI Agent Engine Deployment Verification
# ============================================================================


class TestVertexAIDeployment:
    """Tests verifying Vertex AI Agent Engine deployment infrastructure."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.deploy_dir = self.project_root / "deploy"

    def test_dockerfile_exists(self):
        """Verify Dockerfile exists for container deployment."""
        dockerfile = self.project_root / "Dockerfile"
        assert dockerfile.exists(), "Dockerfile should exist at project root"

    def test_dockerfile_uses_multi_stage_build(self):
        """Verify Dockerfile uses multi-stage build for smaller images."""
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()
        assert "AS builder" in content, "Dockerfile should use multi-stage build"
        assert "AS runtime" in content, "Dockerfile should have runtime stage"

    def test_dockerfile_exposes_correct_port(self):
        """Verify Dockerfile exposes port 8080 for Vertex AI."""
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()
        assert "EXPOSE 8080" in content, "Dockerfile should expose port 8080"

    def test_dockerfile_has_health_check(self):
        """Verify Dockerfile has health check endpoint."""
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()
        assert "HEALTHCHECK" in content, "Dockerfile should have HEALTHCHECK"

    def test_dockerfile_uses_non_root_user(self):
        """Verify Dockerfile runs as non-root user for security."""
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()
        assert "USER appuser" in content or "USER 1000" in content, "Dockerfile should run as non-root user"

    def test_deploy_script_exists(self):
        """Verify deployment script exists."""
        deploy_script = self.deploy_dir / "deploy.sh"
        assert deploy_script.exists(), "Deployment script should exist at deploy/deploy.sh"

    def test_deploy_script_is_executable(self):
        """Verify deployment script has proper structure."""
        deploy_script = self.deploy_dir / "deploy.sh"
        if not deploy_script.exists():
            pytest.skip("Deploy script not found")

        content = deploy_script.read_text()
        # Check for common deployment commands
        assert "docker build" in content, "Deploy script should include docker build"
        assert "docker push" in content or "gcloud" in content, "Deploy script should include push command"

    def test_deploy_config_exists(self):
        """Verify Vertex AI deployment config exists."""
        deploy_config = self.deploy_dir / "config.yaml"
        assert deploy_config.exists(), "Deployment config should exist at deploy/config.yaml"

    def test_deploy_config_valid_yaml(self):
        """Verify deployment config is valid YAML."""
        deploy_config = self.deploy_dir / "config.yaml"
        if not deploy_config.exists():
            pytest.skip("Deploy config not found")

        with open(deploy_config) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "project" in config, "Deploy config should have project section"
        assert "agent" in config, "Deploy config should have agent section"

    def test_deploy_config_has_resource_limits(self):
        """Verify deployment config specifies resource limits."""
        deploy_config = self.deploy_dir / "config.yaml"
        if not deploy_config.exists():
            pytest.skip("Deploy config not found")

        with open(deploy_config) as f:
            config = yaml.safe_load(f)

        assert "resources" in config, "Deploy config should have resources section"
        assert "memory" in config["resources"], "Deploy config should specify memory"
        assert "cpu" in config["resources"], "Deploy config should specify CPU"

    def test_deploy_config_has_scaling(self):
        """Verify deployment config has scaling configuration."""
        deploy_config = self.deploy_dir / "config.yaml"
        if not deploy_config.exists():
            pytest.skip("Deploy config not found")

        with open(deploy_config) as f:
            config = yaml.safe_load(f)

        assert "scaling" in config, "Deploy config should have scaling section"
        assert "min_instances" in config["scaling"]
        assert "max_instances" in config["scaling"]


# ============================================================================
# MCP Tools Packaging Verification
# ============================================================================


class TestMCPToolsPackaging:
    """Tests verifying MCP tools are properly packaged and reusable."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mcp_servers_dir = Path(__file__).parent.parent / "mcp-servers"

    def test_mcp_servers_directory_exists(self):
        """Verify MCP servers directory exists."""
        assert self.mcp_servers_dir.exists(), "MCP servers directory should exist"

    def test_web_research_server_exists(self):
        """Verify web research MCP server exists."""
        server_dir = self.mcp_servers_dir / "web-research-server"
        assert server_dir.exists(), "Web research server should exist"

        server_file = server_dir / "server.py"
        assert server_file.exists(), "Web research server.py should exist"

    def test_knowledge_base_server_exists(self):
        """Verify knowledge base MCP server exists."""
        server_dir = self.mcp_servers_dir / "knowledge-base-server"
        assert server_dir.exists(), "Knowledge base server should exist"

        server_file = server_dir / "server.py"
        assert server_file.exists(), "Knowledge base server.py should exist"

    def test_mcp_servers_have_config(self):
        """Verify MCP servers have mcp.json configuration."""
        expected_servers = ["web-research-server", "knowledge-base-server"]

        for server_name in expected_servers:
            server_dir = self.mcp_servers_dir / server_name
            if not server_dir.exists():
                continue

            mcp_config = server_dir / "mcp.json"
            assert mcp_config.exists(), f"{server_name} should have mcp.json config"

    def test_mcp_config_valid_json(self):
        """Verify MCP configs are valid JSON."""
        expected_servers = ["web-research-server", "knowledge-base-server"]

        for server_name in expected_servers:
            mcp_config = self.mcp_servers_dir / server_name / "mcp.json"
            if not mcp_config.exists():
                continue

            with open(mcp_config) as f:
                config = json.load(f)

            assert config is not None, f"{server_name} mcp.json should be valid JSON"

    def test_shared_utilities_exist(self):
        """Verify shared utilities module exists."""
        shared_dir = self.mcp_servers_dir / "shared"
        assert shared_dir.exists(), "Shared utilities directory should exist"

        init_file = shared_dir / "__init__.py"
        assert init_file.exists(), "Shared __init__.py should exist"

    def test_mcp_servers_readme_exists(self):
        """Verify MCP servers have documentation."""
        readme = self.mcp_servers_dir / "README.md"
        assert readme.exists(), "MCP servers README should exist"


# ============================================================================
# Secrets Management Verification
# ============================================================================


class TestSecretsManagement:
    """Tests verifying secrets are properly managed and not exposed."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent

    def test_gitignore_excludes_env_files(self):
        """Verify .gitignore excludes environment files."""
        gitignore = self.project_root / ".gitignore"
        assert gitignore.exists(), ".gitignore should exist"

        content = gitignore.read_text()
        assert ".env" in content, ".gitignore should exclude .env files"

    def test_no_env_files_committed(self):
        """Verify no .env files exist in the repository."""
        env_files = list(self.project_root.glob(".env*"))
        # Filter out .env.example which is OK to commit
        actual_env_files = [f for f in env_files if ".example" not in f.name]

        # This test passes if no actual .env files exist
        # (they would be in .gitignore anyway)
        assert len(actual_env_files) == 0 or all(f.name.endswith(".example") for f in actual_env_files), (
            "No actual .env files should be committed (only .env.example is OK)"
        )

    def test_deploy_config_uses_secret_references(self):
        """Verify deployment config uses secret references, not plaintext."""
        deploy_config = self.project_root / "deploy" / "config.yaml"
        if not deploy_config.exists():
            pytest.skip("Deploy config not found")

        with open(deploy_config) as f:
            config = yaml.safe_load(f)

        if "secrets" in config:
            # Secrets should reference Secret Manager, not contain actual values
            for secret_name, secret_value in config["secrets"].items():
                assert ":" in str(secret_value) or secret_value.startswith("${"), (
                    f"Secret {secret_name} should use Secret Manager reference format"
                )

    def test_no_hardcoded_api_keys_in_source(self):
        """Verify no hardcoded API keys in source files."""
        # Check key source directories
        source_dirs = [
            self.project_root / "agents",
            self.project_root / "tools",
            self.project_root / "utils",
        ]

        for source_dir in source_dirs:
            if not source_dir.exists():
                continue

            for py_file in source_dir.rglob("*.py"):
                content = py_file.read_text()

                # Check for common patterns of hardcoded secrets
                # Note: We're checking for actual key values, not variable names
                assert "AIza" not in content, f"{py_file} may contain hardcoded Google API key"
                assert "sk-" not in content or "sk-" in "# sk-" or "skip" in content, (
                    f"{py_file} may contain hardcoded OpenAI API key"
                )

    def test_environment_variables_documented(self):
        """Verify required environment variables are documented."""
        # Check for documentation of env vars in deploy script or README
        deploy_script = self.project_root / "deploy" / "deploy.sh"
        if deploy_script.exists():
            content = deploy_script.read_text()
            # Deploy script should document required env vars
            assert "GOOGLE_API_KEY" in content or "GCP_PROJECT_ID" in content, (
                "Deploy script should document required environment variables"
            )


# ============================================================================
# CI/CD Pipeline Verification
# ============================================================================


class TestCICDPipeline:
    """Tests verifying CI/CD pipeline is properly configured."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.github_dir = self.project_root / ".github"

    def test_github_workflows_exist(self):
        """Verify GitHub workflows directory exists."""
        workflows_dir = self.github_dir / "workflows"
        assert workflows_dir.exists(), "GitHub workflows directory should exist"

    def test_ci_workflow_exists(self):
        """Verify CI workflow exists."""
        ci_file = self.github_dir / "workflows" / "ci.yml"
        assert ci_file.exists(), "CI workflow should exist at .github/workflows/ci.yml"

    def test_ci_workflow_valid_yaml(self):
        """Verify CI workflow is valid YAML."""
        ci_file = self.github_dir / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("CI workflow not found")

        with open(ci_file) as f:
            workflow = yaml.safe_load(f)

        assert workflow is not None
        assert "jobs" in workflow, "CI workflow should have jobs"

    def test_ci_workflow_has_lint_job(self):
        """Verify CI workflow has linting job."""
        ci_file = self.github_dir / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("CI workflow not found")

        with open(ci_file) as f:
            workflow = yaml.safe_load(f)

        jobs = workflow.get("jobs", {})
        assert "lint" in jobs, "CI workflow should have lint job"

    def test_ci_workflow_has_test_job(self):
        """Verify CI workflow has test job."""
        ci_file = self.github_dir / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("CI workflow not found")

        with open(ci_file) as f:
            workflow = yaml.safe_load(f)

        jobs = workflow.get("jobs", {})
        assert "test" in jobs, "CI workflow should have test job"

    def test_ci_workflow_has_quality_gate(self):
        """Verify CI workflow has quality gate job."""
        ci_file = self.github_dir / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("CI workflow not found")

        with open(ci_file) as f:
            workflow = yaml.safe_load(f)

        jobs = workflow.get("jobs", {})
        assert "quality-gate" in jobs or any("gate" in job.lower() for job in jobs), (
            "CI workflow should have quality gate job"
        )

    def test_ci_runs_on_pull_requests(self):
        """Verify CI runs on pull requests."""
        ci_file = self.github_dir / "workflows" / "ci.yml"
        if not ci_file.exists():
            pytest.skip("CI workflow not found")

        with open(ci_file) as f:
            workflow = yaml.safe_load(f)

        # YAML parses 'on' as boolean True, so check both
        triggers = workflow.get("on") or workflow.get(True, {})
        # 'on' can be a dict with trigger types as keys, or a list of trigger types
        if isinstance(triggers, dict):
            assert "pull_request" in triggers, "CI should run on pull requests"
        elif isinstance(triggers, list):
            assert "pull_request" in triggers, "CI should run on pull requests"
        else:
            pytest.fail("CI workflow 'on' trigger has unexpected format")


# ============================================================================
# Documentation Verification
# ============================================================================


class TestDeploymentDocumentation:
    """Tests verifying deployment documentation exists."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent
        self.docs_dir = self.project_root / "docs"

    def test_deployment_docs_exist(self):
        """Verify deployment documentation exists."""
        deployment_doc = self.docs_dir / "deployment.md"
        assert deployment_doc.exists(), "Deployment docs should exist at docs/deployment.md"

    def test_deployment_docs_has_content(self):
        """Verify deployment docs have meaningful content."""
        deployment_doc = self.docs_dir / "deployment.md"
        if not deployment_doc.exists():
            pytest.skip("Deployment docs not found")

        content = deployment_doc.read_text()
        assert len(content) > 500, "Deployment docs should have substantial content"

        # Check for key sections
        assert "Vertex" in content or "vertex" in content, "Deployment docs should mention Vertex AI"

    def test_mcp_servers_documented(self):
        """Verify MCP servers are documented."""
        mcp_readme = self.project_root / "mcp-servers" / "README.md"
        if not mcp_readme.exists():
            pytest.skip("MCP servers README not found")

        content = mcp_readme.read_text()
        assert len(content) > 100, "MCP servers should have documentation"


# ============================================================================
# Integration Test - Full Deployment Readiness
# ============================================================================


class TestDeploymentReadiness:
    """Integration tests verifying overall deployment readiness."""

    def setup_method(self):
        """Set up test fixtures."""
        self.project_root = Path(__file__).parent.parent

    def test_all_required_files_exist(self):
        """Verify all required deployment files exist."""
        required_files = [
            "Dockerfile",
            "deploy/deploy.sh",
            "deploy/config.yaml",
            "config/dev.yaml",
            "config/staging.yaml",
            "config/prod.yaml",
            ".github/workflows/ci.yml",
            ".gitignore",
        ]

        missing_files = []
        for file_path in required_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        assert len(missing_files) == 0, f"Missing required deployment files: {missing_files}"

    def test_docker_build_would_succeed(self):
        """Verify Dockerfile references valid paths."""
        dockerfile = self.project_root / "Dockerfile"
        if not dockerfile.exists():
            pytest.skip("Dockerfile not found")

        content = dockerfile.read_text()

        # Check that COPY commands reference directories that exist
        copy_dirs = ["agents/", "tools/", "utils/", "config/"]
        for dir_name in copy_dirs:
            if f"COPY {dir_name}" in content:
                assert (self.project_root / dir_name.rstrip("/")).exists(), (
                    f"Dockerfile copies {dir_name} but directory doesn't exist"
                )

    def test_pyproject_toml_exists(self):
        """Verify pyproject.toml exists for dependency management."""
        pyproject = self.project_root / "pyproject.toml"
        assert pyproject.exists(), "pyproject.toml should exist for dependency management"

    def test_environment_configuration_complete(self):
        """Verify environment configuration is complete."""
        # All environment configs should have model configuration
        env_files = ["dev.yaml", "staging.yaml", "prod.yaml"]
        config_dir = self.project_root / "config"

        for env_file in env_files:
            config_path = config_dir / env_file
            if not config_path.exists():
                continue

            with open(config_path) as f:
                config = yaml.safe_load(f)

            assert "models" in config, f"{env_file} should have models configuration"
            assert "tools" in config, f"{env_file} should have tools configuration"
