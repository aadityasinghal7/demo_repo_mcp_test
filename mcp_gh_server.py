"""
MCP server for GitHub automation:
- Creates PR for feature branch
- Uses OpenAI to generate test suggestion
- Adds test to repo
- Merges PR only if tests pass
"""

import os
import subprocess
from github import Github
from openai import OpenAI
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load secrets
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPO_NAME = "aadityasinghal7/mcp-test-pipeline"

# Clients
gh = Github(GITHUB_TOKEN)
repo = gh.get_repo(REPO_NAME)
openai = OpenAI(api_key=OPENAI_API_KEY)

mcp = FastMCP("github-ci-mcp")

@mcp.tool()
def create_pr(feature_branch: str, base_branch: str = "main"):
    """
    Create a pull request from feature branch to main
    """
    pr = repo.create_pull(
        title=f"Feature: {feature_branch}",
        body="Auto-created PR by MCP server",
        head=feature_branch,
        base=base_branch,
    )
    return {"pr_url": pr.html_url, "pr_number": pr.number}

@mcp.tool()
def suggest_test(file_path: str, function_name: str):
    """
    Use OpenAI to suggest a test for a new function.
    """
    with open(file_path, "r") as f:
        code = f.read()

    prompt = f"""
    You are reviewing this Python code:
    {code}

    Suggest a pytest test for the function `{function_name}`.
    Write only Python code (no explanation).
    """

    resp = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    test_code = resp.choices[0].message.content
    test_file = f"tests/test_{function_name}.py"

    with open(test_file, "w") as f:
        f.write(test_code)

    subprocess.run(["git", "add", test_file])
    subprocess.run(["git", "commit", "-m", f"Add test for {function_name}"])
    subprocess.run(["git", "push", "origin", "HEAD"])

    return {"test_file": test_file, "content": test_code}

@mcp.tool()
def merge_if_tests_pass(pr_number: int):
    """
    Merge PR only if CI tests pass.
    """
    pr = repo.get_pull(pr_number)
    statuses = repo.get_commit(pr.head.sha).get_statuses()

    for status in statuses:
        if status.state != "success":
            return {"merged": False, "reason": "CI checks not passed"}

    pr.merge(merge_method="squash")
    return {"merged": True, "pr_url": pr.html_url}

if __name__ == "__main__":
    mcp.run()
