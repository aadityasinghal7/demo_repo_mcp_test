"""
MCP CLI for GitHub automation:
- Creates PR for a feature branch
- Suggests a pytest test using OpenAI
- Adds test to repo
- Merges PR if tests pass
"""

import os
import subprocess
import argparse
from github import Github
from openai import OpenAI
from fastmcp import FastMCP
from dotenv import load_dotenv

# ---------------- Load Secrets ---------------- #
load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPO_NAME = "aadityasinghal7/mcp-test-pipeline"

# ---------------- Clients ---------------- #
gh = Github(GITHUB_TOKEN)
repo = gh.get_repo(REPO_NAME)
openai = OpenAI(api_key=OPENAI_API_KEY)

# ---------------- Initialize MCP ---------------- #
mcp = FastMCP("github-ci-mcp")

# ---------------- MCP Tools ---------------- #

@mcp.tool()
def create_pr(feature_branch: str, base_branch: str = "main"):
    """Create a pull request from feature branch to main"""
    pr = repo.create_pull(
        title=f"Feature: {feature_branch}",
        body="Auto-created PR by MCP CLI",
        head=feature_branch,
        base=base_branch,
    )
    return {"pr_url": pr.html_url, "pr_number": pr.number}


@mcp.tool()
def suggest_test(file_path: str, function_name: str):
    """Use OpenAI to suggest a pytest test for a function"""
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

    os.makedirs("tests", exist_ok=True)
    with open(test_file, "w") as f:
        f.write(test_code)

    subprocess.run(["git", "add", test_file], check=True)
    subprocess.run(["git", "commit", "-m", f"Add test for {function_name}"], check=True)
    subprocess.run(["git", "push", "origin", "HEAD"], check=True)

    return {"test_file": test_file, "content": test_code}


@mcp.tool()
def merge_if_tests_pass(pr_number: int):
    """Merge PR only if CI tests pass"""
    pr = repo.get_pull(pr_number)
    statuses = repo.get_commit(pr.head.sha).get_statuses()

    for status in statuses:
        if status.state != "success":
            return {"merged": False, "reason": "CI checks not passed"}

    pr.merge(merge_method="squash")
    return {"merged": True, "pr_url": pr.html_url}


# ---------------- CLI Entry Point ---------------- #
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP GitHub Automation CLI")
    parser.add_argument("--tool", required=True, choices=["create_pr", "suggest_test", "merge_if_tests_pass"])
    parser.add_argument("--feature_branch", help="Feature branch for PR")
    parser.add_argument("--file_path", help="File path for suggesting test")
    parser.add_argument("--function_name", help="Function name for test suggestion")
    parser.add_argument("--pr_number", type=int, help="PR number to merge if tests pass")

    args = parser.parse_args()

    if args.tool == "create_pr":
        if not args.feature_branch:
            raise ValueError("feature_branch is required for create_pr")
        result = create_pr(args.feature_branch)

    elif args.tool == "suggest_test":
        if not args.file_path or not args.function_name:
            raise ValueError("file_path and function_name are required for suggest_test")
        result = suggest_test(args.file_path, args.function_name)

    elif args.tool == "merge_if_tests_pass":
        if not args.pr_number:
            raise ValueError("pr_number is required for merge_if_tests_pass")
        result = merge_if_tests_pass(args.pr_number)

    print(result)
