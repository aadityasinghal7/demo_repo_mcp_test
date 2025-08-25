import os
import argparse
import fnmatch
from pathlib import Path
import json

from github import Github
from openai import OpenAI
from fastmcp import FastMCP
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
REPO_NAME = "aadityasinghal7/mecp-test-pipeline"

gh = Github(GITHUB_TOKEN)
repo = gh.get_repo(REPO_NAME)
openai = OpenAI(api_key=OPENAI_API_KEY)

mcp = FastMCP("github-ci-mcp")

IGNORE_PATTERNS = [
    "__pycache__",
    ".venv",
    ".git",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    ".pytest_cache",
]


def should_ignore(path: str) -> bool:
    return any(fnmatch.fnmatch(path, pat) or pat in Path(path).parts for pat in IGNORE_PATTERNS)


def _repo_snapshot(base_dir=".") -> str:
    allowed_root_files = {"README.md", "requirements.txt", "pyproject.toml"}
    chunks = []
    for root, dirs, files in os.walk(base_dir):
        # prune ignored dirs
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d))]
        # at repo root, keep only allowed files
        if os.path.abspath(root) == os.path.abspath(base_dir):
            files = [f for f in files if f in allowed_root_files]
        for f in files:
            file_path = os.path.join(root, f)
            if should_ignore(file_path):
                continue
            rel_path = os.path.relpath(file_path, base_dir)
            try:
                with open(file_path, "r", encoding="utf-8") as fh:
                    content = fh.read()
            except Exception:
                continue
            chunks.append(f"===== {rel_path} =====\n{content}\n")
    return "\n".join(chunks)


@mcp.tool()
def repo_snapshot() -> str:
    return _repo_snapshot(".")


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except Exception as e:
        return f"<<ERROR READING {path}: {e}>>"


def _build_testcase_prompt(test: dict, base_dir: str = ".") -> str:
    name = test.get("name", "")
    desc = test.get("one_line_description", "")
    ttype = test.get("type", "")
    # priority = test.get("priority", "")
    target = test.get("suggested_test_location", "")
    files = list(dict.fromkeys(test.get("relevant_files", [])))  # de-dupe, keep order

    chunks = []
    for rel in files:
        abs_path = os.path.join(base_dir, rel)
        chunks.append(f"===== {rel} =====\n{_read_text(abs_path)}\n")

    return (
        f"You are to create a single pytest test for the missing case '{name}'.\n"
        f"- Description: {desc}\n"
        f"- Type: {ttype}\n"
        f"- Write the test output to: {target}\n"
        f"- Use ONLY the repository context below.\n"
        f"- Output the full test file, including imports and any necessary setup that will overwrite the target file.\n"
        f"- Output only Python test code (no prose, no fences).\n\n"
        f"Repository context for this test:\n" + "\n".join(chunks)
    )


def _print_json(obj) -> None:
    """Pretty-print a Python object as JSON."""
    print(json.dumps(obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCP GitHub Automation CLI")
    parser.add_argument("--tool", choices=["repo_snapshot"], default="repo_snapshot")
    parser.add_argument("--use-agent", dest="use_agent", action="store_true")
    # parser.set_defaults(use_agent=True)

    args = parser.parse_args()

    if args.tool == "repo_snapshot":
        project_snapshot = _repo_snapshot(".")

        if args.use_agent:
            instructions = """
            You are an expert Python test strategist.

            Task:
            - Identify missing tests in the provided project snapshot.

            Output requirements:
            - Output ONLY valid JSON (UTF-8). No markdown, no prose, no code fences, no leading/trailing text.
            - Exact schema:
              {
                "missing_tests": [
                  {
                    "name": "short-name-or-id",
                    "one_line_description": "single-line summary of the missing test",
                    "type": "unit" | "integration" | "e2e",
                    "priority": "low" | "medium" | "high",
                    "relevant_files": ["relative/path/from/repo/root", "..."],
                    "suggested_test_location": "relative/path/for/new/test/file"
                  }
                ]
              }
            - Paths:
              - Use repository-relative paths with forward slashes.
              - Include only files strictly necessary to understand and implement the test.
              - Deduplicate and sort relevant_files alphabetically.
            - one_line_description must be <= 200 characters.
            - Do NOT include any test code.
            - If no tests are missing, return exactly: {"missing_tests": []}

            Guidance:
            - Consider complex business logic, boundary cases, error paths, and integration points.
            - Prefer covering public functions/classes with logic that lack tests.
            - If you are adding a test case, and other similar/related test cases already exist, prefer adding to those files
to the relevant_files list.
            - Prefer not overdoing test cases. For instance, if a repo only has tes
cases to a certain extent, do not suggest more nuanced tests than they have elsewhere.
            - If there are relevant doc files such as README files that are relevant to the functionality
being tested, include them in relevant_files.
            """
            prompt = f"""
            {instructions}

            Project Snapshot:

            {project_snapshot}
            """
            missing_tests_respose = openai.chat.completions.create(
                model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt}]
            )
            parsed = json.loads(missing_tests_respose.choices[0].message.content)
            missing_tests = parsed.get("missing_tests", [])
        else:
            json_path = "example_response1.json"
            with open(json_path, "r", encoding="utf-8") as fh:
                parsed = json.load(fh)
            missing_tests = parsed.get("missing_tests", [])

        _print_json(missing_tests)

        for test in missing_tests[:1]:
            prompt_text = _build_testcase_prompt(test, base_dir=".")
            print(f"\n\n===== Missing test: {test.get('name','<unnamed>')} =====\n{prompt_text}\n")
            test_code_response = openai.chat.completions.create(
                model="gpt-4.1-mini", messages=[{"role": "user", "content": prompt_text}]
            )
            test_code = test_code_response.choices[0].message.content
            test_file = test.get("suggested_test_location")
            os.makedirs(os.path.dirname(test_file), exist_ok=True)
            with open(test_file, "w", encoding="utf-8") as fh:
                fh.write(test_code)
            print(f"Written test code to {test_file}")
