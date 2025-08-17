from github import Github
import os

gh = Github(os.getenv("GITHUB_TOKEN"))

# Test: list your repositories
for repo in gh.get_user().get_repos():
    print(repo.full_name)
