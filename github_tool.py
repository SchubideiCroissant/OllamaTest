import os
from github import Github
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
token = os.getenv("GITHUB_TOKEN")

if not token:
    raise ValueError("Fehler: Kein GitHub-Token gefunden (.env prüfen).")

gh = Github(token)

def get_repo_stats(repo_name: str):
    """Liefert allgemeine Informationen zu einem Repository."""
    repo = gh.get_repo(repo_name)
    return {
        "name": repo.full_name,
        "beschreibung": repo.description,
        "sterne": repo.stargazers_count,
        "forks": repo.forks_count,
        "issues_offen": repo.open_issues_count,
        "sprache": repo.language,
        "letztes_update": repo.updated_at.strftime("%d.%m.%Y %H:%M:%S")
    }

def get_last_commit(repo_name: str):
    """Gibt die Nachricht und das Datum des letzten Commits zurück."""
    repo = gh.get_repo(repo_name)
    commit = repo.get_commits()[0]
    return {
        "nachricht": commit.commit.message,
        "autor": commit.commit.author.name,
        "datum": commit.commit.author.date.strftime("%d.%m.%Y %H:%M:%S")
    }

def list_open_issues(repo_name: str):
    """Listet offene Issues auf."""
    repo = gh.get_repo(repo_name)
    issues = repo.get_issues(state="open")
    return [{"titel": i.title, "erstellt_von": i.user.login} for i in issues[:5]]

def list_user_repos(username: str = "SchubideiCroissant"):
    user = gh.get_user(username) if username else gh.get_user()
    repos = list(user.get_repos())
    repos.sort(key=lambda r: r.updated_at, reverse=True)

    return [{
        "name": repo.name,
        "language": repo.language or "Unbekannt",
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "visibility": "privat" if repo.private else "öffentlich",
        "last_update": repo.updated_at.strftime("%d.%m.%Y %H:%M:%S"),
    } for repo in repos]

def get_repo_details(repo_name: str):
    repo = gh.get_repo(repo_name)
    branches = list(repo.get_branches())
    commits = repo.get_commits().totalCount if hasattr(repo.get_commits(), "totalCount") else "?"
    topics = repo.get_topics()

    return {
        "name": repo.full_name,
        "description": repo.description or "",
        "language": repo.language or "Unbekannt",
        "stars": repo.stargazers_count,
        "forks": repo.forks_count,
        "branches": len(branches),
        "commits": commits,
        "topics": topics,
        "visibility": "privat" if repo.private else "öffentlich",
        "last_push": repo.pushed_at.strftime("%d.%m.%Y %H:%M:%S"),
    }

def format_result(data):
    """Hilfsfunktion für lesbare Ausgabe."""
    if isinstance(data, dict):
        return "\n".join(f"{k.capitalize()}: {v}" for k, v in data.items())
    elif isinstance(data, list):
        return "\n".join([f"- {i['titel']} (von {i['erstellt_von']})" for i in data])
    else:
        return str(data)


