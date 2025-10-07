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
    """Liefert allgemeine Informationen zu einem Repository, inklusive Commits."""
    # Falls kein User Input dann Token-User nehmen
    if "/" not in repo_name:
        user = gh.get_user().login
        repo_name = f"{user}/{repo_name}"

    repo = gh.get_repo(repo_name)

    # Commits abrufen (kann bei großen Repos dauern!)
    commits = list(repo.get_commits()[:2])  # nur die letzten 2 laden
    total_commits = repo.get_commits().totalCount  # Gesamtzahl der Commits

    last_commits = []
    for c in commits:
        last_commits.append({
            "nachricht": c.commit.message.split("\n")[0],  # erste Zeile der Nachricht
            "autor": c.commit.author.name if c.commit.author else "Unbekannt",
            "datum": c.commit.author.date.strftime("%d.%m.%Y %H:%M:%S") if c.commit.author else "?"
        })

    return {
        "name": repo.full_name,
        "beschreibung": repo.description,
        "sterne": repo.stargazers_count,
        "forks": repo.forks_count,
        "issues_offen": repo.open_issues_count,
        "sprache": repo.language,
        "commits_gesamt": total_commits,
        "letztes_update": repo.updated_at.strftime("%d.%m.%Y %H:%M:%S"),
        "letzte_commits": last_commits
    }


def get_last_commit(repo_name: str):
    """Gibt die Nachricht und das Datum des letzten Commits zurück."""
    # Falls kein User Input dann Token-User nehmen
    if "/" not in repo_name:
        user = gh.get_user().login
        repo_name = f"{user}/{repo_name}"

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


def format_result(data):
    """Formatiert Rückgaben der GitHub-Tools für die Terminal-Ausgabe."""
    if isinstance(data, dict):
        # Einzelnes Objekt (z. B. get_repo_stats)
        return "\n".join([f"{k.capitalize()}: {v}" for k, v in data.items()])

    elif isinstance(data, list):
        if not data:
            return "Keine Daten gefunden."

        # Prüfen, ob das Ergebnis aus list_user_repos stammt
        if "letztes_update" in data[0]:
            return "\n".join([
                f"- {i['name']} | Sterne: {i['sterne']} | Forks: {i['forks']} | "
                f"Sprache: {i['sprache']} | Letztes Update: {i['letztes_update']}"
                for i in data
            ])

        # Prüfen, ob es Issues sind
        elif "titel" in data[0]:
            return "\n".join([
                f"- {i['titel']} (von {i['erstellt_von']})"
                for i in data
            ])

        # Fallback für unbekannte Listen
        else:
            return "\n".join(map(str, data))

    else:
        return str(data)



