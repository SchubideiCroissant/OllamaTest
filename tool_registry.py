from github_tool import*
import inspect

TOOLS = {
    "list_user_repos": {
        "function": list_user_repos,
        "description": "Listet alle Repositories eines Benutzers mit Basisstatistiken (Sprache, Sterne, Forks, letzte Aktualisierung)."
    },
    "get_repo_stats": {
        "function": get_repo_stats,
        "description": "Liefert Sterne, Forks, Issues und Sprache eines GitHub-Repositories."
    },
    "get_last_commit": {
        "function": get_last_commit,
        "description": "Gibt die Nachricht, den Autor und das Datum des letzten Commits zurück."
    },
    "list_open_issues": {
        "function": list_open_issues,
        "description": "Listet offene Issues eines Repositories (max. 5)."
    }
}
def generate_tool_descriptions(tools_dict):
    """Erstellt eine klare Toolbeschreibung inkl. Argumente und Signaturen."""
    lines = []
    for name, data in tools_dict.items():
        func = data["function"]
        sig = inspect.signature(func)
        params = ", ".join([f"{p}={v.default if v.default != inspect._empty else 'erforderlich'}"
                            for p, v in sig.parameters.items()])
        desc = data.get("description", "")
        lines.append(f"- **{name}({params})**: {desc}")
    return "\n".join(lines)

def format_output(data):
    return format_result(data)

print(generate_tool_descriptions(TOOLS))