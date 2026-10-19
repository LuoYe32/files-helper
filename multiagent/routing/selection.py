class Selection:
    """Files chosen for the run (by the router or by the user); None means all files in data/."""

    def __init__(self, files: list[str] | None = None) -> None:
        self.files = set(files) if files else None

    def restrict(self, names: set[str]) -> set[str]:
        return names if self.files is None else names & self.files
