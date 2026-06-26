class Langraph:
    """Simple base class for the langraph package."""

    def __init__(self, name: str = "langraph") -> None:
        self.name = name

    def describe(self) -> str:
        return f"Langraph package initialized for {self.name}"
