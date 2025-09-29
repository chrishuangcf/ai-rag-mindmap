"""
Manages the dynamic state of the application, such as the current LLM provider.
This allows for runtime configuration changes without restarting the server.
"""
from src.core.config import LLM_PROVIDER, OPENROUTER_MODEL, OLLAMA_MODEL

class AppState:
    def __init__(self):
        self._llm_provider = LLM_PROVIDER
        self._llm_model = self.get_model_for_provider(self._llm_provider)

    def get_model_for_provider(self, provider: str) -> str:
        """Returns the model name based on the provider."""
        if provider == "openrouter":
            return OPENROUTER_MODEL
        elif provider == "ollama":
            return OLLAMA_MODEL
        return "unknown"

    @property
    def llm_provider(self) -> str:
        return self._llm_provider

    @llm_provider.setter
    def llm_provider(self, provider: str):
        if provider not in ["openrouter", "ollama"]:
            raise ValueError("Invalid LLM provider specified.")
        self._llm_provider = provider
        self._llm_model = self.get_model_for_provider(provider)

    @property
    def llm_model(self) -> str:
        return self._llm_model

# Global state instance
app_state = AppState()

def get_app_state():
    """Returns the global application state instance."""
    return app_state
