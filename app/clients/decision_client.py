import warnings

from langchain_typesafe import ClassifierResponse, Question, State, TypeSafeClassifier

from app.config.settings import DecisionSettings, load_decision_settings


class DecisionClient:
    def __init__(self, settings: DecisionSettings | None = None):
        self.settings = settings or load_decision_settings()
        with warnings.catch_warnings():
            # The upstream class is flagged beta on every construction. We pin the exact
            # version and read it through this module only, so the notice adds only noise.
            warnings.simplefilter("ignore")
            self._classifier_client = TypeSafeClassifier(
                base_url=self.settings.base_url,
                model=self.settings.model_name,
                api_key=self.settings.api_key,
            )

    def decide(self, state: State, questions: dict[str, Question]) -> ClassifierResponse:
        """Make a decision based on the provided state and question."""
        return self._classifier_client.invoke({"state": state, "questions": questions})
