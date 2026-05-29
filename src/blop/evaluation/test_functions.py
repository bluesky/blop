import numpy as np


class TestFunctionEvaluation:
    def __call__(self, uid: str, suggestions: list[dict]) -> list[dict]:

        outcomes = []

        for suggestion in suggestions:
            suggestion_id = suggestion["_id"]
            x1 = suggestion["x1"]
            x2 = suggestion["x2"]

            fitness = np.exp(-((x1 - 2 * x2 - 1) ** 2) - 1e-3 * (2 * x1 + x2 - 0.5) ** 2)
            fitness += 1e-2 * np.random.standard_normal()

            outcomes.append(
                {
                    "_id": suggestion_id,
                    "fitness": fitness,
                }
            )

        return outcomes
