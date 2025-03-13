import json
from dataclasses import dataclass
from typing import Any, Literal, Type, TypedDict
from typing_extensions import NotRequired

@dataclass
class Criterion:
    name: str
    description: str
    score: float = 0  # highest valid accuracy

    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "score": self.score,
        }

    @staticmethod
    def from_dict(d):
        return Criterion(**d)


class DataType(TypedDict):
    """
    Abstract type for data.
    NotRequired fields are the output of the judge function,
    the others are the input fields.
    """

    def _get_input_fields(self):
        return list(self.__required_fields__)

    def _get_output_fields(self):
        return list(self.__optional_fields__)


class ZeroOneData(DataType):
    text: str
    label: NotRequired[Literal[0, 1]]


class PairData(DataType):
    A: str
    B: str
    answer: NotRequired[Literal["A", "B"]]


PredictionOutput = list[dict[str, dict[str | int, int]]]


@dataclass
class EvaluationOutput:
    prediction: PredictionOutput
    is_correct: list[bool]  # True if the voting result is correct
    per_criterion_acc: dict[str, float]  # accuracy for each criterion
    accuracy: float
    thoughts: list[dict[str, str]] | None = None

    def __str__(self):
        criteria = list(self.per_criterion_acc.keys())
        output_json = {}
        for c in criteria:
            n_refuse = sum([p[c]["U"] for p in self.prediction])
            output_json[c] = {
                "Accuracy": self.per_criterion_acc[c],
                "Refuse to Respond": f"{n_refuse / len(self.prediction)} ({n_refuse})",
            }
        return f"Accuracy: {self.accuracy}\nCorrect: {self.is_correct}\n{json.dumps(output_json, ensure_ascii=False, indent=4)}"


@dataclass
class PredictionOutputWithAnswer:
    prediction: PredictionOutput
    answer: list[Literal["A", "B", None] | Literal[0, 1, None]]
    thoughts: list[dict[str, str]] | None = None

@dataclass
class Prompts:
    ...

class BaseMetaTask:
    data_type: Type[DataType]
    prompts: Prompts

    def judge(self, criterion, data: DataType) -> Any:
        raise NotImplementedError

    def metric(self, data: DataType, judge_output: Any) -> float:
        raise NotImplementedError

    def voting_fn(self, judge_outputs: list[Any]) -> Any:
        raise NotImplementedError
