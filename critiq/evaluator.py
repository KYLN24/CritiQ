from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Literal, Sequence

from tqdm import tqdm

from .agent import Agent
from .i18n import local_prompts
from .types import EvaluationOutput, PredictionOutput, PredictionOutputWithAnswer
from .utils import (
    USE_TQDM,
    Criterion,
    PairData,
    ZeroOneData,
    is_pair_dataset,
    is_zero_one_dataset,
    parse_json,
    print_debug,
    reverse_ab,
)


class Evaluator:
    @abstractmethod
    def pred(
        self, criteria: Sequence[Criterion | dict[Literal["name", "description"], str]]
    ) -> PredictionOutputWithAnswer: ...

    @abstractmethod
    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
    ) -> EvaluationOutput: ...


class BaselinePairEvaluator(Evaluator):
    worker_prompt_postfix = local_prompts.BASELINE_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
    ) -> None:
        self.worker_args = worker_args
        self.dataset = dataset
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries

        self.worker_prompt = worker_prompt or local_prompts.BASELINE_WORKER_PROMPT
        assert all([k in self.worker_prompt for k in ["{C}", "{A}", "{B}"]])

    def voting_fn(self, prediction: PredictionOutput) -> list[Literal["A", "B", None]]:
        result = []
        for d in prediction:
            stat = {"A": 0, "B": 0}
            for c in d.values():
                stat["A"] += c["A"]
                stat["B"] += c["B"]
            if stat["A"] > stat["B"]:
                result.append("A")
            elif stat["B"] > stat["A"]:
                result.append("B")
            else:
                result.append(None)
        return result

    def _make_prompt(self, data, criteria):
        c = "\n".join([f"- {c.name}: {c.description}" for c in criteria])
        prompt = (
            self.worker_prompt.replace("{C}", c)
            .replace("{A}", data["A"])
            .replace("{B}", data["B"])
        )
        prompt += self.worker_prompt_postfix
        return prompt

    def _pred_one_openai(
        self, data: PairData, criteria: Criterion, ttl: int
    ) -> tuple[Literal["A", "B", "U"] | None, str | None]:
        worker = Agent(**self.worker_args)
        prompt = self._make_prompt(data, criteria)
        response = worker(prompt, stream=False)
        thought = None
        try:
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criteria, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None, thought
        if answer == "N":  # None
            answer = "U"
        if answer not in ("A", "B", "U"):
            answer = None
        return answer, thought

    def _pred_openai_one_worker(
        self, args: tuple[PairData, Sequence[Criterion]]
    ) -> tuple[dict[dict[Literal["A", "B", "U"], int]], dict[str, str]]:
        data, criteria = args
        prediction = {"all": {"A": 0, "B": 0, "U": 0}}
        thoughts = {"all": None}

        one_pred, thought = self._pred_one_openai(data, criteria, self.max_retries)
        if one_pred in ("A", "B", "U"):
            prediction["all"][one_pred] += 1
            thoughts["all"] = thought

        return prediction, thoughts

    def pred_openai(self, criteria: Sequence[Criterion]):
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            prediction = list(
                tqdm(
                    t.map(
                        self._pred_openai_one_worker,
                        [(data, criteria) for data in self.dataset],
                    ),
                    total=len(self.dataset),
                    dynamic_ncols=True,
                    disable=not USE_TQDM,
                )
            )
            thoughts = [p[1] for p in prediction]
            prediction = [p[0] for p in prediction]
        return prediction, thoughts

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)
        prediction, thoughts = self.pred_openai(criteria)

        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            is_correct.append(a is not None and a == d["answer"])

        per_criterion_acc = {"all": 0}  # All criteria are evaluated together

        n_correct = 0
        n_total = len(self.dataset)
        for d, p in zip(self.dataset, prediciton):
            if p["all"][d["answer"]] > p["all"][reverse_ab(d["answer"])]:
                n_correct += 1
        per_criterion_acc["all"] = n_correct / n_total

        return EvaluationOutput(
            prediction=prediciton,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
            thoughts=prediction_with_answer.thoughts,
        )


class PairEvaluator(Evaluator):
    worker_prompt_postfix = local_prompts.PAIR_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[PairData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
    ) -> None:
        self.worker_args = worker_args
        self.dataset = dataset
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.max_data_chars = max_data_chars

        self.worker_prompt = worker_prompt or local_prompts.PAIR_WORKER_PROMPT
        assert all(
            [
                k in self.worker_prompt
                for k in ["{criterion}", "{description}", "{A}", "{B}"]
            ]
        )

    def voting_fn(
        self, prediction: PredictionOutput, threshold: int = 0
    ) -> list[Literal["A", "B", None]]:
        result = []
        for d in prediction:
            stat = {"A": 0, "B": 0}
            for c in d.values():
                stat["A"] += c["A"]
                stat["B"] += c["B"]
            if stat["A"] - stat["B"] > threshold:
                result.append("A")
            elif stat["B"] - stat["A"] > threshold:
                result.append("B")
            else:
                result.append(None)
        return result

    def _make_prompt(self, data, criterion):
        a = data["A"][: self.max_data_chars] if self.max_data_chars else data["A"]
        b = data["B"][: self.max_data_chars] if self.max_data_chars else data["B"]
        prompt = (
            self.worker_prompt.replace("{criterion}", criterion.name)
            .replace("{description}", criterion.description)
            .replace("{A}", a)
            .replace("{B}", b)
        )
        prompt += self.worker_prompt_postfix
        return prompt

    def _pred_one_openai(
        self, data: PairData, criterion: Criterion, ttl: int
    ) -> tuple[Literal["A", "B", "U"] | None, str | None]:
        worker = Agent(**self.worker_args)
        prompt = self._make_prompt(data, criterion)
        response = worker(prompt, stream=False)
        thought = None
        try:
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criterion, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None, thought
        if answer == "N":  # None
            answer = "U"
        if answer not in ("A", "B", "U"):
            answer = None
        return answer, thought

    def pred_openai(self, criteria: Sequence[Criterion]):
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            futures = []
            for data in self.dataset:
                for criterion in criteria:
                    future = t.submit(
                        self._pred_one_openai, data, criterion, self.max_retries
                    )
                    futures.append(future)

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass

            prediction = []
            thoughts = []
            results = (future.result() for future in futures)
            for data in self.dataset:
                _prediction = {
                    criterion.name: {"A": 0, "B": 0, "U": 0} for criterion in criteria
                }
                _thoughts = {criterion.name: None for criterion in criteria}
                for criterion in criteria:
                    one_pred, thought = next(results)
                    if one_pred in ("A", "B", "U"):
                        _prediction[criterion.name][one_pred] += 1
                        _thoughts[criterion.name] = thought
                prediction.append(_prediction)
                thoughts.append(_thoughts)

        return prediction, thoughts

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)
        prediction, thoughts = self.pred_openai(criteria)
        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            is_correct.append(a is not None and a == d["answer"])

        per_criterion_acc = {criterion.name: 0 for criterion in criteria}
        for criterion in criteria:
            n_correct = 0
            n_total = 0
            for d, p in zip(self.dataset, prediciton):
                if p[criterion.name]["U"] > 0:
                    # Only count when the worker is sure and crriterion is applicable.
                    # If the answer is None, which means the output is invalid to resolve.
                    # It will not be passed and will be regarded as incorrect.
                    continue
                n_total += 1
                if (
                    p[criterion.name][d["answer"]]
                    > p[criterion.name][reverse_ab(d["answer"])]
                ):
                    n_correct += 1
            per_criterion_acc[criterion.name] = (
                0 if n_correct == 0 else n_correct / n_total
            )
            if update_score:
                criterion.score = per_criterion_acc[criterion.name]

        return EvaluationOutput(
            prediction=prediciton,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
            thoughts=prediction_with_answer.thoughts,
        )


class ZeroOneEvaluator(Evaluator):
    worker_prompt_postfix = local_prompts.ZERO_ONE_WORKER_PROMPT_POSTFIX

    def __init__(
        self,
        worker_args: dict,
        dataset: Sequence[ZeroOneData],
        max_concurrent: int = 1,
        max_retries: int = 3,
        worker_prompt: str | None = None,
        max_data_chars: int | None = None,
    ) -> None:
        self.worker_args = worker_args
        self.dataset = dataset
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.max_data_chars = max_data_chars

        self.worker_prompt = worker_prompt or local_prompts.ZERO_ONE_WORKER_PROMPT
        assert all(
            [
                k in self.worker_prompt
                for k in ["{criterion}", "{description}", "{text}"]
            ]
        )

    def voting_fn(
        self, prediction: PredictionOutput, threshold: int = 0
    ) -> list[Literal[0, 1, None]]:
        result = []
        for d in prediction:
            stat = {0: 0, 1: 0}
            for c in d.values():
                stat[1] += c[1]
                stat[0] += c[0]
            if stat[1] - stat[0] > threshold:
                result.append(1)
            elif stat[1] - stat[0] < threshold:
                result.append(0)
            else:
                result.append(None)
        return result

    def _make_prompt(self, data, criterion):
        text = (
            data["text"][: self.max_data_chars] if self.max_data_chars else data["text"]
        )
        prompt = (
            self.worker_prompt.replace("{criterion}", criterion.name)
            .replace("{description}", criterion.description)
            .replace("{text}", text)
        )
        prompt += self.worker_prompt_postfix
        return prompt

    def _pred_one_openai(
        self, data: PairData, criterion: Criterion, ttl: int
    ) -> tuple[Literal[0, 1] | None, str | None]:
        worker = Agent(**self.worker_args)
        prompt = self._make_prompt(data, criterion)
        response = worker(prompt, stream=False)
        thought = None
        try:
            response = parse_json(response)
            answer = response["answer"].strip()[0].upper()
            thought = response["thought"].strip()
        except Exception as e:  # pylint: disable=W0718:broad-exception-caught
            if ttl > 0:
                print_debug(
                    f"Failed to parse worker response, retrying {ttl=}", response, e
                )
                return self._pred_one_openai(data, criterion, ttl - 1)
            else:
                print_debug("Failed to parse worker response", response, e)
                return None
        if answer not in ("Y", "N"):
            return None, thought
        return {"Y": 1, "N": 0}[answer], thought

    def pred_openai(self, criteria: Sequence[Criterion]):
        with ThreadPoolExecutor(max_workers=self.max_concurrent) as t:
            futures = []
            for data in self.dataset:
                for criterion in criteria:
                    future = t.submit(
                        self._pred_one_openai, data, criterion, self.max_retries
                    )
                    futures.append(future)

            for _ in tqdm(
                as_completed(futures),
                total=len(futures),
                dynamic_ncols=True,
                disable=not USE_TQDM,
            ):
                pass

            prediction = []
            thoughts = []
            results = (future.result() for future in futures)
            for data in self.dataset:
                _prediction = {criterion.name: {0: 0, 1: 0} for criterion in criteria}
                _thoughts = {criterion.name: None for criterion in criteria}
                for criterion in criteria:
                    one_pred, thought = next(results)
                    if one_pred in (0, 1):
                        _prediction[criterion.name][one_pred] += 1
                        _thoughts[criterion.name] = thought
                prediction.append(_prediction)
                thoughts.append(_thoughts)

        return prediction, thoughts

    def pred(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        **voting_fn_kwargs,
    ) -> PredictionOutputWithAnswer:
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)
        prediction, thoughts = self.pred_openai(criteria)
        return PredictionOutputWithAnswer(
            prediction=prediction,
            answer=self.voting_fn(prediction, **voting_fn_kwargs),
            thoughts=thoughts,
        )

    def eval(
        self,
        criteria: Sequence[Criterion | dict[Literal["name", "description"], str]],
        update_score=False,
        **voting_fn_kwargs,
    ) -> EvaluationOutput:
        for i, c in enumerate(criteria):
            if isinstance(c, dict):
                criteria[i] = Criterion.from_dict(c)

        prediction_with_answer = self.pred(criteria, **voting_fn_kwargs)
        prediciton = prediction_with_answer.prediction
        answer = prediction_with_answer.answer

        is_correct = []
        for d, a in zip(self.dataset, answer):
            is_correct.append(a is not None and a == d["label"])

        per_criterion_acc = {criterion.name: 0 for criterion in criteria}
        for criterion in criteria:
            n_correct = 0
            n_total = len(self.dataset)
            for d, p in zip(self.dataset, prediciton):
                if p[criterion.name][d["label"]] > p[criterion.name][1 - d["label"]]:
                    n_correct += 1
            per_criterion_acc[criterion.name] = (
                0 if n_correct == 0 else n_correct / n_total
            )
            if update_score:
                criterion.score = per_criterion_acc[criterion.name]

        return EvaluationOutput(
            prediction=prediciton,
            is_correct=is_correct,
            per_criterion_acc=per_criterion_acc,
            accuracy=len(list(filter(None, is_correct))) / len(is_correct),
        )


def get_evaluator_cls_from_dataset(dataset: Sequence[dict]):
    if is_zero_one_dataset(dataset):
        return ZeroOneEvaluator
    elif is_pair_dataset(dataset):
        return PairEvaluator
    else:
        raise ValueError("Invalid validset format")
