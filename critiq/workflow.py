import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Literal, Sequence

from prettytable import PrettyTable
from tqdm import tqdm

from .agent import Agent
from .evaluator import PairEvaluator, ZeroOneEvaluator, get_evaluator_cls_from_dataset
from .i18n import local_prompts
from .types import (
    Criterion,
    DataType,
    PairData,
    ZeroOneData,
)
from .utils import (
    MANAGER_MAX_CONCURRENT,
    USE_TQDM,
    criteria_list_to_dict,
    is_pair_dataset,
    is_zero_one_dataset,
    parse_json,
    print_debug,
    random_reverse,
    reverse_ab,
)


class Workflow:
    """A workflow for managing and optimizing evaluation criteria."""
    
    def __init__(
        self,
        manager_args: dict[str, Any] | None = None,
        worker_args: dict[str, Any] | None = None,
        worker_max_concurrent: int = 1,
        init_criteria: (
            Sequence[Criterion | dict[Literal["name", "description"], str]] | None
        ) = None,
        n_criteria: int = 10,
        manager_prompt: str | None = None,
        manager_prompt_postfix: str | None = None,
        worker_prompt: str | None = None,
        use_tqdm: bool | None = None,
        show_debug: bool | None = None,
        manager_max_concurrent: int | None = None,
    ) -> None:
        # Environment variables with defaults
        self.use_tqdm = use_tqdm if use_tqdm is not None else os.environ.get("WORKFLOW_USE_TQDM", "0") == "1"
        self.show_debug = show_debug if show_debug is not None else os.environ.get("WORKFLOW_SHOW_DEBUG", "0") == "1"
        self.manager_max_concurrent = manager_max_concurrent if manager_max_concurrent is not None else int(os.environ.get("WORKFLOW_MANAGER_MAX_CONCURRENT", "1"))
        
        # Model configuration
        self.manager_args = {
            "model": "gpt-4o",
            "request_kwargs": {
                "temperature": 1.0,
            },
        }
        if manager_args is not None:
            self.manager_args.update(manager_args)

        self.worker_args = {"model": "gpt-4o-mini"}
        if worker_args is not None:
            self.worker_args.update(worker_args)

        self.worker_max_concurrent = worker_max_concurrent

        # Criteria management
        self.current_criteria: list[Criterion] = (
            [
                c if isinstance(c, Criterion) else Criterion.from_dict(c)
                for c in init_criteria
            ]
            if init_criteria
            else []
        )
        self.banned_criteria: set[str] = set()
        self.all_criteria: list[Criterion] = deepcopy(self.current_criteria)

        self.n_criteria = n_criteria

        # Prompts
        self.manager_prompt = (
            manager_prompt
            or local_prompts.MANAGER_PROMPT_TEMPLATE.format(n_criteria=self.n_criteria)
        )
        self.manager_prompt_postfix = (
            manager_prompt_postfix or local_prompts.MANAGER_PROMPT_POSTFIX
        )

        self.worker_prompt = worker_prompt
        self.thoughts = []

    def set_current_criteria(self, criteria: Sequence[Criterion | dict[Literal["name", "description"], str]]) -> None:
        """Set the current criteria list.
        
        Args:
            criteria: List of criteria to set as current criteria
        """
        self.current_criteria = [
            c if isinstance(c, Criterion) else Criterion.from_dict(c)
            for c in criteria
        ]
        self._update_criteria(self.all_criteria, self.current_criteria)

    def _update_criteria(
        self,
        old: Sequence[Criterion],
        new: Sequence[Criterion],
        only_higher_score: bool = True,
    ) -> None:
        if only_higher_score:
            _old = {c.name: c for c in deepcopy(old)}
            for i in deepcopy(new):
                if i.name in _old:
                    if i.score >= _old[i.name].score:
                        _old[i.name] = i
                else:
                    _old[i.name] = i
            old[:] = list(_old.values())
        else:
            old[:] = list(criteria_list_to_dict(deepcopy(old) + deepcopy(new)).values())

    @staticmethod
    def _warmup_zero_one(
        manager: Agent,
        dataset: Sequence[ZeroOneData],
        prompt_template: tuple[str, str] | None = None,
    ) -> None:
        if prompt_template is None:
            prompt_template = (
                local_prompts.WARMUP_ZERO_ONE_PROMPT_TEMPLATE_0,
                local_prompts.WARMUP_ZERO_ONE_PROMPT_TEMPLATE_1,
            )
        assert "{text}" in prompt_template[0] and "{text}" in prompt_template[1]

        prompts = []
        for data in dataset:
            prompts.append(
                prompt_template[data["label"]].replace("{text}", data["text"])
            )
        random.shuffle(prompts)

        for prompt in tqdm(
            prompts,
            desc="Warming up",
            disable=not USE_TQDM,
        ):
            manager(prompt, stream=False)

    @staticmethod
    def _warmup_pair(
        manager: Agent,
        dataset: Sequence[PairData],
        prompt_template: tuple[str, str] | None = None,
    ) -> None:
        if prompt_template is None:
            prompt_template = (
                local_prompts.WARMUP_PAIR_PROMPT_TEMPLATE_AB,
                local_prompts.WARMUP_PAIR_PROMPT_TEMPLATE_BA,
            )
        assert all(["{A}" in p and "{B}" in p for p in prompt_template])

        prompts = []
        for data in dataset:
            prompts.append(
                prompt_template[0 if data["answer"] == "A" else 1]
                .replace("{A}", data["A"])
                .replace("{B}", data["B"])
            )
        random.shuffle(prompts)

        for prompt in tqdm(
            prompts,
            desc="Warming up",
            disable=not USE_TQDM,
        ):
            manager(prompt, stream=False)

    def get_init_criteria(
        self,
        dataset: Sequence[ZeroOneData] | Sequence[PairData],
        prompt_template: tuple[str, str] | None = None,
        knowledge_base: Sequence[Criterion] | None = None,
        max_retries: int = 3,
        n_shot: int = -1,
        max_retrived: int = None,
        retrieval_threashold: float = 0.5,
    ) -> None:
        init_criteria = []
        if knowledge_base is not None:
            n_to_retrieve = (
                min(max_retrived, self.n_criteria)
                if max_retrived is not None
                else self.n_criteria
            )
            knowledge_base = deepcopy(knowledge_base)

            evaluator_cls = get_evaluator_cls_from_dataset(dataset)
            evaluator = evaluator_cls(
                self.worker_args,
                dataset,
                self.worker_max_concurrent,
                worker_prompt=self.worker_prompt,
                max_retries=max_retries,
            )
            evaluator.eval(knowledge_base, update_score=True)
            knowledge_base = list(
                filter(lambda x: x.score > retrieval_threashold, knowledge_base)
            )
            knowledge_base = sorted(knowledge_base, key=lambda x: x.score, reverse=True)

            # Get top n_to_retrieve. For those with the same score, randomly select.
            buffer = []
            for c in knowledge_base:
                if len(buffer) == 0:
                    buffer.append(c)
                elif c.score == buffer[0].score:
                    buffer.append(c)
                elif len(init_criteria) + len(buffer) < n_to_retrieve:
                    init_criteria.extend(buffer)
                    buffer = [c]
                else:
                    break
            random.shuffle(buffer)
            init_criteria.extend(buffer[: n_to_retrieve - len(init_criteria)])
            print(f"Retrieved {len(init_criteria)} criteria from knowledge base.")

        num_new_criteria = self.n_criteria - len(init_criteria)
        if num_new_criteria > 0:
            print(f"Generating {num_new_criteria} new criteria")
            manager = Agent(**self.manager_args)

            if len(dataset) <= 0:
                raise ValueError("Empty dataset")
            if n_shot > 0:
                assert n_shot <= len(dataset)
                dataset = random.sample(dataset, n_shot)
            if is_zero_one_dataset(dataset):  # ZeroOneData
                self._warmup_zero_one(manager, dataset, prompt_template)
            elif is_pair_dataset(dataset):  # PairData
                random_reversed_dataset = random_reverse(dataset)
                self._warmup_pair(manager, random_reversed_dataset, prompt_template)
            else:
                raise ValueError("Invalid dataset format")

            try:
                prompt = self.manager_prompt.replace(
                    str(self.n_criteria), str(num_new_criteria)
                )
                if len(init_criteria) > 0:
                    prompt += (
                        "\n\nThe new criteria should be different from the following:\n"
                    )
                    prompt += "\n".join(
                        [f"{c.name}: {c.description}" for c in init_criteria]
                    )
                prompt += self.manager_prompt_postfix
                response = manager(prompt, stream=True)
                init_criteria += [
                    Criterion(name=name, description=description, score=0.0)
                    for name, description in parse_json(response).items()
                ]
            except Exception as e:
                raise ValueError("Failed to parse initial criteria") from e

        init_criteria = init_criteria[: self.n_criteria]
        self._update_criteria(self.current_criteria, init_criteria)
        self._update_criteria(self.all_criteria, init_criteria)

    def _optimize_loop_pair_data(
        self,
        train_set: Sequence[PairData],
        threshold: tuple[float, float],
    ):
        assert 0 <= threshold[0] < threshold[1] <= 1
        manager = Agent(**self.manager_args)

        manager.history = [
            {
                "role": "user",
                "content": self.manager_prompt + self.manager_prompt_postfix,
            },
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps({c.name: c.description for c in self.current_criteria}, indent=4, ensure_ascii=False)}\n```",
            },
        ]
        evaluator = PairEvaluator(
            worker_args=self.worker_args,
            dataset=train_set,
            max_concurrent=self.worker_max_concurrent,
            worker_prompt=self.worker_prompt,
        )

        eval_output = evaluator.eval(self.current_criteria, update_score=True)
        self._update_criteria(self.all_criteria, self.current_criteria)
        print("Train:", eval_output.accuracy, eval_output.is_correct)

        criteria = criteria_list_to_dict(self.current_criteria)
        prompt = local_prompts.ACCURACY_PROMPT + "\n\n"
        good_criteria: dict[str, Criterion] = {}
        mid_criteria: dict[str, Criterion] = {}
        low_criteria: list[str] = []

        acc_table = PrettyTable()
        acc_table.field_names = ["Criterion", "Type", "Accuracy"]
        for criterion_name in sorted(eval_output.per_criterion_acc):
            acc = eval_output.per_criterion_acc[criterion_name]
            prompt += f"{criterion_name}: {acc}\n"
            if acc >= threshold[1]:
                good_criteria[criterion_name] = criteria[criterion_name].description
                acc_table.add_row([criterion_name, "Good", acc])
            elif acc > threshold[0]:
                mid_criteria[criterion_name] = None
                acc_table.add_row([criterion_name, "Mid", acc])
            else:
                low_criteria.append(criterion_name)
                acc_table.add_row([criterion_name, "Low", acc])
        print(acc_table)

        pred_result = eval_output.prediction
        thoughts = eval_output.thoughts
        new_criteria = {}
        if len(good_criteria) > 0:
            prompt += "\n\n"
            prompt += local_prompts.GOOD_CRITERIA_PROMPT_TEMPLATE.format(
                criteria=", ".join(good_criteria.keys()), threshold=threshold[1]
            )
            new_criteria.update(good_criteria)
            print("\n\n#### Good")
            print(", ".join(good_criteria.keys()))

        if len(mid_criteria) > 0:
            print("\n\n#### Mid")
            mid_table = PrettyTable()
            mid_table.field_names = ["Criterion", "Old", "New"]
            mid_table.align["Old"] = "l"
            mid_table.align["New"] = "l"
            mid_table.max_width["Old"] = 50
            mid_table.max_width["New"] = 50

            def _optimize_get_critic(
                criterion_name,
                data,
                answer,
                wrong_answer,
                thought,
                manager_for_critique,
            ):
                # Var `prompt`, `mid_criteria` and `local_prompts` are captured from context
                prompt_for_critique = "\n\n".join(
                    (
                        prompt,
                        local_prompts.MID_CRITERIA_PROMPT_TEMPLATE.format(
                            criterion_name=criterion_name,
                            threshold_0=threshold[0],
                            threshold_1=threshold[1],
                        ),
                        local_prompts.CRITERION_NAME_DESC_FORMAT_TEMPLAT.format(
                            name=criterion_name, desc=mid_criteria[criterion_name]
                        ),
                        local_prompts.MID_CRITIQUE_PROMPT,
                        local_prompts.MID_A_PROMPT_TEMPLATE.format(data["A"]),
                        local_prompts.MID_B_PROMPT_TEMPLATE.format(data["B"]),
                        local_prompts.MID_HOWEVER_PROMPT_TEMPLATE.format(
                            wrong=wrong_answer,
                            correct=answer,
                            thought=thought,
                        ),
                        local_prompts.MID_REFLECTION_PROMPT,
                    )
                )
                try:
                    response = manager_for_critique(prompt_for_critique, stream=False)
                    return parse_json(response)["critique"]
                except Exception as e:
                    print_debug("Failed to parse critique", e)
                    return None

            with ThreadPoolExecutor(
                max_workers=MANAGER_MAX_CONCURRENT  # TODO: by config instead of env var
            ) as executor:
                futures = {}
                for criterion_name in mid_criteria:
                    f_c = []
                    for idx, data in enumerate(train_set):
                        stat = pred_result[idx][criterion_name]
                        thought = thoughts[idx][criterion_name]
                        answer = data["answer"]
                        wrong_answer = reverse_ab(data["answer"])
                        # Only do refletion on false cases
                        if stat[wrong_answer] > stat[answer]:
                            f_c.append(
                                executor.submit(
                                    _optimize_get_critic,
                                    criterion_name,
                                    data,
                                    answer,
                                    wrong_answer,
                                    thought,
                                    manager.fork(),
                                )
                            )
                    futures[criterion_name] = f_c

                all_futures = sum(futures.values(), [])
                for _ in tqdm(
                    as_completed(all_futures),
                    desc="Reflection",
                    total=len(all_futures),
                    disable=not USE_TQDM,
                ):
                    pass
                critiques = {
                    c: [f.result() for f in futures_c]
                    for c, futures_c in futures.items()
                }

                futures = []
                for criterion_name in mid_criteria:
                    critique_prompt = "\n".join(
                        f"{i}: {c}" for i, c in enumerate(critiques[criterion_name])
                    )
                    _prompt = "\n\n".join(
                        (
                            prompt,
                            local_prompts.MID_CRITERIA_PROMPT_TEMPLATE.format(
                                criterion_name=criterion_name,
                                threshold_0=threshold[0],
                                threshold_1=threshold[1],
                            ),
                            local_prompts.CRITERION_NAME_DESC_FORMAT_TEMPLAT.format(
                                name=criterion_name, desc=mid_criteria[criterion_name]
                            ),
                            local_prompts.MID_REFINE_PROMPT_TEMPLATE.format(
                                critique_prompt
                            ),
                            local_prompts.MID_FORMAT_PROMPT_TEMPLATE.format(
                                criterion_name=criterion_name
                            ),
                        )
                    )
                    # Fork a new agent to avoid out of context window.
                    futures.append(
                        executor.submit(manager.fork(), _prompt, stream=False)
                    )

                responses = [
                    future.result()
                    for future in tqdm(
                        as_completed(futures),
                        desc="Optimization",
                        total=len(futures),
                        disable=not USE_TQDM,
                    )
                ]

                for criterion_name, response in zip(mid_criteria, responses):
                    if response is not None:
                        try:
                            _new = parse_json(response)
                            new_criteria.update(_new)
                            mid_table.add_row(
                                [
                                    criterion_name,
                                    criteria[criterion_name].description,
                                    _new[criterion_name],
                                ]
                            )
                        except Exception as e:
                            print_debug(
                                f"Failed to parse new criteria {criterion_name}", e
                            )

            print(mid_table)

        if len(low_criteria) > 0:
            print("\n#### Low")
            low_table = PrettyTable()
            low_table.field_names = ["Criterion", "Description"]
            low_table.align["Description"] = "l"
            low_table.max_width["Description"] = 100
            print(", ".join(low_criteria))
            self.banned_criteria.update(low_criteria)
            if len(low_criteria) != self.n_criteria - len(new_criteria):
                warnings.warn(
                    f"Num of low_criteria({len(low_criteria)}) != n_criteria - #new_criteria ({self.n_criteria - len(new_criteria)})"
                )
            prompt += (
                "\n\n"
                + local_prompts.LOW_PROMPT_TEMPLATE.format(
                    criteria=", ".join(self.banned_criteria),
                    threshold_0=threshold[0],
                    num=len(low_criteria),
                )
                + "\n\n"
                + local_prompts.LOW_FORMAT_PROMPT
            )
            response = manager(prompt, stream=False)
            if response is not None:
                try:
                    _new = parse_json(response)
                    new_criteria.update(_new)
                    for name, desc in _new.items():
                        low_table.add_row([name, desc])
                except Exception as e:
                    print_debug("Failed to parse new criteria", e)
            print(low_table)

        self.current_criteria = [
            Criterion.from_dict({"name": name, "description": desc})
            for name, desc in new_criteria.items()
        ]

    def _optimize_loop_zero_one_data(
        self,
        train_set: Sequence[ZeroOneData],
        threshold: tuple[float, float],
    ):
        assert 0 <= threshold[0] < threshold[1] <= 1
        manager = Agent(**self.manager_args)

        manager.history = [
            {
                "role": "user",
                "content": self.manager_prompt + self.manager_prompt_postfix,
            },
            {
                "role": "assistant",
                "content": f"```json\n{json.dumps({c.name: c.description for c in self.current_criteria}, indent=4, ensure_ascii=False)}\n```",
            },
        ]
        evaluator = ZeroOneEvaluator(
            worker_args=self.worker_args,
            dataset=train_set,
            max_concurrent=self.worker_max_concurrent,
            worker_prompt=self.worker_prompt,
        )

        eval_output = evaluator.eval(self.current_criteria, update_score=True)
        self._update_criteria(self.all_criteria, self.current_criteria)
        print("Train:", eval_output.accuracy, eval_output.is_correct)

        criteria = criteria_list_to_dict(self.current_criteria)
        prompt = local_prompts.ACCURACY_PROMPT + "\n\n"
        good_criteria: dict[str, Criterion] = {}
        mid_criteria: dict[str, Criterion] = {}
        low_criteria: list[str] = []
        for criterion_name in sorted(eval_output.per_criterion_acc):
            acc = eval_output.per_criterion_acc[criterion_name]
            print(f"{criterion_name}:\t{acc}")
            prompt += f"{criterion_name}: {acc}\n"
            if acc >= threshold[1]:
                good_criteria[criterion_name] = criteria[criterion_name].description
            elif acc > threshold[0]:
                mid_criteria[criterion_name] = None
            else:
                low_criteria.append(criterion_name)
        prompt += "\n"

        pred_result = eval_output.prediction
        new_criteria = {}
        if len(good_criteria) > 0:
            prompt += local_prompts.GOOD_CRITERIA_PROMPT_TEMPLATE.format(
                criteria=", ".join(good_criteria.keys()), threshold=threshold[1]
            )
            new_criteria.update(good_criteria)
            print("\n===== Good")
            print(", ".join(good_criteria.keys()))

        if len(mid_criteria) > 0:
            print("\n===== Mid")
            for criterion_name in mid_criteria:
                _manager = manager.fork()
                _prompt = prompt[:]
                _prompt += (
                    local_prompts.MID_CRITERIA_PROMPT_TEMPLATE.format(
                        criterion_name=criterion_name,
                        threshold_0=threshold[0],
                        threshold_1=threshold[1],
                    )
                    + "\n\n"
                )
                for idx, data in enumerate(train_set):
                    stat = pred_result[idx][criterion_name]
                    if stat[0] != stat[1]:
                        answer = 1 if stat[1] > stat[0] else 0
                        _prompt += (
                            local_prompts.MID_01_PROMPT_TEMPLATE.format(
                                text=data["text"]
                            )
                            + "\n\n"
                            + (
                                local_prompts.MID_0_FOR_1_PROMPT
                                if answer == 0
                                else local_prompts.MID_1_FOR_0_PROMPT
                            )
                        )

                _prompt += (
                    "\n\n"
                    + local_prompts.MID_REFINE_PROMPT_TEMPLATE
                    + "\n\n"
                    + local_prompts.MID_FORMAT_PROMPT_TEMPLATE.format(
                        criterion_name=criterion_name
                    )
                )

                response = _manager(_prompt, stream=False)
                if response is not None:
                    try:
                        _new = parse_json(response)
                        new_criteria.update(_new)
                        print(criterion_name, "\t->\t", _new[criterion_name])
                    except Exception as e:
                        print_debug("Failed to parse new criteria", e)

        if len(low_criteria) > 0:
            print("\n===== Low")
            print(", ".join(low_criteria))
            self.banned_criteria.update(low_criteria)
            prompt += (
                local_prompts.LOW_PROMPT_TEMPLATE.format(
                    criteria=", ".join(self.banned_criteria),
                    threshold_0=threshold[0],
                    num=self.n_criteria - len(new_criteria),
                )
                + "\n\n"
                + local_prompts.LOW_FORMAT_PROMPT
            )
            response = manager(prompt, stream=False)
            if response is not None:
                prompt = ""
                try:
                    _new = parse_json(response)
                    new_criteria.update(parse_json(response))
                    print("NEW:")
                    print(json.dumps(_new, indent=4, ensure_ascii=False))
                except Exception as e:
                    print_debug("Failed to parse new criteria", e)

        self.current_criteria = [
            Criterion.from_dict({"name": name, "description": desc})
            for name, desc in new_criteria.items()
        ]

    def optimize(
        self,
        train_set: Sequence[PairData | ZeroOneData],
        valid_set: Sequence[PairData | ZeroOneData] | None = None,
        output_dir: str | None = None,
        num_epochs: int = 1,
        threshold: tuple[float, float] = (0.5, 0.75),
        save_thought: bool = False,
        max_retries: int = 3,
    ) -> None:
        do_valid = valid_set is not None and len(valid_set) != 0
        if len(self.current_criteria) <= 0:
            raise ValueError(
                "No initial criteria, please give some or call `get_init_criteria`"
            )

        if output_dir is None:
            warnings.warn("`output_dir` is not set. Results will not be saved.")
        elif not os.path.exists(output_dir):
            os.mkdir(output_dir)

        if is_zero_one_dataset(train_set):
            evaluator_class = ZeroOneEvaluator
        elif is_pair_dataset(train_set):
            evaluator_class = PairEvaluator
        else:
            raise ValueError("Invalid dataset format")
        if do_valid:
            valid_evaluator = evaluator_class(
                self.worker_args,
                valid_set,
                self.worker_max_concurrent,
                worker_prompt=self.worker_prompt,
                max_retries=max_retries,
            )

        if is_pair_dataset(train_set):
            train_set = random_reverse(train_set)
        for epoch in range(num_epochs):
            print(f"## Iteration {epoch}")
            # In _optimize_loop:
            # 1. Evaluate `current_criteria` on the training set.
            # 2. update `all_critieria` by `current_criteria` if the
            #    score is higher.
            # 3. Optimize the criteria by the scores, `current_criteria` will
            #    be updated. New `current_criteria` will be used in the next iter.
            #
            # After this loop, you'll get a new **unmerged** `current_criteria`
            # and an updated `all_criteria`.
            if is_pair_dataset(train_set):
                self._optimize_loop_pair_data(train_set, threshold)
            elif is_zero_one_dataset(train_set):
                self._optimize_loop_zero_one_data(train_set, threshold)
            else:
                raise ValueError("Invalid trainset format")
            if do_valid:
                eval_output = valid_evaluator.eval(
                    self.current_criteria, update_score=False
                )
                print(eval_output)
            # thoughts = eval_output.thoughts if save_thought else None
            if output_dir is not None:
                self.save(output_dir, epoch=epoch, thought=None)

        # The scores of final criteria after optimization is not updated on the
        # training set while not merged to `all_critieria`. So we need to update
        # the scores and update the `all_criteria` here.
        evaluator = evaluator_class(
            worker_args=self.worker_args,
            dataset=train_set,
            max_concurrent=self.worker_max_concurrent,
            worker_prompt=self.worker_prompt,
        )
        eval_output = evaluator.eval(self.current_criteria, update_score=True)
        self._update_criteria(self.all_criteria, self.current_criteria)
        print("Final Train Acc:", eval_output.accuracy, eval_output.is_correct)
        if output_dir is not None:
            self.save(output_dir, epoch="final", thought=None)

    def get_best_criteria(self, threshold: int = 0.75) -> list[Criterion]:
        return [c for c in self.all_criteria if c.score >= threshold]

    def get_state_dict(self) -> dict[str, Any]:
        return {
            "manager_args": self.manager_args,
            "worker_args": self.worker_args,
            "worker_max_concurrent": self.worker_max_concurrent,
            "current_criteria": [c.to_dict() for c in self.current_criteria],
            "banned_criteria": list(self.banned_criteria),
            "all_criteria": [c.to_dict() for c in self.all_criteria],
            "n_criteria": self.n_criteria,
            "manager_prompt": self.manager_prompt,
            "manager_prompt_postfix": self.manager_prompt_postfix,
            "worker_prompt": self.worker_prompt,
        }

    def load_state_dict(self, state: dict[str, Any]):
        current_state = self.get_state_dict()
        current_state.update(state)
        self.manager_args = current_state["manager_args"]
        self.worker_args = current_state["worker_args"]
        self.worker_max_concurrent = current_state["worker_max_concurrent"]

        self.current_criteria = [
            Criterion.from_dict(c) for c in current_state["current_criteria"]
        ]
        self.banned_criteria.update(current_state["banned_criteria"])
        self.all_criteria = [
            Criterion.from_dict(c) for c in current_state["all_criteria"]
        ]

        self.n_criteria = current_state["n_criteria"]
        self.manager_prompt = current_state["manager_prompt"]
        self.manager_prompt_postfix = current_state["manager_prompt_postfix"]
        self.worker_prompt = current_state["worker_prompt"]

    def save(self, path, epoch, thought) -> None:
        with open(os.path.join(path, f"epoch_{epoch}.json"), "w", encoding="utf8") as f:
            json.dump(
                self.get_state_dict(),
                f,
                ensure_ascii=False,
                indent=4,
            )
        if thought:
            with open(
                os.path.join(path, f"thought_{epoch}.json"), "w", encoding="utf8"
            ) as f:
                json.dump(
                    thought,
                    f,
                    ensure_ascii=False,
                    indent=4,
                )

    def load(self, path: str) -> None:
        with open(path, "r", encoding="utf8") as f:
            state = json.load(f)
        self.load_state_dict(state)
