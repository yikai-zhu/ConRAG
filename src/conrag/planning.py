from collections import defaultdict, deque
from typing import ReadOnly, TypedDict

from conrag.common import clean_text

NOT_FOUND = "Information not found"


class PlanStep(TypedDict):
    id: ReadOnly[int]
    sub_question: ReadOnly[str]
    dependencies: ReadOnly[list[int]]


def normalize_plan(raw: object) -> list[PlanStep]:
    if not isinstance(raw, list):
        return []

    steps: dict[int, PlanStep] = {}
    for item in raw:
        if not isinstance(item, dict):
            return []
        step_id = item.get("id")
        sub_question = clean_text(item.get("sub_question", ""))
        dependencies = item.get("dependencies", [])
        if (
            isinstance(step_id, bool)
            or not isinstance(step_id, int)
            or not sub_question
            or not isinstance(dependencies, list)
            or step_id in steps
        ):
            return []

        clean_dependencies: list[int] = []
        for dep_id in dependencies:
            if isinstance(dep_id, bool) or not isinstance(dep_id, int):
                return []
            clean_dependencies.append(dep_id)
        steps[step_id] = {"id": step_id, "sub_question": sub_question, "dependencies": clean_dependencies}

    if set(steps) != set(range(len(steps))):
        return []
    if any(dep_id not in steps for step in steps.values() for dep_id in step["dependencies"]):
        return []
    return topological_plan(steps)


def topological_plan(steps: dict[int, PlanStep]) -> list[PlanStep]:
    children: dict[int, list[int]] = defaultdict(list)
    indegree = {step_id: len(step["dependencies"]) for step_id, step in steps.items()}
    for step_id, step in steps.items():
        for dep_id in step["dependencies"]:
            children[dep_id].append(step_id)

    ready = deque(sorted(step_id for step_id, degree in indegree.items() if degree == 0))
    ordered: list[PlanStep] = []
    while ready:
        step_id = ready.popleft()
        ordered.append(steps[step_id])
        for child_id in sorted(children[step_id]):
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                ready.append(child_id)
    return ordered if len(ordered) == len(steps) else []


def render_plan(plan: list[PlanStep], answers: dict[int, str]) -> str:
    if not plan:
        return "No plan."

    rendered_questions = {step["id"]: step["sub_question"] for step in plan}
    lines: list[str] = []
    for step in plan:
        step_id = step["id"]
        answer = answers.get(step_id, NOT_FOUND)
        lines.append(f"[{step_id}] Question: {rendered_questions[step_id]}\nAnswer: {answer}")
        if answer != NOT_FOUND:
            for future in plan:
                rendered_questions[future["id"]] = rendered_questions[future["id"]].replace(f"<dep:{step_id}>", answer)
    return "\n\n".join(lines)
