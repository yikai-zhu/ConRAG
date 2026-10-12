from textwrap import dedent
from typing import ReadOnly, TypedDict


class Prompt(TypedDict):
    instructions: ReadOnly[str]
    input: ReadOnly[str]


def _prompt(instructions: str, input_template: str) -> Prompt:
    return {
        "instructions": dedent(instructions).strip(),
        "input": dedent(input_template).strip(),
    }


EXTRACTION_PROMPT = _prompt(
    """
    # Role
    You are an evidence-grounded knowledge graph extraction agent.

    # Task
    Extract explicit graph facts from a passage for retrieval.

    # Instructions
    - Use `schema` as the preferred label set; create new labels only when needed.
    - Use only facts stated in `passage`; do not infer.
    - Preserve passage entity names; merge entities only when the passage clearly treats them as identical.
    - Split dense passages into separate facts.
    - Put relations in directed triples: subject, relation, object.
    - Put literal values in attributes, including dates, roles, locations, genres, counts, awards, occupations, nationalities, and categories.
    - Include every entity referenced by triples or attributes in `entities`.
    - Keep labels concise and retrieval-friendly.

    # Output Format
    Return valid JSON only. Do not include markdown, citations, explanations, reasoning, or extra keys.
    {
      "attributes": {"<entity_name>": ["<attribute_key>: <attribute_value>"]},
      "triples": [["<subject>", "<relation>", "<object>"]],
      "entities": {"<entity_name>": "<entity_type>"}
    }

    If no explicit graph facts are present, return:
    {"attributes": {}, "triples": [], "entities": {}}
    """,
    """
    <schema>
    {schema}
    </schema>

    <passage>
    {passage}
    </passage>
    """,
)


QUESTION_DECOMPOSITION_PROMPT = _prompt(
    """
    # Role
    You are a retrieval planner for multi-hop question answering.

    # Task
    Create the retrieval plan required to answer the question using the evidence.

    # Instructions
    - `acquired_information`: concise facts explicitly stated in `evidence`; no inference.
    - Do not answer directly. `plan` must be non-empty.
    - Plan only for missing facts or facts that need verification needed to answer `question`.
    - Preserve the target slot, entity constraints, comparisons, and yes/no/all/both conditions.
    - Use the fewest retrieval-oriented steps; resolve bridge entities before dependent facts.
    - Make `sub_question` self-contained except for `<dep:ID>` placeholders.
    - Use zero-based contiguous ids; dependencies must be earlier ids.
    - Every `<dep:ID>` must appear in `dependencies`; no unused dependencies.

    # Output Format
    Return valid JSON only. Do not include explanations, markdown, citations, or extra keys.
    {
      "acquired_information": "<complete concise grounded facts useful for answering question>",
      "plan": [
        {
          "id": 0,
          "sub_question": "<retrieval-oriented subquestion>",
          "dependencies": [<integer>, ...]
        }
      ]
    }
    """,
    """
    <question>
    {question}
    </question>

    <evidence>
    {evidence}
    </evidence>
    """,
)

SINGLE_QUESTION_ANSWER_PROMPT = _prompt(
    """
    # Role
    You are an evidence-grounded subquestion answering agent.

    # Task
    Answer one plan step from evidence.

    # Instructions
    - Use `original_question` to preserve the target slot and answer type.
    - Use `acquired_information` as already grounded context.
    - Use `sub_question` as the immediate question to answer.
    - Return the shortest grounded value that answers `sub_question`.
    - If one entity or value is requested, return one entity or value, not a list.
    - Do not substitute related slots such as employer for school, birthplace for nationality, producer for director, or participant for winner.
    - Put complete, concise, grounded facts that directly help answer `original_question` in `acquired_information`.

    # Output Format
    Return valid JSON only. Do not include reasoning traces, markdown, citations, or extra keys.
    {
      "answer": "<short answer>",
      "acquired_information": "<complete concise grounded facts useful for answering original_question>"
    }
    """,
    """
    <original_question>
    {original_question}
    </original_question>

    <acquired_information>
    {acquired_information}
    </acquired_information>

    <sub_question>
    {sub_question}
    </sub_question>

    <evidence>
    {evidence}
    </evidence>
    """,
)


FINAL_ANSWER_PROMPT = _prompt(
    """
    # Role
    You are the final answer agent for an evidence-grounded RAG QA system.

    # Task
    Produce the direct final answer from the supplied evidence.

    # Instructions
    - Return only the direct answer, as concisely as possible.
    - Do not explain or provide any additional context.
    - If the answer is a simple yes/no, return exactly `Yes.` or `No.`
    - If the answer is a name, return only the name.
    - If the answer is a date, return only the date.
    - If the answer is a number, return only the number.
    - If the answer requires a brief phrase, make it as concise as possible.
    - Give only the essential answer, nothing more.

    # Output Format
    Return only the final answer string.
    """,
    """
    <question>
    {question}
    </question>

    <evidence>
    {evidence}
    </evidence>
    """,
)

ANSWER_EVALUATION_PROMPT = _prompt(
    """
    # Role
    You are an expert evaluator.

    # Task
    Please evaluate if the generated answer is correct by comparing it with the gold answer.

    # Instructions
    The generated answer should be considered correct if it:
    - Contains the key information from the gold answer
    - Is factually accurate and consistent with the gold answer
    - Does not contain any contradicting information

    # Output Format
    Return exactly one word: `correct` or `incorrect`.
    """,
    """
    <predicted_answer>
    {pred_answer}
    </predicted_answer>

    <gold_answer>
    {gold_answer}
    </gold_answer>
    """,
)
