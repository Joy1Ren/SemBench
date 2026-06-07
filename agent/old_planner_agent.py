from __future__ import annotations

import os
import pathlib
import re

import yaml
from jinja2 import Template
from openrouter import OpenRouter

_PROMPTS_FILE = pathlib.Path(__file__).parent / "planner_agent.yaml"
with _PROMPTS_FILE.open() as _f:
    _PROMPTS = yaml.safe_load(_f)
PLANNER_AGENT_SYSTEM_PROMPT: str = _PROMPTS["PZ_INSTRUCTION"]

_CODE_BLOCK_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)

_DEFAULT_OPERATORS = {
    "sem_filter": "ds.sem_filter(filter: str, depends_on: list[str]) — Semantically filter rows where `filter` is true.",
    "sem_topk": "ds.sem_topk(search_str: str, k: int, index_name: str) — Return the top-k most semantically relevant rows.",
    "sem_join": "ds.sem_join(other, condition: str, depends_on: list[str]) — Semantically join two datasets where `condition` holds.",
    "sem_add_columns": "ds.sem_add_columns(cols: list[dict], depends_on: list[str]) — Add new LLM-derived columns. Each column dict has 'name', 'type', 'desc'.",
    "filter": "ds.filter(fn: Callable) — Exact (non-semantic) row filter using a lambda.",
    "project": "ds.project(columns: list[str]) — Select a subset of columns.",
    "limit": "ds.limit(n: int) — Keep at most n rows.",
    "count": "ds.count() — Aggregate: count rows.",
    "average": "ds.average() — Aggregate: average a numeric column.",
    "groupby": "ds.groupby(GroupBySig(group_by_fields, agg_funcs, agg_fields)) — Group and aggregate.",
}


def _extract_code(text: str) -> str | None:
    m = _CODE_BLOCK_RE.search(text)
    return m.group(1).strip() if m else None


class OldPlannerAgent:
    def __init__(
        self,
        model_id: str,
        logical_operators: dict[str, str] | None = None,
    ):
        self.model_id = model_id
        self.client = OpenRouter(api_key=os.environ["OPENROUTER_API_KEY"])
        self.system_prompt = Template(PLANNER_AGENT_SYSTEM_PROMPT).render(
            code_opening_tag="```python",
            code_closing_tag="```",
            logical_operators=logical_operators or _DEFAULT_OPERATORS,
        )

    def plan(self, query: str, dataset_description: str) -> str | None:
        """Return a PZ Python script for the given query and dataset description."""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": f"Query: {query}\n\nDataset(s):\n{dataset_description}",
            },
        ]
        response = self.client.chat.send(
            model=self.model_id,
            messages=messages,
        )
        return _extract_code(response.choices[0].message.content)
