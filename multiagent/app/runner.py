import os
import time
from dataclasses import dataclass

from multiagent.config import DEFAULT_MODEL, PRICES
from multiagent.crew import build_pipeline, review_loop


@dataclass
class RunStats:
    model: str
    seconds: float
    prompt_tokens: int
    completion_tokens: int
    requests: int
    report_file: str

    @property
    def cost_usd(self) -> float | None:
        """Estimated cost from PRICES (USD per 1M input/output tokens); None for unknown models."""
        price = PRICES.get(self.model.split("/")[-1])
        if price is None:
            return None
        return (self.prompt_tokens * price[0] + self.completion_tokens * price[1]) / 1_000_000

    def line(self) -> str:
        cost = f"${self.cost_usd:.3f}" if self.cost_usd is not None else "н/д"
        return (
            f"{self.model:<32} {self.seconds:6.1f} с  запросов {self.requests:>3}  "
            f"вход {self.prompt_tokens:>7}  выход {self.completion_tokens:>6}  ~{cost}"
        )


def run(
    model: str | None,
    output_file: str,
    parallel: bool,
    verbose: bool = True,
    request: str | None = None,
    files: list[str] | None = None,
    reviews: int = 1,
) -> RunStats:
    """Run the agents once and return timing, token usage and estimated cost.

    After the report is written, the reviewer and the writer iterate for up to `reviews` rounds.
    """
    model = model or os.getenv("MODEL", DEFAULT_MODEL)
    started = time.perf_counter()

    pipeline = build_pipeline(
        model=model, output_file=output_file, parallel=parallel, verbose=verbose, request=request, files=files
    )
    result = pipeline.crew.kickoff()
    usage = result.token_usage
    if reviews:
        evidence = "\n\n".join(task.raw for task in result.tasks_output[:-1])
        usage.add_usage_metrics(review_loop(model, output_file, evidence, pipeline.facts, request, reviews, verbose))

    return RunStats(
        model=model,
        seconds=time.perf_counter() - started,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        requests=usage.successful_requests,
        report_file=output_file,
    )
