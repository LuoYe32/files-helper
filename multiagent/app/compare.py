import os
import sys

from dotenv import load_dotenv

from multiagent.app.runner import run

DEFAULT_MODELS = ["anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"]


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Не задан ANTHROPIC_API_KEY: впишите ключ в файл .env")

    results = []
    for model in sys.argv[1:] or DEFAULT_MODELS:
        print(f"--- {model} ---", flush=True)
        output = f"output/report_{model.split('/')[-1]}.md"
        results.append(run(model, output, parallel=True, verbose=False))

    print("\n=== Сравнение ===")
    for stats in results:
        print(stats.line(), "->", stats.report_file)


if __name__ == "__main__":
    main()
