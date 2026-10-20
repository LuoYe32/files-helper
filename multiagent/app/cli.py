import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from multiagent.app.runner import run
from multiagent.config import DATA_DIR


def interactive(args: argparse.Namespace) -> None:
    """Prompt for questions in a loop; each answer is saved to output/answer_N.md."""
    print("Задавайте вопросы по файлам из data/. Пустая строка или Ctrl+D — выход.")
    number = 1
    while True:
        try:
            request = input("\nВаш запрос: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not request:
            break
        output = f"output/answer_{number}.md"
        stats = run(
            args.model,
            output,
            args.parallel,
            verbose=not args.quiet,
            request=request,
            files=args.files,
            reviews=args.reviews,
        )
        print(f"\n{Path(output).read_text(encoding='utf-8')}\n\n{stats.line()}")
        number += 1


def main() -> None:
    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("Не задан ANTHROPIC_API_KEY: впишите ключ в файл .env")

    parser = argparse.ArgumentParser(description="Мультиагентный анализ файлов из data/")
    parser.add_argument("--model", help="например anthropic/claude-sonnet-5-5 (по умолчанию MODEL из .env)")
    parser.add_argument("--parallel", action="store_true", help="аналитики работают одновременно")
    parser.add_argument("--ask", help="один вопрос по файлам; без него строится общий отчёт")
    parser.add_argument("-i", "--interactive", action="store_true", help="диалог: вводите вопросы по одному")
    parser.add_argument("--files", nargs="+", metavar="FILE", help="обработать только эти файлы из data/ (без маршрутизатора)")
    parser.add_argument("--reviews", type=int, default=1, help="кругов рецензии и доработки (0 — без рецензента)")
    parser.add_argument("--quiet", action="store_true", help="не печатать подробный лог агентов")
    parser.add_argument("--output", default="output/report.md")
    args = parser.parse_args()

    missing = [f for f in args.files or [] if not (DATA_DIR / f).is_file()]
    if missing:
        sys.exit(f"Нет таких файлов в data/: {missing}")

    if args.interactive:
        interactive(args)
        return

    stats = run(
        args.model,
        args.output,
        args.parallel,
        verbose=not args.quiet,
        request=args.ask,
        files=args.files,
        reviews=args.reviews,
    )
    print(f"\n=== Готово: {stats.report_file} ===")
    print(stats.line())


if __name__ == "__main__":
    main()
