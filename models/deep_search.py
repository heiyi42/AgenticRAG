import argparse
import asyncio
import os

from agenticRAG.agentic_answer import answer_question
from agenticRAG.cli_utils import (
    build_memory_factory,
    is_clear_command,
    is_exit_command,
    print_deep_result,
    reset_thread_after_clear,
)
from agenticRAG.agentic_config import DEFAULT_THREAD_ID

DEEP_QUERY_TIMEOUT_S = int(os.getenv("DEEP_QUERY_TIMEOUT_S", "120"))
DEEP_ENABLE_SUMMARY_MEMORY = os.getenv("DEEP_ENABLE_SUMMARY_MEMORY", "1").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEEP_SUMMARY_TRIGGER_TOKENS = int(os.getenv("DEEP_SUMMARY_TRIGGER_TOKENS", "2000"))
DEEP_MAX_TURNS_BEFORE_SUMMARY = int(os.getenv("DEEP_MAX_TURNS_BEFORE_SUMMARY", "4"))
DEEP_KEEP_RECENT_TURNS = int(os.getenv("DEEP_KEEP_RECENT_TURNS", "1"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep Search CLI")
    parser.add_argument("question", nargs="?", help="要提问的问题")
    parser.add_argument(
        "--thread-id",
        default=DEFAULT_THREAD_ID,
        help="会话ID（相同thread_id会复用会话记忆）",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEEP_QUERY_TIMEOUT_S,
        help="单次问题超时秒数（默认 120）",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="打印子问题/检索细节",
    )
    parser.add_argument(
        "--no-summary-memory",
        action="store_true",
        help="关闭总结式短期记忆",
    )
    return parser.parse_args()


async def _ask(question: str, thread_id: str, timeout_s: int) -> dict:
    return await asyncio.wait_for(
        answer_question(question, thread_id=thread_id),
        timeout=timeout_s,
    )


async def _run() -> None:
    args = _parse_args()
    current_thread_id = args.thread_id
    pending_question = args.question
    use_summary_memory = DEEP_ENABLE_SUMMARY_MEMORY and not args.no_summary_memory
    memory_for_thread = build_memory_factory(
        use_summary_memory=use_summary_memory,
        summary_trigger_tokens=DEEP_SUMMARY_TRIGGER_TOKENS,
        max_turns_before_summary=DEEP_MAX_TURNS_BEFORE_SUMMARY,
        keep_recent_turns=DEEP_KEEP_RECENT_TURNS,
    )
    memory = memory_for_thread(current_thread_id)

    print(
        f"Deep Search 已启动: thread_id={current_thread_id}, timeout={args.timeout}s, "
        f"summary_memory={use_summary_memory}"
    )
    print("输入 quit/exit/q 退出，输入 clear 清空会话记忆。")

    while True:
        if pending_question is not None:
            question = pending_question.strip()
            pending_question = None
        else:
            question = input("\n请输入问题: ").strip()

        if not question:
            continue
        if is_exit_command(question):
            print("已退出。")
            return
        if is_clear_command(question):
            current_thread_id, memory = reset_thread_after_clear(
                base_thread_id=args.thread_id,
                current_thread_id=current_thread_id,
                use_summary_memory=use_summary_memory,
                memory_for_thread=memory_for_thread,
            )
            print(f"会话记忆已清空。新thread_id={current_thread_id}")
            continue

        ask_question = (
            memory.build_augmented_question(question)
            if memory is not None
            else question
        )
        try:
            result = await _ask(ask_question, current_thread_id, args.timeout)
        except asyncio.TimeoutError:
            print(f"\n查询超时（>{args.timeout}s），请缩短问题或稍后重试。")
            continue
        except Exception as e:
            print(f"\n查询失败: {e}")
            continue

        if memory is not None:
            memory.update(question, result.get("final_answer", ""))
        print_deep_result(result, show_details=args.details)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()
