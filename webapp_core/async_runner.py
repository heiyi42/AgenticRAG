from __future__ import annotations

import asyncio
from threading import Event, Lock, Thread


class AsyncLoopRunner:
    """Run all async tasks on one dedicated event loop thread."""

    def __init__(self) -> None:
        self._thread: Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = Event()
        self._lock = Lock()

    def _bootstrap(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        loop.run_forever()

        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
        loop.close()

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._ready.clear()
            self._thread = Thread(
                target=self._bootstrap,
                name="webapp-async-loop",
                daemon=True,
            )
            self._thread.start()
        if not self._ready.wait(timeout=5):
            raise RuntimeError("异步执行器启动超时")

    def run(self, coro):
        self.start()
        if self._loop is None:
            raise RuntimeError("异步事件循环不可用")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def submit(self, coro):
        self.start()
        if self._loop is None:
            raise RuntimeError("异步事件循环不可用")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def stop(self) -> None:
        with self._lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
            self._ready.clear()
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=1)


async_runner = AsyncLoopRunner()


def run_async(coro):
    return async_runner.run(coro)


def submit_async(coro):
    return async_runner.submit(coro)
