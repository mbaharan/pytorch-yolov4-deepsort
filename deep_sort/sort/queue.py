import janus

class Queue:
    def __init__(self, budget=0):
        # The depth of queue. 0 means infinite queue.
        self._queue = janus.Queue(budget)

    def sync_put_nowait(self, item):
        self._queue.sync_q.put_nowait(item)

    def sync_put(self, item):
        self._queue.sync_q.put(item)

    def sync_get(self):
        return self._queue.sync_q.get()

    def async_task_done(self):
        self._queue.async_q.task_done()

    def async_put_nowait(self, item):
        self._queue.async_q.put_nowait(item)

    async def async_put(self, item):
        await self._queue.async_q.put(item)

    async def async_get(self):
        return await self._queue.async_q.get()
    
    async def async_wait(self):
        await self._queue.async_q.join()

'''
import asyncio

class Queue:
    def __init__(self):
        self._loop = asyncio._get_running_loop()
        self._queue = asyncio.Queue()

    def sync_put_nowait(self, item):
        self._loop.call_soon_threadsafe(self._queue.put_nowait, item)

    def sync_put(self, item):
        asyncio.run_coroutine_threadsafe(self._queue.put(item), self._loop).result()

    def sync_get(self):
        return asyncio.run_coroutine_threadsafe(self._queue.get(), self._loop).result()

    def async_put_nowait(self, item):
        self._queue.put_nowait(item)

    async def async_put(self, item):
        await self._queue.put(item)

    async def async_get(self):
        return await self._queue.get()
'''