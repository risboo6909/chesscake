import queue
import asyncio
import threading
import random
import string
import ray

from result import Err, Ok, Result
from cv_board import recognize_board, from_file_object, from_path
from nn_pieces import recognize_pieces
from io import BytesIO
from queue import Queue

MAX_QUEUE_SIZE = 10


class Task(object):
    def __init__(self, turn: str, bottom_left: str, board: BytesIO, user_id: str):
        self.turn = turn
        self.bottom_left = bottom_left
        self.board = board
        self.user_id = user_id

        # generate random ascii string
        self.ticket = "".join(random.choice(string.ascii_letters) for i in range(10))
        # will be filled up automatically
        self.event = None
        # task result will be filled up automatically
        self.result = None


class Event_ts(asyncio.Event):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

    def set(self):
        self._loop.call_soon_threadsafe(super().set)

    def clear(self):
        self._loop.call_soon_threadsafe(super().clear)


@ray.remote
def recognize(models, board, turn, bottom_left) -> Result:
    cropped_squares = recognize_board(from_file_object(board), debug=False)
    if isinstance(cropped_squares, Err):
        return Err("Couldn't recognize board :(")
    board = recognize_pieces(
        models, cropped_squares.value, turn=turn, bottom_left=bottom_left
    )
    return Ok(board)


class Result(object):
    def __init__(self, success: bool, place: int):
        self.success = success
        self.place = place
        self.result = None
        self.event = Event_ts()
        self.event.clear()


class TaskManager(object):
    def __init__(self, models, stats):
        self.tasks = Queue(maxsize=MAX_QUEUE_SIZE)
        self.models = models
        self.stats = stats
        self.user_ids = set()

    def job_requested(self, user_id):
        return user_id in self.user_ids

    def get_queue_size(self):
        return self.tasks.qsize()

    def start(self):
        worker = threading.Thread(target=lambda: self.worker())
        worker.start()

    def worker(self):
        while True:
            try:
                self.stats.queue_size = self.get_queue_size()
                task = self.dequeue_task()
                print("Processing task '{}'".format(task.ticket))
                future = recognize.remote(
                    self.models, task.board, task.turn, task.bottom_left
                )
                task.result = ray.get([future])
                task.event.set()
                print("Task '{}' processed".format(task.ticket))
            except Exception as e:
                print("Failed to complete task: {}, reason: {}".format(task.ticket, e))
            finally:
                self.user_ids.remove(task.user_id)

    def enqueue_task(self, task) -> Result:
        try:
            self.stats.queue_size = self.get_queue_size()
            self.user_ids.add(task.user_id)
            self.tasks.put_nowait(task)
        except queue.Full:
            self.user_ids.remove(task.user_id)
            return Result(False, 0)
        else:
            res = Result(True, self.tasks.qsize())
            task.event = res.event
            return res

    def dequeue_task(self) -> Task:
        return self.tasks.get()
