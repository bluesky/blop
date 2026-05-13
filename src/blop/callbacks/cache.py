from bluesky.callbacks import CallbackBase
from event_model import Event, EventDescriptor, RunStart

from blop.protocols import ID_KEY


class OptimizationCache(CallbackBase):
    """
    a small callback that follows the same behavior as logger but collects internal event data to be using in evaluation
    """
    def __init__(self):
        self.run_log = {}
        self.sample_seq = {}

    def start(self, doc: RunStart):
        bs = doc.get("blop_suggestions", None)
        if bs is None:
            return
        self.run_uid = doc.get("uid", None)
        self.ids = [s[ID_KEY] for s in bs]

    def event(self, doc: Event) -> Event:
        sn = doc.get("seq_num", None)
        if sn is None:
            raise KeyError
        id = self.ids[sn - 1]
        self.sample_seq[id] = doc.get("data", {}) | {ID_KEY: id}
        return doc

    def descriptor(self, doc: EventDescriptor): ...

    def stop(self, doc):
        self.run_log[self.run_uid] = self.sample_seq
        self.sample_seq = {}
        self.run_uid = None

    def __getitem__(self, key):
        return self.run_log[key]
