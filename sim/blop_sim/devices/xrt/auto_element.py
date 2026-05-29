import time
from collections import OrderedDict

from bluesky.protocols import Readable, Triggerable

from blop.protocols import MovableHasName

from ...backends import XRTBackend, build_histRGB

primitives = {int, float, bool, str, type(None)}
aliases = "xyzwhijk"
l_index = {k: i for i, k in enumerate(aliases)}
known_vars = {"_center", "_roll", "_yaw", "_pitch", "R"}


class InferredVariable(MovableHasName, Readable):
    def __init__(self, name: str, element: object, PV: str):
        self.base_object = element
        self.root = name
        self.PV = PV
        self.member_route = PV.split(":")
        if self.val is not None and not (isinstance(self.val, str) and self.val == "auto"):
            self.type = type(self.val)
        else:
            self.type = float
            print(f"""created inferred variable {self.name} of float type as value is None or auto. 
                Be careful when setting this variable as the type is guessed as float by default.""")

    @property
    def val(self):
        if len(self.member_route) == 1:
            return getattr(self.base_object, self.member_route[0])

        submember = getattr(self.base_object, self.member_route[0])
        if type(submember) is list:  # list branch
            return submember[l_index[self.member_route[1]]]
        return submember[self.member_route[1]]  # dict branch

    @val.setter
    def val(self, value):
        if self.member_route is None:
            raise ValueError("the variable has yet to be inferred")
        if type(value) is not self.type and value != "auto":
            raise ValueError("attempting to set an inferred variable to a different inferred type outside of auto")

        if len(self.member_route) == 1:
            setattr(self.base_object, self.member_route[0], value)
            return

        submember = getattr(self.base_object, self.member_route[0])
        if type(submember) is list:  # list branch
            submember[l_index[self.member_route[1]]] = value
        elif type(submember) is dict:  # dict branch
            submember[self.member_route[1]] = value
        else:
            raise ValueError("member route is broken or type is not primitive/inferrable")

    # has name interface
    @property
    def name(self):
        return f"{self.root}:{self.PV}"

    def __repr__(self):
        return f"<InferredVariable::{self.name}={self.type}:{self.val}>"

    # movable interface
    def set(self, value):
        self.val = value

    # readable interface
    def read(self):
        return {self.name: {"value": self.val, "timestamp": -1}}

    def describe(self):
        return {self.name: {"source": f"{self.root}:inferred", "dtype": str(self.type), "shape": []}}


class InferredDetector(Readable, Triggerable):  # this is by element
    def __init__(self, beamLine: XRTBackend, name: str, shape: tuple[int, int], primary: bool = False):
        self._beamline = beamLine
        self._name = name
        self.shape = shape
        if primary:
            self.set_primary()

    @property
    def primary(self):
        return self is self._beamline.target

    def set_primary(self):
        self._beamline.target = self

    def trigger(self):
        if self.primary:
            self._beamline.generate_beam()

    def read(self) -> OrderedDict:
        beam = self._beamline[self._name][:1]  # for now, only 1
        result = OrderedDict()
        for _face in beam:
            hist, _, _ = build_histRGB(beam, isScreen=True, shape=self.shape)
            result[self._name] = {"value": hist, "timestamp": time.time()}
        return result

    def describe(self):
        return OrderedDict([(self._name, {"source": self._name, "dtype": "ndarray", "shape": list(self.shape)})])

    @property
    def name(self):
        return self._name


def element_to_variables(element, name: str, filter_for: set = None) -> dict[str, InferredVariable]:
    lib = {}
    for key, item in vars(element).items():
        if filter_for is not None and key not in filter_for:
            continue
        member = type(item)
        if member in primitives:
            inferred = InferredVariable(name=name, element=element, PV=key)
            lib[inferred.name] = inferred
        elif member is list:
            for x in range(len(item)):
                inferred = InferredVariable(name=name, element=element, PV=f"{key}:{aliases[x]}")
                lib[inferred.name] = inferred
        elif member is dict:
            for x in item.keys():
                inferred = InferredVariable(name=name, element=element, PV=f"{key}:{x}")
                lib[inferred.name] = inferred
    return lib


def infer_variables(beamLine: XRTBackend, filter_for: set[str] = known_vars):
    variables = {}
    for name, element in beamLine.elements.items():
        variables[name] = element_to_variables(element, name, filter_for=filter_for)
    return variables


def infer_detectors(beamLine: XRTBackend):
    dets = {}
    for name in beamLine.elements.keys():
        dets[name] = InferredDetector(beamLine, name, shape=[256, 256], primary=("Sample" | "Screen" in name))
    return dets
