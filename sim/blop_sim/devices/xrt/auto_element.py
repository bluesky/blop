import time
from collections import OrderedDict
from collections.abc import Callable

import numpy as np
from bluesky.protocols import Readable, Status, Triggerable

from blop.protocols import MovableHasName

from ...backends import XRTBackend, build_histRGB

primitives = (int, float, bool, str, type(None))
data_map = {int: "integer", float: "number", bool: "boolean", str: "string"}
aliases = "xyzwhijk"
l_index = {k: i for i, k in enumerate(aliases)}
known_variables = {
    "center",
    "opening",
    "alpha",
    "extraPitch",
    "extraRoll",
    "extraYaw",
    "roll",
    "pitch",
    "yaw",
    "R",
    "r",
    "K",
    "n",
    "cryst1roll",
    "cryst1pitch",
    "cryst1yaw",
    "cryst2roll",
    "cryst2pitch",
    "cryst2finePitch",
    "cryst2yaw",
}


class always_good_stat(Status):
    def add_callback(self, callback: Callable[["Status"], None]) -> None:
        """Add a callback function to be called upon completion.

        The function must take the status as an argument.

        If the Status object is done when the function is added, it should be
        called immediately.
        """
        callback(self)

    def exception(self, timeout: float | None = 0.0) -> BaseException | None:
        return FileNotFoundError()

    def done(self) -> bool:
        """If done return True, otherwise return False."""
        return True

    def success(self) -> bool:
        """If done return whether the operation was successful."""
        return True


class InferredVariable(MovableHasName, Readable):
    def __init__(self, name: str, element: object, PV: str, root: XRTBackend | None = None):
        self.base_object = element
        self.base = name
        self.root = root
        self.PV = PV
        self.member_route = PV.split(":")
        self.parent = None
        self.alias = None
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
        # if type(submember) is list:  # list branch
        return submember[l_index[self.member_route[1]]]
        # return submember[self.member_route[1]]  # dict branch

    @val.setter
    def val(self, value):
        if self.member_route is None:
            raise ValueError("the variable has yet to be inferred")
        if not isinstance(value, self.type) and value != "auto":
            raise ValueError("attempting to set an inferred variable to a different inferred type outside of auto")
        # setattr(self.base_object, self.member_route[0], value)

        if self.root is not None:
            self.root.invalidate(self.base)

        if len(self.member_route) == 1:
            setattr(self.base_object, self.member_route[0], value)
            return

        submember = getattr(self.base_object, self.member_route[0])
        # if isinstance(submember, list):  # list branch
        submember[l_index[self.member_route[1]]] = value
        # elif type(submember) is dict:  # dict branch
        #     submember[self.member_route[1]] = value
        # else:
        #     raise ValueError("member route is broken or type is not primitive/inferrable")

    # has name interface
    @property
    def name(self):
        if self.alias is not None:
            return str(self.alias)
        root = "" if self.root is None else self.root.name + ":"
        return f"{root}{self.base}:{self.PV}"

    def __repr__(self) -> str:
        return f"<InferredVariable::{self.name}={self.type}:{self.val}>"

    # movable interface
    def set(self, value):
        self.val = value
        return always_good_stat()

    # readable interface
    def read(self) -> OrderedDict[str, dict]:
        return OrderedDict([(self.name, {"value": self.val, "timestamp": -1})])

    def describe(self) -> OrderedDict[str, dict]:
        return OrderedDict([(self.name, {"source": f"{self.base}:inferred", "dtype": data_map[self.type], "shape": []})])


class InferredDetector(Readable, Triggerable):  # this is by element
    def __init__(self, beamLine: XRTBackend, name: str, shape: tuple[int, int], primary: bool = False):
        self._beamline = beamLine
        self._name = name
        self.shape = shape
        self.parent = None
        if primary:
            self.set_primary()

    @property
    def primary(self):
        return self is self._beamline.target

    def set_primary(self):
        self._beamline.target = self

    def trigger(self) -> Status:
        if self.primary:
            self._beamline.generate_beam()
        return always_good_stat()

    def read(self) -> OrderedDict:
        beam = self._beamline[self._name]
        # print(beam)
        result = OrderedDict()
        for face in beam[:1]:  # for now, only 1
            hist, _, _ = build_histRGB(face, face, isScreen=True, shape=self.shape)
            data = {"value": hist, "timestamp": time.time()}
            result[self._name] = data
        return result

    def describe(self) -> OrderedDict:
        return OrderedDict(
            [
                (
                    self._name,
                    {
                        "source": self._beamline.name,
                        "shape": [*self.shape],
                        "dtype": "array",
                        "dtype_numpy": np.dtype(np.float64).str,
                    },
                )
            ]
        )

    @property
    def name(self):
        return self._name


def element_to_variables(
    element, name: str, filter_for: set = None, root: XRTBackend | None = None
) -> dict[str, InferredVariable]:
    lib = {}
    for key, item in vars(type(element)).items():
        if type(item) is not property:
            continue
        if filter_for is not None and key not in filter_for:
            continue

        try:
            val = getattr(element, key)
        except Exception:
            continue
        if isinstance(val, primitives):
            inferred = InferredVariable(name=name, element=element, PV=key, root=root)
            lib[inferred.PV] = inferred
        elif isinstance(val, list):
            for x in range(len(val)):
                inferred = InferredVariable(name=name, element=element, PV=f"{key}:{aliases[x]}", root=root)
                lib[inferred.PV] = inferred
        # elif member is dict:
        #     print(f"inferring dict {key} for element {name}")
        #     for x in val.keys():
        #         inferred = InferredVariable(name=name, element=element, PV=f"{key}:{x}")
        #         lib[inferred.name] = inferred
    return lib


def infer_variables(beamLine: XRTBackend, filter_for: set[str] = known_variables) -> dict[str, dict[str, InferredVariable]]:
    variables = {}
    for name, element in beamLine.elements.items():
        eles = element_to_variables(element, name, filter_for=filter_for, root=beamLine)
        if eles:
            variables[name] = eles
    return variables


def infer_detectors(beamLine: XRTBackend) -> dict[str, InferredDetector]:
    dets = {}
    for name in beamLine.elements.keys():
        dets[name] = InferredDetector(beamLine, name, shape=[128, 128], primary=("Screen" in name))
    return dets
