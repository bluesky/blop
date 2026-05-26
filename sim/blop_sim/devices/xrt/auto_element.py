from blop.protocols import MovableHasName, Readable

primitives = {int, float, bool, str}
aliases = "xyzwhijk"
l_index = {k: i for i, k in enumerate(aliases)}


class InferredVariable(MovableHasName, Readable):
    def __init__(self, name: str, element: object, PV: str):
        self.base_object = element
        self.root = name
        self.PV = PV
        self.member_route = PV.split(":")
        self.type = type(self._read()) | int

    def set(self, value):
        if self.member_route is None:
            raise ValueError("the variable has yet to be inferred")
        if type(value) is not self.type and value != "auto":
            raise ValueError("attempting to set an inferred variable to a different inferred type outside of auto")

        if len(self.member_route) == 1:
            setattr(self.base_object, self.member_route[0], value)
            return

        submember = getattr(self.base_object, self.member_route[0])
        if type(submember) is list:  # list branch
            submember[l_index[self.member_route]] = value
        elif type(submember) is dict:  # dict branch
            submember[self.member_route] = value
        else:
            raise ValueError("member route is broken or type is not primitive/inferrable")

    def _read(self):
        if len(self.member_route) == 1:
            return getattr(self.base_object, self.member_route[0])

        submember = getattr(self.base_object, self.member_route[0])
        if type(submember) is list:  # list branch
            return submember[l_index[self.member_route[1]]]
        return submember[self.member_route[1]]  # dict branch

    def read(self):
        return {self.name: {"value": self._read(), "timestamp": -1}}

    @property
    def name(self):
        return f"{self.root}:{self.PV}"

    def describe(self):
        return {self.name: {"source": f"{self.root}:inferred", "dtype": str(self.type), "shape": []}}


def element_to_variables(element, name) -> dict[str, InferredVariable]:
    lib = {}
    for key, item in vars(element).items():
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
