from blop.protocols import MovableHasName, Readable

primitives = {int, float, bool, str, type(None)}
aliases = "xyzwhijk"
l_index = {k: i for i, k in enumerate(aliases)}


class InferredVariable(MovableHasName, Readable):
    def __init__(self, name: str, element: object, PV: str):
        self.base_object = element
        self.root = name
        self.PV = PV
        self.member_route = PV.split(":")
        self.type = type(self.val) if self.val != "auto" and self.val is not None else float
        if self.val == "auto" or self.val is None:
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
