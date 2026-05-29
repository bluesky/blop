from .beamline import beamline_generation_strategy

all_strategies = ["beamline"]


class InvalidStrategyError(Exception): ...


def get_generation_strategy(name):

    if name not in all_strategies:
        raise InvalidStrategyError(f"Invalid strategy '{name}', valid strategies are one of {all_strategies}")

    if name == "beamline":
        return beamline_generation_strategy
