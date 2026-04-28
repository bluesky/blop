from botorch.models.transforms.input import Normalize
from botorch.models.transforms.outcome import Standardize

from ax.generation_strategy.generation_node import GenerationNode
from ax.generation_strategy.generator_spec import GeneratorSpec
from ax.generation_strategy.transition_criterion import MinTrials
from ax.adapter.registry import Generators
from ax.generators.torch.botorch_modular.surrogate import ModelConfig, SurrogateSpec
from botorch.acquisition.logei import qLogNoisyExpectedImprovement

from blop.bayesian.models import LatentGP


sobol_node = GenerationNode(
            name="Sobol",
            generator_specs=[
                GeneratorSpec(generator_enum=Generators.SOBOL, model_kwargs={"seed": 0}),
            ],
            transition_criteria=[
                MinTrials(
                    threshold=16,
                    transition_to="LatentGP",
                    use_all_trials_in_exp=True,
                ),
            ],
        )

latent_gp_node = GenerationNode(
            name="LatentGP",
            generator_specs=[
                GeneratorSpec(
                    generator_enum=Generators.BOTORCH_MODULAR,
                    model_kwargs={
                        "surrogate_spec": SurrogateSpec(
                            model_configs=[
                                ModelConfig(
                                    botorch_model_class=LatentGP,
                                    input_transform_classes=[Normalize],
                                    # model_options={"skew_dims": True},
                                    outcome_transform_classes=[Standardize],
                                ),
                            ],
                        ),
                        "botorch_acqf_class": qLogNoisyExpectedImprovement,
                        "acquisition_options": {},
                    },
                    model_gen_kwargs={
                        "optimizer_kwargs": {
                            "num_restarts": 10,
                            "sequential": True,
                        },
                    },
                ),
            ],
        )
