from ax.generation_strategy.generation_strategy import GenerationStrategy

from .nodes import latent_gp_node, sobol_node

beamline_generation_strategy = GenerationStrategy(name="Custom Generation Strategy", nodes=[sobol_node, latent_gp_node])
