from fmre_dynamic_dropout import FMREDynamicDropout
from l2_dynamic_dropout import L2DynamicDropout
from optimal_brain_damage_dropout import OptimalBrainDamageDropout
from gradient_dynamic_dropout import GradientDynamicDropout
from learning_sparse_masks_dropout import LearningSparseMasksDropout
from neuron_shapley_dropout import NeuronShapleyDropout
from taylor_expansion_dropout import TaylorExpansionDropout

def get_dropout_method(method_name, **kwargs):
    """
    Factory function to retrieve dropout classes based on the method name.
    Additional parameters for the dropout classes can be passed using kwargs.

    :param method_name: Name of the dropout method as a string.
    :param kwargs: Additional keyword arguments for initializing the dropout class.
    :return: An instance of the dropout class.
    """
    if method_name == 'gradient_dynamic_dropout':
        return GradientDynamicDropout(**kwargs)
    elif method_name == 'l2_dynamic_dropout':
        return L2DynamicDropout(**kwargs)
    elif method_name == 'optimal_brain_damage_dropout':
        return OptimalBrainDamageDropout(**kwargs)
    elif method_name == 'fmre_dynamic_dropout':
        return FMREDynamicDropout(**kwargs)
    elif method_name == 'learning_sparse_masks_dropout':
        return LearningSparseMasksDropout(**kwargs)
    elif method_name == 'neuron_shapley_dropout':
        return NeuronShapleyDropout(**kwargs)
    elif method_name == 'taylor_expansion_dropout':
        return TaylorExpansionDropout(**kwargs)
    else:
        raise ValueError(f"Unknown dropout method: {method_name}")


