# dropout_factory.py
from fmre_dynamic_dropout import FMREDynamicDropout
from l2_dynamic_dropout import L2DynamicDropout
from optimal_brain_damage_dropout import OptimalBrainDamageDropout
from gradient_dynamic_dropout import GradientDynamicDropout
from learning_sparse_masks_dropout import LearningSparseMasksDropout
from neuron_shapley_dropout import NeuronShapleyDropout
from taylor_expansion_dropout import TaylorExpansionDropout

def get_dropout_method(method_name):
    """
    Maps method name strings to dropout class types.
    """
    methods = {
        'gradient_dynamic_dropout': GradientDynamicDropout,
        'l2_dynamic_dropout': L2DynamicDropout,
        'optimal_brain_damage_dropout': OptimalBrainDamageDropout,
        'fmre_dynamic_dropout': FMREDynamicDropout,
        'learning_sparse_masks_dropout': LearningSparseMasksDropout,
        'neuron_shapley_dropout': NeuronShapleyDropout,
        'taylor_expansion_dropout': TaylorExpansionDropout
    }
    return methods[method_name]  # Return the class, not an instance

