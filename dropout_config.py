from fmre_dynamic_dropout import FMREDynamicDropout
from l2_dynamic_dropout import L2DynamicDropout
from optimal_brain_damage_dropout import OptimalBrainDamageDropout
from gradient_dynamic_dropout import GradientDynamicDropout
from learning_sparse_masks_dropout import LearningSparseMasksDropout
from neuron_shapley_dropout import NeuronShapleyDropout
from taylor_expansion_dropout import TaylorExpansionDropout

dropout_config = {
    'conv1': {
        'method': GradientDynamicDropout,
        'params': {'initial_rate': 0.1, 'max_rate': 0.5}
    },
    'conv2': {
        'method': L2DynamicDropout,
        'params': {'initial_rate': 0.1, 'max_rate': 0.5}
    },
    'fc1': {
        'method': NeuronShapleyDropout,
        'params': {'sample_size': 100, 'sparsity_target': 0.5}
    }
    # Add other layers and methods as needed
}

