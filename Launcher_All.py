from typing import Optional
import sys

from botorch.acquisition import ConstrainedMCObjective

from bo.acquisition_functions.acquisition_functions import AcquisitionFunctionType
from bo.bo_loop import OptimizationLoop, EiDecoupledOptimizationloop, EiOptimizationloop, DecoupledEikgOptimizationloop
from bo.acquisition_functions.model.Model import ConstrainedDeoupledGPModelWrapper
from bo.result_utils.result_container import Results
from bo.synthetic_test_functions.synthetic_test_functions import *

device = torch.device("cpu")
dtype = torch.double


def obj_callable(Z: torch.Tensor, X: Optional[torch.Tensor] = None):
    return Z[..., 0]


def constraint_callable_wrapper(constraint_idx):
    def constraint_callable(Z):
        return Z[..., constraint_idx]

    return constraint_callable


if __name__ == "__main__":

    # Set problem and constrains here
    black_box_function = ConstrainedFunc3(noise_std=1e-6, negate=True)
    num_constraints = 3
    filename_pf = 'fun3'
    budget = 160

    seed = int(sys.argv[1])
    print(f'Running seed {seed}')

    # Decoupled cKG
    print('\n Starting dcKG:')
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename=filename_pf + "dckg" + str(seed) + ".pkl")
    loop_dckg = OptimizationLoop(black_box_func=black_box_function,
                            objective=constrained_obj,
                            ei_type=AcquisitionFunctionType.DECOUPLED_CONSTRAINED_KNOWLEDGE_GRADIENT,
                            bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                            performance_type="model",
                            model=model,
                            seed=seed,
                            budget=budget,
                            number_initial_designs=6,
                            results=results,
                            penalty_value=torch.tensor([100.0]))
    loop_dckg.run()

    # EI + KG
    print('\n Starting EI+KG:')
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename=filename_pf + "eikg" + str(seed) + ".pkl")
    loop_eikg = DecoupledEikgOptimizationloop(black_box_func=black_box_function,
                                              objective=constrained_obj,
                                              ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                              bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                              performance_type="model",
                                              model=model,
                                              seed=seed,
                                              budget=budget,
                                              number_initial_designs=6,
                                              results=results,
                                              penalty_value=torch.tensor([100.0]))
    loop_eikg.run()

    # Decoupled EI
    print('\n Starting dEI:')
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename=filename_pf + "dei" + str(seed) + ".pkl")
    loop_dei = EiDecoupledOptimizationloop(black_box_func=black_box_function,
                                           objective=constrained_obj,
                                           ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                           bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                           performance_type="model",
                                           model=model,
                                           seed=seed,
                                           budget=budget,
                                           number_initial_designs=6,
                                           results=results,
                                           penalty_value=torch.tensor([100.0]))
    loop_dei.run()

    # Coupled EI
    print('\n Starting cEI:')
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename=filename_pf + "cei" + str(seed) + ".pkl")
    loop_cei = EiOptimizationloop(black_box_func=black_box_function,
                                  objective=constrained_obj,
                                  ei_type=AcquisitionFunctionType.BOTORCH_CONSTRAINED_EXPECTED_IMPROVEMENT,
                                  bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                  performance_type="model",
                                  model=model,
                                  seed=seed,
                                  budget=int(budget/(num_constraints+1)),
                                  number_initial_designs=6,
                                  results=results,
                                  penalty_value=torch.tensor([100.0]))
    loop_cei.run()

    # Coupled cKG
    print('\n Starting cKG:')
    model = ConstrainedDeoupledGPModelWrapper(num_constraints=num_constraints)
    constrained_obj = ConstrainedMCObjective(
        objective=obj_callable,
        constraints=[constraint_callable_wrapper(idx) for idx in range(1, num_constraints + 1)],
    )
    results = Results(filename=filename_pf + "ckg" + str(seed) + ".pkl")
    loop_ckg = EiOptimizationloop(black_box_func=black_box_function,
                                  objective=constrained_obj,
                                  ei_type=AcquisitionFunctionType.MC_CONSTRAINED_KNOWLEDGE_GRADIENT,
                                  bounds=torch.tensor([[0.0, 0.0], [1.0, 1.0]], device=device, dtype=dtype),
                                  performance_type="model",
                                  model=model,
                                  seed=seed,
                                  budget=int(budget/(num_constraints+1)),
                                  number_initial_designs=6,
                                  results=results,
                                  penalty_value=torch.tensor([100.0]))
    loop_ckg.run()
