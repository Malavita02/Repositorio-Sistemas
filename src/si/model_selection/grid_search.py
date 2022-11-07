import numpy as np
import sys
from si.model_selection.cross_validate import cross_validate
from si.model_selection.split import train_test_split
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

def grid_search(model, dataset: Dataset, parameter_grid, scoring, cv : int = 3, test_size : float = 0.2)-> Dict[str, List[float]]:
    #validade the parameter grid
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
    scores = []
    #for each combination
    for combination in itertools.product(*parameter_grid.values()):
        #parameter configuration
        parameters = {}
        #ser the parameters
        for parameter, value in zip(parameter_grid.keys(), combination):
            setattr(model, parameter, value)
            parameters[parameter] = value
        #cross validade the model
        score = cross_validate(model= model, dataset= dataset, scoring=scoring, cv=cv, test_size=test_size)

        #add the parameter configuration
        score["parameters"] = parameters
        
        #add the score
        scores.append(score)
    return scores