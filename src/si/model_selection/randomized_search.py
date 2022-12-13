import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'src/si')
from model_selection.cross_validate import cross_validate
from data.dataset import Dataset
from sklearn.preprocessing import StandardScaler

def randomized_search_cv(model, dataset: Dataset, parameter_grid, scoring = None, cv : int = 3, n_iter : int = 3, test_size : float = 0.2)-> dict[str, list[float]]:
    #validade the parameter grid
    for parameter in parameter_grid:
        if not hasattr(model, parameter):
            raise AttributeError(f"Model {model} does not have parameter {parameter}.")
    scores = []
    #for each combination
    for i in range(n_iter):
        combination = [np.random.choice(a=p,size=1)[0] for p in [*parameter_grid.values()]]
        
        #parameter configuration
        parameters = {}
        #set the parameters
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

if __name__ == '__main__':
    # import dataset
    from data.dataset import Dataset
    from linear_model.logistic_regression import LogisticRegression

    # load and split the dataset
    dataset_ = Dataset.from_random(600, 100, 2)

    # initialize the Logistic Regression model
    knn = LogisticRegression()

    # parameter grid
    parameter_grid_ = {
        'l2_penalty': (1, 10, 10),
        'alpha': (0.001, 0.0001, 100),
        'max_iter': (1000, 2000, 200)
    }

    # cross validate the model
    scores_ = randomized_search_cv(knn, dataset_, parameter_grid=parameter_grid_, cv=3)

    # print the scores
    print(scores_)