import numpy as np
import sys
from si.model_selection.split import train_test_split
sys.path.insert(0, 'src/si')
from data.dataset import Dataset

def cross_validate(model, dataset: Dataset, scoring, cv : int = 3, test_size: float = 0.2)-> dict [str, list[float]]:
    scores = {
        "seeds": [],
        "train": [],
        "test": []
    }
    
    for i in range(cv):
        random_state = np.random.randint(0, 1000)
    
        scores["seeds"].append(random_state)

        train, test = train_test_split()
        model.fit(train)

        if scoring is None:
            scores["train"].append(model.score(train))
            scores["test"].append(model.score(test))
        
        else:
            y_train = train.y
            y_test = test.y
            scores["train"].append(scoring(y_train, model.score(train)))
            scores["test"].append(scoring(y_test, model.score(train)))
        
        return scores

#falta criar train e test e testar