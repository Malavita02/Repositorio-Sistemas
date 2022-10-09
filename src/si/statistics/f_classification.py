from scipy import stats
from si.data.dataset import Dataset

def f_classification(dataset):
    data = Dataset.get_classes(dataset)
    F, p = stats.f_oneway(data)
    return F, p