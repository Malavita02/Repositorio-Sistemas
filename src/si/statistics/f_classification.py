class F_classification:
    def __init__(self) -> None:
        pass

    def alguma_coisa(self):
        classes = dataset.get_classes()
        groups = [dataset.X[dataset.y == c] for c in classes]
        F, p = stats.f_oneway(*groups)
        return F, p