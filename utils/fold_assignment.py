### create a fold iterable for custom fold assignment
class FoldIterable:
    def __init__(self, fold_assignment):
        self.fold_assignment = fold_assignment
        self.n_folds = len(set(fold_assignment))

    def __iter__(self):
        for fold in range(self.n_folds):
            fold_ref = list(set(self.fold_assignment))[fold]
            training_indice = [index for index, fold_tmp in enumerate(self.fold_assignment) if fold_tmp != fold_ref]
            validation_indice = [index for index, fold_tmp in enumerate(self.fold_assignment) if fold_tmp == fold_ref]
            yield training_indice, validation_indice

