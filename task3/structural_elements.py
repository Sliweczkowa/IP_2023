import numpy as np

class StructuralElement:
    def __init__(self, array: np.ndarray, origin: (int, int)):
        if origin[0] < 0 or origin[0] >= len(array) or origin[1] < 0 or origin[1] >= len(array[0]):
            raise IndexError
        else:
            self.array = array
            self.origin = origin
    def __len__(self):
        if isinstance(self.array, tuple):
            return len(self.array)
        else:
            return 1


I = StructuralElement(np.ones((1, 2), dtype=bool), (0, 0))
II = StructuralElement(np.ones((2, 1), dtype=bool), (0, 0))
III = StructuralElement(np.ones((3, 3), dtype=bool), (1, 1))
IV = StructuralElement(np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool), (1, 1))
V = StructuralElement(np.array([[1, 1], [1, 0]], dtype=bool), (0, 0))
VI = StructuralElement(np.array([[0, 1], [1, 0]], dtype=bool), (0, 0))
VII = StructuralElement(np.ones((1, 3), dtype=bool), (0, 1))
VIII = StructuralElement(np.array([[1, 0, 1]], dtype=bool), (0, 1))
IX = StructuralElement(np.array([[1, 1], [1, 0]], dtype=bool), (0, 1))
X = StructuralElement(np.array([[1, 1], [1, 0]], dtype=bool), (1, 0))
XI = (
    StructuralElement(np.array([[1, 2, 2], [1, 0, 2], [1, 2, 2]], dtype=int), (1, 1)),
    StructuralElement(np.array([[1, 1, 1], [2, 0, 2], [2, 2, 2]], dtype=int), (1, 1)),
    StructuralElement(np.array([[2, 2, 1], [2, 0, 1], [2, 2, 1]], dtype=int), (1, 1)),
    StructuralElement(np.array([[2, 2, 2], [2, 0, 2], [1, 1, 1]], dtype=int), (1, 1)))
XII = (
    StructuralElement(np.array([[0, 0, 0], [2, 1, 2], [1, 1, 1]], dtype=int), (1, 1)),
    StructuralElement(np.array([[2, 0, 0], [1, 1, 0], [1, 1, 2]], dtype=int), (1, 1)),
    StructuralElement(np.array([[1, 2, 0], [1, 1, 0], [1, 2, 0]], dtype=int), (1, 1)),
    StructuralElement(np.array([[1, 1, 2], [1, 1, 0], [2, 0, 0]], dtype=int), (1, 1)),
    StructuralElement(np.array([[1, 1, 1], [2, 1, 2], [0, 0, 0]], dtype=int), (1, 1)),
    StructuralElement(np.array([[2, 1, 1], [0, 1, 1], [0, 0, 2]], dtype=int), (1, 1)),
    StructuralElement(np.array([[0, 2, 1], [0, 1, 1], [0, 2, 1]], dtype=int), (1, 1)),
    StructuralElement(np.array([[0, 0, 2], [0, 1, 1], [2, 1, 1]], dtype=int), (1, 1)))
