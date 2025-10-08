import numpy as np


def dict_to_prob_vector(prob_dict: dict, classes: list) -> list:
    """
    Transform a dictionary of class probabilities into an ordered vector
    according to the provided classes list.
    """
    return [prob_dict.get(cls, 0) for cls in classes]


def convert_to_array(probs, num_classes: int = 4) -> np.ndarray:
    """
    Convert assorted representations (string '[...]', list, ndarray) into a
    numpy array of probabilities. Falls back to a zero vector if conversion
    fails, preserving original notebooks' behavior.
    """
    if isinstance(probs, str):
        try:
            # Remove brackets and split on whitespace or commas
            cleaned = probs.strip('[]')
            # Support both space and comma separators
            if ',' in cleaned:
                parts = [p.strip() for p in cleaned.split(',') if p.strip()]
            else:
                parts = [p for p in cleaned.split() if p]
            return np.array([float(x) for x in parts])
        except Exception:
            return np.array([0.0] * num_classes)
    elif isinstance(probs, (list, tuple)):
        return np.array(probs)
    elif isinstance(probs, np.ndarray):
        return probs
    else:
        return np.array([0.0] * num_classes)