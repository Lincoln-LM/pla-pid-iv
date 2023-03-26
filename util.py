import numpy as np
from dataclasses import dataclass, fields


def dtype_dataclass(cls):
    """Decorator to turn a dataclass into a numpy dtype"""
    dataclass_cls = dataclass(cls)
    for field in fields(dataclass_cls):
        if hasattr(field, "__metadata__"):
            print(field.name, field.type.__metadata__)
    dataclass_cls.dtype = np.dtype(
        [
            (
                field.name,
                field.type.dtype
                if hasattr(field.type, "dtype")
                else (
                    field.type.__metadata__[0]
                    if hasattr(field.type, "__metadata__")
                    else field.type
                ),
            )
            for field in fields(dataclass_cls)
            if field.name != "dtype"
        ]
    )
    return dataclass_cls


def dtype_array(dtype: np.dtype, length: int) -> np.dtype:
    """Numpy dtype for an array of dtype of length"""
    return np.dtype([("", dtype, (length,))])[0]
