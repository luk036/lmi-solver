# -*- coding: utf-8 -*-
from typing import Optional, Tuple, Union

import numpy as np

from .ldlt_mgr import LDLTMgr

Arr = Union[np.ndarray]
Cut = Tuple[Arr, float]


class LMI0Oracle:
    """Oracle for Linear Matrix Inequality constraint

    find  x
    s.t.  F * x ⪰ 0

    """

    def __init__(self, F):
        """[summary]

        Arguments:
            F (List[Arr]): [description]
        """
        self.F = F
        self.Q = LDLTMgr(len(F[0]))

    def assess_feas(self, x: Arr) -> Optional[Cut]:
        """[summary]

        Arguments:
            x (Arr): [description]

        Returns:
            Optional[Cut]: [description]
        """

        def get_elem(i, j):
            n = len(x)
            return sum(self.F[k][i, j] * x[k] for k in range(n))

        if not self.Q.factor(get_elem):
            ep = self.Q.witness()
            g = np.array([-self.Q.sym_quad(Fk) for Fk in self.F])
            return g, ep
        return None
