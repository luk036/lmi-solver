# -*- coding: utf-8 -*-
import math
from typing import Callable
import numpy as np


class LDLTMgr:
    """LDLT factorization (mainly for LMI oracles)

    - LDL^T square-root-free version
    - Option allow semidefinite
    - Cholesky–Banachiewicz style, row-based
    - Lazy evaluation
    - A matrix A in R^{m x m} is positive definite
                         iff v' A v > 0 for all v in R^n.
    - O(p^3) per iteration, independent of N
    """

    __slots__ = ("p", "v", "_n", "_T", "allow_semidefinite")

    def __init__(self, N: int):
        """Construct a new chol ext object

        Arguments:
            N (int): dimension
        """
        self.p = (0, 0)
        self.v: np.ndarray = np.zeros(N)

        self._n: int = N
        self._T: np.ndarray = np.zeros((N, N))  # pre-allocate storage

    def factorize(self, A: np.ndarray) -> bool:
        """Perform Cholesky Factorization

        Arguments:
            A (np.array): Symmetric Matrix

         If $A$ is positive definite, then $p$ is zero.
         If it is not, then $p$ is a positive integer,
         such that $v = R^{-1} e_p$ is a certificate vector
         to make $v'*A[:p,:p]*v < 0$
        """
        return self.factor(lambda i, j: A[i, j])

    def factor(self, get_elem: Callable[[int, int], float]) -> bool:
        """Perform Cholesky Factorization (square-root free version)

        Arguments:
            get_elem (callable): function to access symmetric matrix

         Construct $A(i, j)$ on demand, lazy evalution
        """
        start = 0  # range start
        self.p = (0, 0)
        for i in range(self._n):
            # j = start
            d = get_elem(i, start)
            for j in range(start, i):
                self._T[j, i] = d  # keep it for later use
                self._T[i, j] = d / self._T[j, j]  # the L[i, j]
                s = j + 1
                d = get_elem(i, s) - \
                    self._T[i, start:s].dot(self._T[start:s, s])
            self._T[i, i] = d
            if d <= 0.0:
                self.p = start, i + 1
                break
        return self.is_spd()

    def factor_with_allow_semidefinite(
        self, get_elem: Callable[[int, int], float]
    ) -> bool:
        """Perform Cholesky Factorization (square-root free version)

        Arguments:
            get_elem (callable): function to access symmetric matrix

         Construct $A(i, j)$ on demand, lazy evalution
        """
        start = 0  # range start
        self.p = (0, 0)
        for i in range(self._n):
            # j = start
            d = get_elem(i, start)
            for j in range(start, i):
                self._T[j, i] = d  # keep it for later use
                self._T[i, j] = d / self._T[j, j]  # the L[i, j]
                s = j + 1
                d = get_elem(i, s) - \
                    self._T[i, start:s].dot(self._T[start:s, s])
            self._T[i, i] = d
            if d < 0.0:
                self.p = start, i + 1
                break
            elif d == 0:
                start = i + 1  # T[i, i] == 0 (very unlikely), restart at i+1
        return self.is_spd()

    def is_spd(self):
        """Is $A$ symmetric positive definite (spd)

        Returns:
            bool: True if $A$ is a spd
        """
        return self.p[1] == 0

    def witness(self):
        """witness that certifies $A$ is not symmetric positive definite (spd)
            (square-root-free version)

           evidence: v' A v = -ep

        Raises:
            AssertionError: $A$ indeeds a spd matrix

        Returns:
            array: v
            float: ep
        """
        if self.is_spd():
            raise AssertionError()
        start, n = self.p
        m = n - 1
        self.v[m] = 1.0
        for i in range(m, start, -1):
            self.v[i - 1] = -self._T[i:n, i - 1].dot(self.v[i:n])
        return -self._T[m, m]

    def sym_quad(self, A: np.ndarray):
        """[summary]

        Arguments:
            v ([type]): [description]
            A ([type]): [description]

        Returns:
            [type]: [description]
        """
        s, n = self.p
        v = self.v[s:n]
        return v @ A[s:n, s:n] @ v

    def sqrt(self) -> np.ndarray:
        """Return upper triangular matrix R where A = R' * R

        Raises:
            AssertionError: [description]

        Returns:
            np.ndarray: [description]
        """
        if not self.is_spd():
            raise AssertionError()
        M = np.zeros((self._n, self._n))
        for i in range(self._n):
            M[i, i] = math.sqrt(self._T[i, i])
            for j in range(i + 1, self._n):
                M[i, j] = self._T[j, i] * M[i, i]
        return M
