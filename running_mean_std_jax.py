# running_mean_std_jax.py
import jax.numpy as jnp
import threading
from typing import Tuple

class RunningMeanStd:
    def __init__(self, epsilon: float = 1e-4, shape: Tuple[int, ...] = ()):
        """
        Calculates the running mean and std of a data stream.
        Thread-safe version using threading.Lock.
        """
        self.mean = jnp.zeros(shape)
        self.var = jnp.ones(shape)
        self.count = epsilon
        self._lock = threading.Lock()

    def get_stats(self) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
        """Потокобезопасный метод для одновременного получения mean, var и count."""
        with self._lock:
            return self.mean, self.var, self.count

    def set_stats(self, mean: jnp.ndarray, var: jnp.ndarray, count: float):
        """Потокобезопасный метод для одновременной записи параметров."""
        with self._lock:
            self.mean = mean
            self.var = var
            self.count = count

    def copy(self) -> "RunningMeanStd":
        new_object = RunningMeanStd(shape=self.mean.shape)
        with self._lock:
            new_object.mean = self.mean
            new_object.var = self.var
            new_object.count = self.count
        return new_object

    def combine(self, other: "RunningMeanStd") -> None:
        other_mean, other_var, other_count = other.get_stats()
        self.update_from_moments(other_mean, other_var, other_count)

    def update(self, arr: jnp.ndarray) -> None:
        batch_mean = jnp.mean(arr, axis=0)
        batch_var = jnp.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)
        

    def update_from_moments(self, batch_mean: jnp.ndarray, batch_var: jnp.ndarray, batch_count: float) -> None:
        with self._lock:
            delta = batch_mean - self.mean
            tot_count = self.count + batch_count

            new_mean = self.mean + delta * batch_count / tot_count
            m_a = self.var * self.count
            m_b = batch_var * batch_count
            m_2 = m_a + m_b + jnp.square(delta) * self.count * batch_count / tot_count
            new_var = m_2 / tot_count

            new_count = batch_count + self.count

            self.mean = new_mean
            self.var = new_var
            self.count = new_count