from typing import Generator, List, Tuple
import numpy as np

class PurgedWalkForwardSplitter:
    """
    Purged and Embargoed Walk-Forward Time-Series Cross-Validator.
    
    Prevents data leakage in multi-step time series forecasting:
    1. Rolling-origin expanding or fixed window training folds.
    2. Purging: Drops training samples at the end of train fold whose target horizon
       overlaps with the test fold.
    3. Embargoing: Drops samples immediately following a test fold to prevent
       serial correlation contamination into subsequent folds.
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        min_train_size: int = 250,
        test_size: int = 60,
        purge_window: int = 5,
        embargo_window: int = 5,
        expanding: bool = True
    ):
        """
        Args:
            n_splits: Number of walk-forward folds.
            min_train_size: Minimum number of samples required in initial training fold.
            test_size: Number of samples per test fold.
            purge_window: Number of samples prior to test set to drop due to multi-step target overlap.
            embargo_window: Number of samples following test set to drop before next fold's train set.
            expanding: If True, train set expands across folds. If False, fixed train window size.
        """
        self.n_splits = n_splits
        self.min_train_size = min_train_size
        self.test_size = test_size
        self.purge_window = purge_window
        self.embargo_window = embargo_window
        self.expanding = expanding

    def get_n_splits(self, X: np.ndarray = None, y: np.ndarray = None, groups: np.ndarray = None) -> int:
        return self.n_splits

    def split(
        self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate (train_indices, test_indices) tuples.
        
        Args:
            X: Data array of shape (N, ...)
            
        Yields:
            train_idx: 1D numpy array of purged training indices.
            test_idx: 1D numpy array of test indices.
        """
        N = len(X)
        required_samples = self.min_train_size + self.n_splits * self.test_size
        if N < required_samples:
            raise ValueError(
                f"Dataset length ({N}) is too small for min_train_size={self.min_train_size}, "
                f"n_splits={self.n_splits}, test_size={self.test_size}. Required: >= {required_samples}."
            )
            
        start_test_idx = N - (self.n_splits * self.test_size)
        
        for fold in range(self.n_splits):
            test_start = start_test_idx + fold * self.test_size
            test_end = test_start + self.test_size
            
            # Unpurged train boundary ends just before test_start
            raw_train_end = test_start
            
            # Purged train boundary removes purge_window samples before test_start
            purged_train_end = max(0, raw_train_end - self.purge_window)
            
            if self.expanding:
                train_start = 0
            else:
                train_start = max(0, purged_train_end - self.min_train_size)
                
            train_indices = np.arange(train_start, purged_train_end)
            test_indices = np.arange(test_start, min(N, test_end))
            
            # Verify no index leakage
            if len(train_indices) == 0:
                raise ValueError(f"Fold {fold}: Purged training set is empty.")
                
            overlap = set(train_indices).intersection(set(test_indices))
            if overlap:
                raise AssertionError(f"Fold {fold}: Train and test set indices overlap! {overlap}")
                
            # Verify purge gap
            if len(train_indices) > 0 and len(test_indices) > 0:
                gap = test_indices[0] - train_indices[-1]
                if gap <= self.purge_window:
                    assert gap > self.purge_window, (
                        f"Fold {fold}: Purge gap failure. Gap={gap}, expected > {self.purge_window}"
                    )
                    
            yield train_indices, test_indices
