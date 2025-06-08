import os
import random
from pathlib import Path
from typing import List, Optional, Union
from collections import deque

class RandomImageIterator:
    def __init__(
        self, 
        root_dir: Union[str, Path],
        allowed_extensions: tuple = ('.jpg', '.jpeg', '.png', '.bmp', '.gif'),
        cache_size: int = 300  
    ):
        self.root_path = Path(root_dir)
        self.allowed_extensions = allowed_extensions
        self.cache_size = cache_size
        
        if not self.root_path.exists() or not self.root_path.is_dir():
            raise ValueError(f"Directory {root_dir} does not exist or is not a directory")
        
        self.all_images = [
            str(f) for f in self.root_path.rglob("*")
            if f.is_file() and f.suffix.lower() in self.allowed_extensions
        ]
        
        if not self.all_images:
            raise ValueError(f"No images found in {root_dir}")
            
        self.cache = deque(maxlen=self.cache_size)

        self._refill_cache()
        
    def _refill_cache(self):
        """refill the cache size with all random images"""
        remaining = self.cache_size - len(self.cache)
        if remaining > 0:
            additional_images = random.sample(
                self.all_images, 
                min(remaining, len(self.all_images))
            )
            self.cache.extend(additional_images)
    
    def __iter__(self):
        return self
    
    def __next__(self) -> str:
        if not self.cache:
            self._refill_cache()
            if not self.cache:
                raise StopIteration
        
        if len(self.cache) < self.cache_size // 2:
            self._refill_cache()
            
        return self.cache.popleft()
    
    def __len__(self) -> int:
        return len(self.all_images)
    
    def get_random(self) -> str:
        return next(self)
    
    def reset(self):
        self.cache.clear()
        self._refill_cache()