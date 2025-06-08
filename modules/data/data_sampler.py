import torch
import random
import torch.distributed as dist

class ModalityAnchoredBatchSampler:
    def __init__(self, text_idxs, image_idxs, vqa_idxs, image_per_batch, text_per_batch, vqa_per_batch):
        if dist.is_available() and dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        self.image_per_batch = image_per_batch
        self.text_per_batch = text_per_batch
        self.vqa_per_batch = vqa_per_batch

        self.image_idxs = image_idxs[self.rank::self.world_size]
        self.text_idxs = text_idxs[self.rank::self.world_size]
        self.vqa_idxs = vqa_idxs[self.rank::self.world_size]

        self.num_batches = min(
            len(self.image_idxs) // 2 if self.image_per_batch == 0 else len(self.image_idxs) // self.image_per_batch,
            len(self.text_idxs) // self.text_per_batch if self.text_per_batch > 0 else float('inf'),
            len(self.vqa_idxs) // self.vqa_per_batch if self.vqa_per_batch > 0 else float('inf'),
        )

        random.shuffle(self.image_idxs)
        random.shuffle(self.text_idxs)
        random.shuffle(self.vqa_idxs)

        print(f"[Rank {self.rank}] Batches: {self.num_batches}, "
              f"Image: {len(self.image_idxs)}, Text: {len(self.text_idxs)}, VQA: {len(self.vqa_idxs)}")

    def __iter__(self):
        for i in range(self.num_batches):
            batch = []
            batch += self.image_idxs[i * self.image_per_batch: (i + 1) * self.image_per_batch]
            batch += self.text_idxs[i * self.text_per_batch: (i + 1) * self.text_per_batch]
            batch += self.vqa_idxs[i * self.vqa_per_batch: (i + 1) * self.vqa_per_batch]
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches