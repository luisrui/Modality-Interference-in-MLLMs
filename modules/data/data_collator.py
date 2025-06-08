import torch

from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Dict
from qwen_vl_utils import process_vision_info

@dataclass
class LLaVAOriginDataCollator:
    def __init__(self, processor):
        """
        DataCollator for LLaVA
        """
        self.processor = processor
        self.pattern = processor.tokenizer("ASSISTANT:", add_special_tokens=False).input_ids

    def __call__(self, features):
        '''
        Batch collator for LLaVA features
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  # [3, H, W] 
            "labels": labels,
        }
        '''
        features = [f for group_feature in features for f in group_feature]
        images = [f[0] for f in features]
        texts = [f[1] for f in features]
        data_types = [f[2] for f in features]

        whole_inputs, labels = self.process_llava_samples(images, texts)

        return {
            "input_ids": whole_inputs.input_ids,
            "attention_mask": whole_inputs.attention_mask,
            "pixel_values": whole_inputs.pixel_values,  
            "labels": labels,
            "data_types": data_types
        }
    
    def find_last_token_pos(self, input_ids):
        pattern_len = len(self.pattern)
        pattern_tensor = torch.tensor(self.pattern, device=input_ids.device).unsqueeze(0)  # [1, pattern_len]

        # windows for search [batch_size, seq_len - pattern_len + 1, pattern_len]
        windows = input_ids.unfold(1, pattern_len, 1)  # [batch_size, num_windows, pattern_len]
        matches = (windows == pattern_tensor).all(dim=2)  # [batch_size, num_windows]
        last_match_indices = matches.float().argmax(dim=1) + pattern_len - 1
        last_match_indices[matches.sum(dim=1) == 0] = -1 

        return last_match_indices  # [batch_size]

    def process_llava_samples(self, images, texts):
        whole_inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding="longest",
            padding_side="right"
        )
        # whole_inputs = self.processor(
        #     images=images,
        #     text=texts,
        #     return_tensors="pt",
        #     padding="max_length",  
        #     truncation=True,  
        #     max_length=1500,  
        #     padding_side="right"
        # )

        last_positions = self.find_last_token_pos(whole_inputs.input_ids)

        labels = whole_inputs.input_ids.clone()

        for i, last_prompt_pos in enumerate(last_positions.tolist()):
            if last_prompt_pos == -1:
                labels[i, :] = -100
            else:
                labels[i, :last_prompt_pos + 1] = -100
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return whole_inputs, labels
    
@dataclass
class LLaVADataCollator:
    def __init__(self, processor):
        """
        DataCollator for LLaVA
        """
        self.processor = processor
        self.pattern = processor.tokenizer("ASSISTANT:", add_special_tokens=False).input_ids

    def __call__(self, features: List[Dict]):
        '''
        Batch collator for LLaVA features
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  
            "labels": labels,
        }
        '''
        origin_samples = [sample for f in features for sample in f["origin"]]
        augmented_samples = [aug_sample for f in features for aug_sample in f["augmented"]]
        
        all_images = [sample[0] for sample in origin_samples] + [aug[0] for aug in augmented_samples]
        all_texts = [sample[1] for sample in origin_samples] + [aug[1] for aug in augmented_samples]
        data_types = [sample[2] for sample in origin_samples] + [aug[2] for aug in augmented_samples]
        # augmented_data_types = [aug[2] for aug in augmented_samples]

        num_origin = len(origin_samples)
        whole_inputs, labels = self.process_llava_samples(all_images, all_texts)
        batch_origin = {
            "input_ids": whole_inputs.input_ids[:num_origin],
            "attention_mask": whole_inputs.attention_mask[:num_origin],
            "pixel_values": whole_inputs.pixel_values[:num_origin],
            "labels": labels[:num_origin],
        }
        
        if len(augmented_samples) != 0:
            global_to_local_idx = {sample[3]: i for i, sample in enumerate(origin_samples)}

            batch_augmented = {
                "input_ids": whole_inputs.input_ids[num_origin:],
                "attention_mask": whole_inputs.attention_mask[num_origin:],
                "pixel_values": whole_inputs.pixel_values[num_origin:],
                "labels": labels[num_origin:],
                "sample_idxs": torch.tensor(
                    [global_to_local_idx[aug[3]] for aug in augmented_samples],
                    dtype=torch.int64,
                ),
            }
            batch_origin['sample_idxs'] = torch.tensor(
                [global_to_local_idx[ori[3]] for ori in origin_samples],
                dtype=torch.int64,
            )
        else:
            batch_augmented = None

        return {
            "origin": batch_origin,
            "augmented": batch_augmented,
            "data_types": data_types
        }
    
    def find_last_token_pos(self, input_ids):
        pattern_len = len(self.pattern)
        pattern_tensor = torch.tensor(self.pattern, device=input_ids.device).unsqueeze(0)  # [1, pattern_len]

        # windows for search [batch_size, seq_len - pattern_len + 1, pattern_len]
        windows = input_ids.unfold(1, pattern_len, 1)  # [batch_size, num_windows, pattern_len]
        matches = (windows == pattern_tensor).all(dim=2)  # [batch_size, num_windows]
        last_match_indices = matches.float().argmax(dim=1) + pattern_len - 1
        last_match_indices[matches.sum(dim=1) == 0] = -1 

        return last_match_indices  # [batch_size]

    def process_llava_samples(self, images, texts):
        whole_inputs = self.processor(
            images=images,
            text=texts,
            return_tensors="pt",
            padding="max_length",  
            truncation=True,  
            max_length=1000,  
            padding_side="right"
        )

        last_positions = self.find_last_token_pos(whole_inputs.input_ids)

        labels = whole_inputs.input_ids.clone()

        for i, last_prompt_pos in enumerate(last_positions.tolist()):
            if last_prompt_pos == -1:
                labels[i, :] = -100
            else:
                labels[i, :last_prompt_pos + 1] = -100
        
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return whole_inputs, labels
    
@dataclass
class InstructBlipOriginDataCollator:
    def __init__(self, processor):
        """
        DataCollator for InstructBlip
        """
        self.processor = processor
        #self.pattern = processor.tokenizer("Answer:", add_special_tokens=False).input_ids
        self.patterns = [
            processor.tokenizer("<answer>", add_special_tokens=False).input_ids,
            processor.tokenizer("<answer> ", add_special_tokens=False).input_ids,
            processor.tokenizer(" <answer>", add_special_tokens=False).input_ids,
        ]

    def __call__(self, features: List[Dict]):
        '''
        Batch collator for InstructBlip features
        '''
        features = [f for group_feature in features for f in group_feature]
        images = [f[0] for f in features]
        texts = [f[1] for f in features]
        data_types = [f[2] for f in features]
        
        qformer_inputs, labels = self.process_instructblip_samples(images, texts)

        return {
                "input_ids": qformer_inputs.input_ids,
                "attention_mask": qformer_inputs.attention_mask,
                "pixel_values": qformer_inputs.pixel_values,
                "qformer_input_ids": qformer_inputs.qformer_input_ids,
                "qformer_attention_mask" : qformer_inputs.qformer_attention_mask,
                "labels":labels,
                "data_types": data_types
            }
    
    def process_instructblip_samples(self, images, texts):
        qformer_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            max_length=512, 
            padding="max_length", 
            truncation=True,
            padding_side='right'
        )
        last_positions = self.find_last_token_pos(qformer_inputs.input_ids)
        labels = qformer_inputs.input_ids.clone()

        for i, last_prompt_pos in enumerate(last_positions.tolist()):
            if last_prompt_pos == -1:
                labels[i, :] = -100
            else:
                labels[i, :last_prompt_pos + 1] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return qformer_inputs, labels
    
    def find_last_token_pos(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.full((B,), -1, dtype=torch.long, device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)

        for pattern in self.patterns:
            pattern_len = len(pattern)
            if pattern_len == 0 or pattern_len > L:
                continue

            pattern_tensor = torch.tensor(pattern, device=device).unsqueeze(0)  # [1, P]
            windows = input_ids.unfold(1, pattern_len, 1)  # [B, L - P + 1, P]
            match = (windows == pattern_tensor).all(dim=2)  # [B, L - P + 1]

            for i in range(B):
                if not found[i]:
                    match_idx = match[i].nonzero(as_tuple=False)
                    if len(match_idx) > 0:
                        positions[i] = match_idx[-1].item() + pattern_len - 1  # 最后一个 token 的位置
                        found[i] = True
        
        assert found.any(), "No pattern found in input_ids"
        return positions

@dataclass
class InstructBlipDataCollator:
    def __init__(self, processor):
        """
        DataCollator for LLaVA
        """
        self.processor = processor
        self.patterns = [
            processor.tokenizer("<answer>", add_special_tokens=False).input_ids,
            processor.tokenizer("<answer> ", add_special_tokens=False).input_ids,
            processor.tokenizer(" <answer>", add_special_tokens=False).input_ids,
        ]

    def __call__(self, features: List[Dict]):
        '''
        Batch collator for LLaVA features
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,  
            "labels": labels,
        }
        '''

        origin_samples = [sample for f in features for sample in f["origin"]]
        augmented_samples = [aug_sample for f in features for aug_sample in f["augmented"]]
        
        all_images = [sample[0] for sample in origin_samples] + [aug[0] for aug in augmented_samples]
        all_texts = [sample[1] for sample in origin_samples] + [aug[1] for aug in augmented_samples]
        data_types = [sample[2] for sample in origin_samples] + [aug[2] for aug in augmented_samples]

        num_origin = len(origin_samples)
        qformer_inputs, labels = self.process_instructblip_samples(all_images, all_texts)
        batch_origin = {
            "input_ids": qformer_inputs.input_ids[:num_origin],
            "attention_mask": qformer_inputs.attention_mask[:num_origin],
            "pixel_values": qformer_inputs.pixel_values[:num_origin],
            "qformer_input_ids": qformer_inputs.qformer_input_ids[:num_origin],
            "qformer_attention_mask" : qformer_inputs.qformer_attention_mask[:num_origin],
            "labels":labels[:num_origin]
        }
        
        if len(augmented_samples) != 0:
            global_to_local_idx = {sample[3]: i for i, sample in enumerate(origin_samples)}

            batch_augmented = {
                "input_ids": qformer_inputs.input_ids[num_origin:],
                "attention_mask": qformer_inputs.attention_mask[num_origin:],
                "pixel_values": qformer_inputs.pixel_values[num_origin:],
                "qformer_input_ids": qformer_inputs.qformer_input_ids[num_origin:],
                "qformer_attention_mask" : qformer_inputs.qformer_attention_mask[num_origin:],
                "labels": labels[num_origin:],
                "sample_idxs": torch.tensor(
                    [global_to_local_idx[aug[3]] for aug in augmented_samples],
                    dtype=torch.int64,
                ),
            }
            batch_origin['sample_idxs'] = torch.tensor(
                [global_to_local_idx[ori[3]] for ori in origin_samples],
                dtype=torch.int64,
            )
        else:
            batch_augmented = None

        return {
            "origin": batch_origin,
            "augmented": batch_augmented,
            "data_types": data_types
        }
    
    def process_instructblip_samples(self, images, texts):
        qformer_inputs = self.processor(
            text=texts,
            images=images,
            return_tensors="pt",
            max_length=512, 
            padding="max_length", 
            truncation=True,
            padding_side='right'
        )
        last_positions = self.find_last_token_pos(qformer_inputs.input_ids)

        labels = qformer_inputs.input_ids.clone()

        for i, last_prompt_pos in enumerate(last_positions.tolist()):
            if last_prompt_pos == -1:
                labels[i, :] = -100
            else:
                labels[i, :last_prompt_pos + 1] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        return qformer_inputs, labels
    
    def find_last_token_pos(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.full((B,), -1, dtype=torch.long, device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)

        for pattern in self.patterns:
            pattern_len = len(pattern)
            if pattern_len == 0 or pattern_len > L:
                continue

            pattern_tensor = torch.tensor(pattern, device=device).unsqueeze(0)  # [1, P]
            windows = input_ids.unfold(1, pattern_len, 1)  # [B, L - P + 1, P]
            match = (windows == pattern_tensor).all(dim=2)  # [B, L - P + 1]

            for i in range(B):
                if not found[i]:
                    match_idx = match[i].nonzero(as_tuple=False)
                    if len(match_idx) > 0:
                        positions[i] = match_idx[-1].item() + pattern_len - 1 
                        found[i] = True
        
        assert found.any(), "No pattern found in input_ids"
        return positions

@dataclass 
class QwenOriginDataCollator:
    def __init__(self, processor, truncate_length=768):
        """
        DataCollator for LLaVA
        """
        self.processor = processor
        self.patterns = [
            processor.tokenizer("assistant\n", add_special_tokens=False).input_ids,
            processor.tokenizer("assistant", add_special_tokens=False).input_ids,
            processor.tokenizer(" assistant", add_special_tokens=False).input_ids,
            processor.tokenizer("assistant ", add_special_tokens=False).input_ids,
        ]
        # self.processor.tokenizer.padding_side = 'left'
        # self.processor.tokenizer.truncation_side = 'left'
        self.max_length = truncate_length

    def __call__(self, features: List[Dict]):
        '''
        Batch collator for InstructBlip features
        ''' 
        features = [f for group_feature in features for f in group_feature]
        messages = [f[0] for f in features]
        texts = [f[1] for f in features]
        data_types = [f[2] for f in features]
        
        whole_inputs, labels = self.process_qwen_samples(messages, texts)

        return {
                "input_ids": whole_inputs.input_ids,
                "attention_mask": whole_inputs.attention_mask,
                "pixel_values": whole_inputs.pixel_values,
                "image_grid_thw": whole_inputs.image_grid_thw,
                "labels":labels,
                "data_types": data_types
            }
    
    def process_qwen_samples(self, messages, texts):
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="longest", 
            padding_side='left',
            truncation=False,
        )
        last_positions = self.find_last_token_pos(inputs.input_ids)
        labels = inputs.input_ids.clone()

        for i, last_prompt_pos in enumerate(last_positions.tolist()):
            if last_prompt_pos == -1:
                labels[i, :] = -100
            else:
                labels[i, :last_prompt_pos + 1] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        input_ids, attention_mask, labels = self.left_truncate(inputs["input_ids"], labels)
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask

        return inputs, labels
    
    def find_last_token_pos(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.full((B,), -1, dtype=torch.long, device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)

        for pattern in self.patterns:
            pattern_len = len(pattern)
            if pattern_len == 0 or pattern_len > L:
                continue

            pattern_tensor = torch.tensor(pattern, device=device).unsqueeze(0)  # [1, P]
            windows = input_ids.unfold(1, pattern_len, 1)  # [B, L - P + 1, P]
            match = (windows == pattern_tensor).all(dim=2)  # [B, L - P + 1]

            for i in range(B):
                if not found[i]:
                    match_idx = match[i].nonzero(as_tuple=False)
                    if len(match_idx) > 0:
                        positions[i] = match_idx[-1].item() + pattern_len - 1  
                        found[i] = True
        
        assert found.any(), "No pattern found in input_ids"
        return positions

    def left_truncate(self, input_ids, labels, image_patch_token_id=151655):
        """
        Left-truncate input_ids and labels to ensure all image_patch tokens are preserved.
        If the full sequence exceeds self.max_length, truncate the tail (i.e., answer) after image_patch tokens.
        Left pad to maintain batch alignment.
        """
        batch_size, seq_len = input_ids.shape
        pad_token_id = self.processor.tokenizer.pad_token_id

        new_input_ids, new_labels, new_attention_masks = [], [], []

        for i in range(batch_size):
            ids = input_ids[i]
            lbl = labels[i]

            # Find the position of first non-pad token and last image_patch token
            first_non_pad = (ids != pad_token_id).nonzero(as_tuple=False)
            first_token = first_non_pad[0].item()
            image_patch_pos = (ids == image_patch_token_id).nonzero(as_tuple=False)
            last_image_token = image_patch_pos[-1].item()

            must_keep_ids = ids[first_token:last_image_token + 1]
            must_keep_lbl = lbl[first_token:last_image_token + 1]
            num_must_tokens = must_keep_ids.size(0)

            tail_ids = ids[last_image_token + 1:]
            tail_lbl = lbl[last_image_token + 1:]
            budget_tail = self.max_length - num_must_tokens
            budget_tail = max(budget_tail, 0)

            if tail_ids.size(0) > budget_tail:
                tail_ids = tail_ids[-budget_tail:]
                tail_lbl = tail_lbl[-budget_tail:]

            preserved_ids = torch.cat([must_keep_ids, tail_ids], dim=0)
            preserved_lbl = torch.cat([must_keep_lbl, tail_lbl], dim=0)

            pad_len = self.max_length - preserved_ids.size(0)
            padded_ids = F.pad(preserved_ids, (pad_len, 0), value=pad_token_id)
            padded_lbl = F.pad(preserved_lbl, (pad_len, 0), value=-100)
            padded_mask = (padded_ids != pad_token_id).long()

            new_input_ids.append(padded_ids)
            new_labels.append(padded_lbl)
            new_attention_masks.append(padded_mask)

        return torch.stack(new_input_ids), torch.stack(new_attention_masks), torch.stack(new_labels)

@dataclass
class QwenDataCollator:
    def __init__(self, processor, truncate_length=768):
        """
        DataCollator for Qwen fine-tuning with both origin and augmented samples.
        """
        self.processor = processor
        self.patterns = [
            processor.tokenizer("assistant\n", add_special_tokens=False).input_ids,
            processor.tokenizer("assistant", add_special_tokens=False).input_ids,
            processor.tokenizer(" assistant", add_special_tokens=False).input_ids,
            processor.tokenizer("assistant ", add_special_tokens=False).input_ids,
        ]
        # self.processor.tokenizer.padding_side = 'left'
        # self.processor.tokenizer.truncation_side = 'left'
        self.max_length = truncate_length

    def __call__(self, features: List[Dict]):
        origin_samples = [s for f in features for s in f["origin"]]
        augmented_samples = [s for f in features for s in f["augmented"]]

        all_messages = [s[0] for s in origin_samples] + [s[0] for s in augmented_samples]
        all_texts = [s[1] for s in origin_samples] + [s[1] for s in augmented_samples]
        data_types = [s[2] for s in origin_samples] + [s[2] for s in augmented_samples]

        num_origin = len(origin_samples)
        whole_inputs, labels = self.process_qwen_samples(all_messages, all_texts)

        batch_origin = {
            "input_ids": whole_inputs.input_ids[:num_origin],
            "attention_mask": whole_inputs.attention_mask[:num_origin],
            "pixel_values": whole_inputs.pixel_values[:num_origin],
            "image_grid_thw": whole_inputs.image_grid_thw[:num_origin],
            "labels": labels[:num_origin],
        }

        if len(augmented_samples) > 0:
            global_to_local_idx = {sample[3]: i for i, sample in enumerate(origin_samples)}

            batch_augmented = {
                "input_ids": whole_inputs.input_ids[num_origin:],
                "attention_mask": whole_inputs.attention_mask[num_origin:],
                "pixel_values": whole_inputs.pixel_values[num_origin:],
                "image_grid_thw": whole_inputs.image_grid_thw[num_origin:],
                "labels": labels[num_origin:],
                "sample_idxs": torch.tensor(
                    [global_to_local_idx[s[3]] for s in augmented_samples],
                    dtype=torch.long,
                ),
            }
            batch_origin['sample_idxs'] = torch.tensor(
                [global_to_local_idx[s[3]] for s in origin_samples],
                dtype=torch.long,
            )
        else:
            batch_augmented = None

        return {
            "origin": batch_origin,
            "augmented": batch_augmented,
            "data_types": data_types
        }

    def process_qwen_samples(self, messages, texts):
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding="longest",
            padding_side="left",
            truncation=False,
        )

        last_positions = self.find_last_token_pos(inputs.input_ids)
        labels = inputs.input_ids.clone()

        for i, last_prompt_pos in enumerate(last_positions.tolist()):
            if last_prompt_pos == -1:
                labels[i, :] = -100
            else:
                labels[i, :last_prompt_pos + 1] = -100

        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        input_ids, attention_mask, labels = self.left_truncate(inputs.input_ids, labels)
        inputs["input_ids"] = input_ids
        inputs["attention_mask"] = attention_mask

        return inputs, labels

    def find_last_token_pos(self, input_ids):
        B, L = input_ids.shape
        device = input_ids.device
        positions = torch.full((B,), -1, dtype=torch.long, device=device)
        found = torch.zeros(B, dtype=torch.bool, device=device)

        for pattern in self.patterns:
            pattern_len = len(pattern)
            if pattern_len == 0 or pattern_len > L:
                continue

            pattern_tensor = torch.tensor(pattern, device=device).unsqueeze(0)
            windows = input_ids.unfold(1, pattern_len, 1)
            match = (windows == pattern_tensor).all(dim=2)

            for i in range(B):
                if not found[i]:
                    match_idx = match[i].nonzero(as_tuple=False)
                    if len(match_idx) > 0:
                        positions[i] = match_idx[-1].item() + pattern_len - 1
                        found[i] = True

        assert found.any(), "No pattern found in input_ids"
        return positions

    def left_truncate(self, input_ids, labels, image_patch_token_id=151655):
        """
        Left-truncate input_ids and labels to ensure all image_patch tokens are preserved.
        If the full sequence exceeds self.max_length, truncate the tail (i.e., answer) after image_patch tokens.
        Left pad to maintain batch alignment.
        """
        batch_size, seq_len = input_ids.shape
        pad_token_id = self.processor.tokenizer.pad_token_id

        new_input_ids, new_labels, new_attention_masks = [], [], []

        for i in range(batch_size):
            ids = input_ids[i]
            lbl = labels[i]

            # Find the position of first non-pad token and last image_patch token
            first_non_pad = (ids != pad_token_id).nonzero(as_tuple=False)
            first_token = first_non_pad[0].item()
            image_patch_pos = (ids == image_patch_token_id).nonzero(as_tuple=False)
            last_image_token = image_patch_pos[-1].item()

            must_keep_ids = ids[first_token:last_image_token + 1]
            must_keep_lbl = lbl[first_token:last_image_token + 1]
            num_must_tokens = must_keep_ids.size(0)

            tail_ids = ids[last_image_token + 1:]
            tail_lbl = lbl[last_image_token + 1:]
            budget_tail = self.max_length - num_must_tokens
            budget_tail = max(budget_tail, 0)

            if tail_ids.size(0) > budget_tail:
                tail_ids = tail_ids[-budget_tail:]
                tail_lbl = tail_lbl[-budget_tail:]

            preserved_ids = torch.cat([must_keep_ids, tail_ids], dim=0)
            preserved_lbl = torch.cat([must_keep_lbl, tail_lbl], dim=0)

            pad_len = self.max_length - preserved_ids.size(0)
            padded_ids = F.pad(preserved_ids, (pad_len, 0), value=pad_token_id)
            padded_lbl = F.pad(preserved_lbl, (pad_len, 0), value=-100)
            padded_mask = (padded_ids != pad_token_id).long()

            new_input_ids.append(padded_ids)
            new_labels.append(padded_lbl)
            new_attention_masks.append(padded_mask)

        return torch.stack(new_input_ids), torch.stack(new_attention_masks), torch.stack(new_labels)