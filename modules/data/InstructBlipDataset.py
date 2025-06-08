import json
import numpy as np
import re
import os

from torch.utils.data import Dataset
from typing import Any, Dict
from PIL import Image
from transformers import InstructBlipProcessor
from modules.prompt import qa_vqa_prompt_multi_choices_train, blip_visual_format_train

from ..utils import RandomImageIterator, generate_unrelated_distraction_prompts

class BlipVQAOriginDataset(Dataset):
    """
    PyTorch Dataset for Blip-Series Models. This class takes a HuggingFace Dataset as input.

    Each row, consists of image path(png/jpg/jpeg) and ground truth data (json/jsonl/txt).
    """
    
    def __init__(
        self,
        args : dict,
        processor : InstructBlipProcessor,
        mode: str
    ):
        super().__init__()
        '''
        dataset format should be:
        [
            {
                "image" : "path/to/image",
                "question" : "question to be asked",
                "answer" : "answer to the question"
            },
            ...
        ]
        '''
        self.args = args
        self.processor = processor

        if mode == "train":
            self.dataset = json.load(open(args.train_data_path, "r")) #Load Json format Dataset 
        else:
            self.dataset = json.load(open(args.eval_data_path, "r"))

        self.dataset_length = len(self.dataset)
        self.image_loader  = RandomImageIterator(f"data/VQADatasets/MMVP/images/")
        
        self.processor.tokenizer.padding_side = 'right'
        self.processor.tokenizer.truncation_side = 'right'

        self.prompt_format = blip_visual_format_train
        self.system_prompt = qa_vqa_prompt_multi_choices_train

        self.text_idxs = []
        self.image_idxs = []
        self.vqa_idxs = []

        for i, sample in enumerate(self.dataset):
            sample_type = sample.get("type")
            if sample_type == "text_heavy":
                self.text_idxs.append(i)
            elif sample_type == "image_heavy":
                self.image_idxs.append(i)
            elif sample_type == "VQA":
                self.vqa_idxs.append(i)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.

        Returns: {
            image : the original Receipt image
            question : the question to be asked
            answer : the answer to the question
        }
        """
        sample = self.dataset[idx]
        data_type = sample.get("type") 

        question = sample.get("question")
        answer = sample.get("answer")
        if answer.endswith(".."):
            answer = answer[:-1]

        origin_and_perturb_samples = []
        if data_type == "text_heavy":
            random_image = Image.fromarray(np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8))
            switch_image = Image.open(self.image_loader.get_random())
            black_image = Image.new('RGB', (336, 336), color = 'black')
            white_image = Image.new('RGB', (336, 336), color = 'white')

            for image in [random_image, switch_image, black_image, white_image]:
                processed_prompt = self.prompt_format.format(system_prompt=self.system_prompt, user_input=question, answer=answer)
                origin_and_perturb_samples.append((image, processed_prompt, data_type))

        elif data_type == "image_heavy":
            image_path = sample.get("image")
            image = Image.open(image_path).convert("RGB")

            interfere_facts = sample.get("facts")
            if not interfere_facts:
                choices = re.findall(r"\w+: (.+)", question)
                if choices:
                    random_option = np.random.choice(choices)
                    related_question = f"This picture seems to depict a {random_option}." + question
                else:
                    related_question = f"This picture seems to depict Nothing." + question
            else:
                related_question = list(interfere_facts.values())[0] + ' ' + question

            related_question = question
            unrelated_question = generate_unrelated_distraction_prompts() + question

            for ques in [question, related_question, unrelated_question]:
                processed_prompt =self.prompt_format.format(system_prompt=self.system_prompt, user_input=ques, answer=answer)
                origin_and_perturb_samples.append((image, processed_prompt, data_type))
            
        elif data_type == "VQA":
            image_path = sample.get("image")
            image = Image.open(image_path).convert("RGB")

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            processed_prompt = self.prompt_format.format(system_prompt="", user_input=question, answer=answer)
            origin_and_perturb_samples.append((image, processed_prompt, data_type))

        return origin_and_perturb_samples

class BlipVQADataset(Dataset):
    """
    PyTorch Daset for Blip-Series Models. 
    """
    def __init__(
        self,
        args : dict,
        processor : InstructBlipProcessor,
        mode: str
    ):
        super().__init__()
        '''
        dataset format should be:
        [
            {
                "image" : "path/to/image",
                "question" : "question to be asked",
                "answer" : "answer to the questitaon"
            },
            ...
        ]
        '''
        self.args = args
        self.processor = processor

        if mode == "train":
            self.dataset = json.load(open(args.train_data_path, "r")) #Load Json format Dataset 
        else:
            self.dataset = json.load(open(args.eval_data_path, "r"))

        self.dataset_length = len(self.dataset)
        self.image_loader  = RandomImageIterator(f"data/VQADatasets/MMVP/images/")

        self.processor.tokenizer.padding_side = 'right'
        self.processor.tokenizer.truncation_side = 'right'

        self.prompt_format = blip_visual_format_train
        self.system_prompt = qa_vqa_prompt_multi_choices_train

        self.text_idxs = []
        self.image_idxs = []
        self.vqa_idxs = []

        for i, sample in enumerate(self.dataset):
            sample_type = sample.get("type")
            if sample_type == "text_heavy":
                self.text_idxs.append(i)
            elif sample_type == "image_heavy":
                self.image_idxs.append(i)
            elif sample_type == "VQA":
                self.vqa_idxs.append(i)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Returns one item of the dataset.
        Process a single input sample for BLIP model.
        
        Args:
            image_path: Path to the image file
            question: Question text
            answer: Answer text
            
        Returns:
            Dictionary containing processed input_ids, attention_mask, pixel_values, qformer_input_ids, qformer_attention_mask, labels
        {   
            'origin' :{
                image : the original Receipt image
                question : the question to be asked
                answer : the answer to the question
            },
            'augmented' : {
                image : the augmented Receipt image
                question : the question to be asked
                answer : the answer to the question
            }
        }
        """
        sample = self.dataset[idx]
        data_type = sample.get("type") 

        question = sample.get("question")
        answer = sample.get("answer")
        if answer.endswith(".."):
            answer = answer[:-1]

        origin_samples, augmented_samples = [], []
        if data_type == "text_heavy":
            random_image = Image.fromarray(np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8))
            switch_image = Image.open(self.image_loader.get_random())
            black_image = Image.new('RGB', (336, 336), color = 'black')
            white_image = Image.new('RGB', (336, 336), color = 'white')

            for image in [random_image, switch_image, black_image, white_image]:
                processed_prompt = self.prompt_format.format(system_prompt=self.system_prompt, user_input=question, answer=answer)
                if image == random_image:
                    origin_samples.append((image, processed_prompt, data_type, idx))
                else:
                    augmented_samples.append((image, processed_prompt, data_type, idx))

        elif data_type == "image_heavy":
            image_path = sample.get("image")
            image = Image.open(image_path).convert("RGB")

            interfere_facts = sample.get("facts")
            if len(interfere_facts) == 0:
                choices = re.findall(r"\w+: (.+)", question)
                random_option = np.random.choice(choices)
                related_question = f"This picture seems to depict a {random_option}." + question
            else:
                related_question = list(interfere_facts.values())[0] + ' ' + question
            unrelated_question = generate_unrelated_distraction_prompts() + question

            for ques in [question, related_question, unrelated_question]:
                processed_prompt = self.prompt_format.format(system_prompt=self.system_prompt, user_input=ques, answer=answer)
                if ques == question:
                    origin_samples.append((image, processed_prompt, data_type, idx))
                else:
                    augmented_samples.append((image, processed_prompt, data_type, idx))
            
        elif data_type == "VQA":
            image_path = sample.get("image")
            image = Image.open(image_path).convert("RGB")

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            processed_prompt = self.prompt_format.format(system_prompt="", user_input=question, answer=answer)
            origin_samples.append((image, processed_prompt, data_type, idx))

        return {
            "origin": origin_samples,
            "augmented": augmented_samples
        }