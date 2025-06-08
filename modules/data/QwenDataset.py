import json
import numpy as np
import re

from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, InterpolationMode
from PIL import Image
from modules.prompt import qa_vqa_prompt_multi_choices_train
from transformers import AutoProcessor

from ..utils import RandomImageIterator, generate_unrelated_distraction_prompts

class QwenVQAOriginDataset(Dataset):
    """
    PyTorch Dataset for Qwen fine-tuning.

    Expected JSON format of the dataset:
    [
        {
            "image": "path/to/image",
            "question": "some question",
            "answer": "some answer"
        },
        ...
    ]
    """

    def __init__(
        self,
        args,
        processor: AutoProcessor,
        mode: str = "train",
    ):
        super().__init__()
        self.args = args
        self.processor = processor

        if mode == "train":
            self.dataset = json.load(open(args.train_data_path, "r"))
        else:
            self.dataset = json.load(open(args.eval_data_path, "r"))
        self.dataset_length = len(self.dataset)
        self.image_loader  = RandomImageIterator(f"data/VQADatasets/MMVP/images/")
        
        self.system_prompt = qa_vqa_prompt_multi_choices_train
        self.image_resize_size = getattr(args, "image_resize", 392)

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

    def __getitem__(self, idx: int):
        """
        Returns one item of the dataset with the fields:
            - input_ids
            - attention_mask
            - pixel_values
            - labels
        which are compatible with `LlavaForConditionalGeneration.forward(...)`.
        """
        sample = self.dataset[idx]
        data_type = sample.get("type") 

        origin_and_perturb_samples = []
        if data_type == "text_heavy":
            #image_path = sample.get("image")
            question = sample.get("question")
            answer = sample.get("answer")

            random_image = Image.fromarray(np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8))
            switch_image = Image.open(self.image_loader.get_random())
            black_image = Image.new('RGB', (336, 336), color = 'black')
            white_image = Image.new('RGB', (336, 336), color = 'white')

            for image in [random_image, switch_image, black_image, white_image]:
                #image = self.resize_image(image)
                processed_prompt, message = self.process_prompt(image, question, answer, data_type)
                origin_and_perturb_samples.append((message, processed_prompt, data_type))

        elif data_type == "image_heavy":
            image_path = sample.get("image")
            image = Image.open(image_path).convert("RGB")
            #image = self.resize_image(image)

            question = sample.get("question")
            answer = sample.get("answer")

            interfere_facts = sample.get("facts")
            if len(interfere_facts) == 0:
                choices = re.findall(r"\w+: (.+)", question)
                random_option = np.random.choice(choices)
                related_question = f"This picture seems to depict a {random_option}." + question
            else:
                related_question = list(interfere_facts.values())[0] + ' ' + question
                
            unrelated_question = generate_unrelated_distraction_prompts() + question

            for ques in [question, related_question, unrelated_question]:
                processed_prompt, message = self.process_prompt(image, ques, answer, data_type)
                origin_and_perturb_samples.append((message, processed_prompt, data_type))
            
        elif data_type == "VQA":
            image_path = sample.get("image")
            image = Image.open(image_path).convert("RGB")
            #image = self.resize_image(image)

            question = sample.get("question")
            answer = sample.get("answer")

            question = question.replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            processed_prompt, message = self.process_prompt(image, question, answer, data_type)
            origin_and_perturb_samples.append((message, processed_prompt, data_type))

        return origin_and_perturb_samples
    
    def process_prompt(self, image, question, answer, data_type):
        img_content = {
            "type": "image",
            "image": image,
            "min_pixels": self.args.image_min_pixels,
            "max_pixels": self.args.image_max_pixels,
        }
        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt if data_type != "VQA" else ""},
                ],
            },
            {
                "role": "user",
                "content": [
                    img_content,
                    {"type": "text", "text": question},
                ],
            }
        ]
        train_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True) + answer

        return train_prompt, conversation

class QwenVQADataset(Dataset):
    """
    PyTorch Dataset for Qwen fine-tuning with both origin and augmented samples.

    Returns:
        dict with keys:
        - "origin": List[Tuple[conversation, input_text, data_type]]
        - "augmented": List[Tuple[conversation, input_text, data_type]]
    """

    def __init__(
        self,
        args,
        processor: AutoProcessor,
        mode: str = "train",
    ):
        super().__init__()
        self.args = args
        self.processor = processor

        if mode == "train":
            self.dataset = json.load(open(args.train_data_path, "r"))
        else:
            self.dataset = json.load(open(args.eval_data_path, "r"))

        self.dataset_length = len(self.dataset)
        self.image_loader  = RandomImageIterator(f"data/VQADatasets/MMVP/images/")

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

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        data_type = sample.get("type")

        origin_samples, augmented_samples = [], []

        if data_type == "text_heavy":
            question = sample["question"]
            answer = sample["answer"]

            images = {
                "origin": Image.fromarray(np.random.randint(0, 256, (336, 336, 3), dtype=np.uint8)),
                "switch": Image.open(self.image_loader.get_random()),
                "black": Image.new('RGB', (336, 336), color='black'),
                "white": Image.new('RGB', (336, 336), color='white'),
            }

            for tag, img in images.items():
                prompt, conversation = self.process_prompt(img, question, answer, data_type)
                if tag == "origin":
                    origin_samples.append((conversation, prompt, data_type, idx))
                else:
                    augmented_samples.append((conversation, prompt, data_type, idx))

        elif data_type == "image_heavy":
            image = Image.open(sample["image"]).convert("RGB")
            question = sample["question"]
            answer = sample["answer"]

            interfere_facts = sample.get("facts", {})
            if interfere_facts:
                related_question = list(interfere_facts.values())[0] + " " + question
            else:
                choices = re.findall(r"\w+: (.+)", question)
                random_option = np.random.choice(choices) if choices else "object"
                related_question = f"This picture seems to depict a {random_option}." + question

            unrelated_question = generate_unrelated_distraction_prompts() + question

            for ques in [question, related_question, unrelated_question]:
                prompt, conversation = self.process_prompt(image, ques, answer, data_type)
                if ques == question:
                    origin_samples.append((conversation, prompt, data_type, idx))
                else:
                    augmented_samples.append((conversation, prompt, data_type, idx))

        elif data_type == "VQA":
            image = Image.open(sample["image"]).convert("RGB")
            question = sample["question"].replace("<image>\n", "").replace("\n<image>", "").replace("<image>", "")
            answer = sample["answer"]

            prompt, conversation = self.process_prompt(image, question, answer, data_type)
            origin_samples.append((conversation, prompt, data_type, idx))

        return {
            "origin": origin_samples,
            "augmented": augmented_samples
        }

    def process_prompt(self, image, question, answer, data_type):
        img_content = {
            "type": "image",
            "image": image,
            "min_pixels": self.args.image_min_pixels,
            "max_pixels": self.args.image_max_pixels,
        }

        conversation = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": self.system_prompt if data_type != "VQA" else ""},
                ],
            },
            {
                "role": "user",
                "content": [
                    img_content,
                    {"type": "text", "text": question},
                ],
            }
        ]

        train_prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        ) + answer

        return train_prompt, conversation