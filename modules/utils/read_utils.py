import yaml
import json
import re
import torch
import os
import base64
import math
from io import BytesIO

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_TOKEN_INDEX = -200

def read_yaml(path):
    file = open(path, "r", encoding="utf-8")
    string = file.read()
    dict = yaml.safe_load(string)

    return dict

def load_dataset(data_name, mode, task):
    if mode in ['text_heavy', 'pure_text']:
        if "infoseek" in data_name:
            dataset_nickname = "infoseek"
            with open("data/infoseek/sampled_val_mc.json", "r") as fin:
                dataset = json.load(fin)
        elif "openbookqa" in data_name:
            dataset_nickname = "OpenBookQA"
            with open(f"data/OpenBookQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)
        elif "mmlu" in data_name:
            dataset_nickname = "MMLU"
            with open(f"data/MMLU/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

    elif mode == 'image_heavy':
        if "caltech-101" in data_name:
            dataset_nickname = "caltech-101"
            with open(f"data/caltech-101/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)
        if "mini-imagenet" in data_name:
            dataset_nickname = "mini-imagenet"
            with open(f"data/mini-imagenet/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

    elif mode == 'VQA':

        if "viquae" in data_name:
            dataset_nickname = "viquae"
            with open(f"data/VQADatasets/viquae/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if "imagewikiqa" in data_name:
            dataset_nickname = "ImageWikiQA"
            with open(f"data/VQADatasets/ImageWikiQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if 'scienceqa' in data_name:
            dataset_nickname = "ScienceQA"
            with open(f"data/VQADatasets/ScienceQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if "aokvqa" in data_name:
            dataset_nickname = "A-OKVQA"
            with open(f"data/VQADatasets/A-OKVQA/multiple_choice_data_{task}.json", "r") as fin:
                dataset = json.load(fin)

        if "llava-instruct-665k" in data_name:
            dataset_nickname = "LLaVa-Instruct-665K"
            with open(f"/data/users/your name/data/LLaVa-Instruct-665K/llava_v1_5_mix665k_img.json", "r") as fin:
                dataset = json.load(fin)

        if "textcaps" in data_name:
            dataset_nickname = "TextCaps"
            with open(f"/data/users/your name/data/TextCaps/TextCaps_conversation.json", "r") as fin:
                dataset = json.load(fin)

        if "seed-bench-img" in data_name:
            dataset_nickname = "SEED-Bench-Img"
            with open(f"/data/users/your name/data/SEED-Bench-Img/multiple_choice_data.json", "r") as fin:
                dataset = json.load(fin)

        if "mmbench-en" in data_name:
            dataset_nickname = "MMBench-EN"
            with open(f"/data/users/your name/data/MMBench-EN/multiple_choice_data.json", "r") as fin:
                dataset = json.load(fin) 
    
    else:
        raise ValueError("Invalid mode")
    return dataset_nickname, dataset

def transform_image_path(data_nickname, image_path_or_text):
    if data_nickname in ["caltech-101", "mini-imagenet"]:
        return f"data/{data_nickname}/images/{image_path_or_text}"
    elif data_nickname in ["viquae", "ImageWikiQA", "ScienceQA"]:
        return os.path.join(f"data/VQADatasets/{data_nickname}/images/", image_path_or_text)
    elif data_nickname in ["A-OKVQA"]:
        return os.path.join("/data/users/your name/data/LLaVa-Instruct-665K/coco/images/", image_path_or_text)
    elif data_nickname in ['TextCaps', 'SEED-Bench-Img']:
        return os.path.join(f"/data/users/your name/data/{data_nickname}/", image_path_or_text)
    elif data_nickname in ["LLaVa-Instruct-665K"]:
        return os.path.join(f"/data/users/your name/data/LLaVa-Instruct-665K/", image_path_or_text)
    elif data_nickname in ["MMBench-EN"]:
        return BytesIO(base64.b64decode(image_path_or_text))
    else:
        raise ValueError("Invalid dataset nickname")

def extract_final_choice(text, dataset_nickname):
    # if dataset_nickname in ["caltech-101", "mini-imagenet"]:
    #     pattern = r'\b([A-Z])(?:[:\.]|$)'
    # else:
    #     pattern = r'\b([A-D])(?:[:\.]|$)'
    if dataset_nickname in ["caltech-101", "mini-imagenet"]:    
        pattern = r'\b([A-Za-z])\b'
    else:
        pattern = r'\b([A-Da-d])\b'
    matches = re.findall(pattern, text)
    # Find the first matched answer
    return matches[0].upper() if matches else None

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    elif image_aspect_ratio == "anyres":
        for image in images:
            image = process_anyres_image(image, image_processor, model_cfg.image_grid_pinpoints)
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

def process_anyres_image(image, processor, grid_pinpoints):
    """
    Process an image with variable resolutions.

    Args:
        image (PIL.Image.Image): The input image to be processed.
        processor: The image processor object.
        grid_pinpoints (str): A string representation of a list of possible resolutions.

    Returns:
        torch.Tensor: A tensor containing the processed image patches.
    """
    if type(grid_pinpoints) is list:
        possible_resolutions = grid_pinpoints
    else:
        possible_resolutions = ast.literal_eval(grid_pinpoints)
    best_resolution = select_best_resolution(image.size, possible_resolutions)
    image_padded = resize_and_pad_image(image, best_resolution)

    patches = divide_to_patches(image_padded, processor.crop_size['height'])

    image_original_resize = image.resize((processor.size['shortest_edge'], processor.size['shortest_edge']))

    image_patches = [image_original_resize] + patches
    image_patches = [processor.preprocess(image_patch, return_tensors='pt')['pixel_values'][0]
                     for image_patch in image_patches]
    return torch.stack(image_patches, dim=0)
