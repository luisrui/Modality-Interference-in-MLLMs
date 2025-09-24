import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from transformers import LlavaProcessor, LlavaForConditionalGeneration
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer,AutoModel, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info


from ..prompt import llava_visual_format, blip_visual_format, text_format, cogvlm_format

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
#from ..pretrained_models.cogVLM.cogvlm import CogVLMForCausalLM

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

class LocalModelChat():
    def __init__(self, args, mode, initial_prompt=""):
        self.args = args
        self.mode = mode
        self.prompt = initial_prompt
        self.device = torch.device(args.device)
        self.generate_config = {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "temperature": args.temperature,
        }
        if self.args.temperature == 0.0:
            self.generate_config.update({"do_sample": False})

        if "blip" in args.model_name.lower():
            self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                "Salesforce/instructblip-vicuna-7b",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.prompt_format = blip_visual_format
            self.num_visual_tokens = int((self.model.config.vision_config.image_size / self.model.config.vision_config.patch_size) ** 2)
            #self.model.config.text_config.max_position_embeddings = 2048
        elif "llava-v1.6-vicuna-7b" in args.model_name.lower():
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf")
            self.model = LlavaNextForConditionalGeneration.from_pretrained(
                "llava-hf/llava-v1.6-vicuna-7b-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.prompt_format = llava_visual_format
            self.num_visual_tokens = int((self.model.config.vision_config.image_size / self.model.config.vision_config.patch_size) ** 2)
        elif "llava-1.5-7b" in args.model_name.lower():
            self.processor = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf", num_additional_image_tokens = 0)
            self.model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf",
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            )
            self.prompt_format = llava_visual_format
        elif "cogvlm" in args.model_name.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
            self.model = CogVLMForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.prompt_format = cogvlm_format
            self.split_patch = "ASSISTANT:"
        elif "qwen" in args.model_name.lower():
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", trust_remote_code=True)
            self.model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", device_map="cuda", trust_remote_code=True).eval()
        elif "vicuna" in args.model_name.lower():
            self.processor = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
            self.model = AutoModelForCausalLM.from_pretrained(
                "lmsys/vicuna-7b-v1.5",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True)
            self.prompt_format = text_format
        else:
            raise ValueError(f"Model {args.model_name} not supported")

        self.model.eval()
        self.model.to(self.device)

    def remode(self, mode):
        # self.mode = mode
        if mode == "pure_text":
            self.prompt_format = text_format
        else:
            self.prompt_format = llava_visual_format

    @torch.inference_mode()
    def chat(self, **kwargs):
        if self.mode == "pure_text": #No image here
            text = kwargs["text"]
            
            messages = [[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]] * self.args.sampling_times
            if "vicuna" in self.args.model_name.lower():
                text = self.prompt_format.format(system_prompt=self.prompt, user_input=text)
            else:
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            inputs = self.processor(text=text, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )
            slen = inputs.input_ids.size(1)
            if "blip" in self.args.model_name.lower():
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            else:
                generated_text = self.processor.batch_decode(outputs[:, slen:], skip_special_tokens=True)[0].strip()
            ### return the probs of target tokens of 4 selections   
            is_scored = kwargs.get("is_scored")
            if is_scored:
                if "blip" not in self.args.model_name.lower() and "llava" not in self.args.model_name.lower() and "vicuna" in self.args.model_name.lower():
                    target_token_ids = torch.tensor(self.processor.convert_tokens_to_ids(["A", "B", "C", "D"]))
                else:    
                    target_token_ids = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"]))
                probs = self.model(**inputs).logits[:, -1, :].detach().cpu()
                target_probs = torch.index_select(probs, 1, target_token_ids).squeeze()

        else: # Image + Text as input
            text = kwargs.get("text")
            image = kwargs["image"]
            messages = [[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]] * self.args.sampling_times
            try:
                text = self.processor.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            except:
                text = self.prompt_format.format(system_prompt=self.prompt, user_input=text)
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )
            slen = inputs.input_ids.size(1)
            if "blip" in self.args.model_name.lower():
                generated_text = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            else:
                generated_text = self.processor.batch_decode(outputs[:, slen:], skip_special_tokens=True)[0].strip()

            is_scored = kwargs.get("is_scored")
            if is_scored:
                target_token_ids = torch.tensor(self.processor.tokenizer.convert_tokens_to_ids(["A", "B", "C", "D"]))
                probs = self.model(**inputs).logits[:, -1, :].detach().cpu()
                target_probs = torch.index_select(probs, 1, target_token_ids).squeeze()

        is_scored = kwargs.get("is_scored")
        if is_scored:
            return generated_text, target_probs.tolist()
        return generated_text 

    @torch.inference_mode()
    def cogvlm_predict(self, context):
        text = context["text"]
        image = context["image"]

        format_text = self.prompt_format.format(system_prompt=self.prompt, user_input=text)
        inputs = self.model.build_conversation_input_ids(self.tokenizer, query=format_text, history=[], images=[image]) 
        inputs = {
            'input_ids': inputs['input_ids'].unsqueeze(0).to('cuda'),
            'token_type_ids': inputs['token_type_ids'].unsqueeze(0).to('cuda'),
            'attention_mask': inputs['attention_mask'].unsqueeze(0).to('cuda'),
            'images': [[inputs['images'][0].to('cuda').to(torch.bfloat16)]],
        }
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )

        slen = inputs.input_ids.size(1)
        decoded = self.processor.batch_decode(
            outputs[:, slen:],
            skip_special_tokens=True
        )[0].strip()
        answer = decoded.split(self.split_patch)[-1].strip() if self.split_patch in decoded else decoded
        return answer
    
class LocalModelChat_FromCheckpoint():
    def __init__(self, args, mode, initial_prompt=""):
        self.args = args
        self.checkpoint_path = getattr(args, 'checkpoint_path', None)
        self.mode = mode
        self.prompt = initial_prompt
        self.device = torch.device(args.device)
        self.generate_config = {
            "do_sample": True,
            "max_new_tokens": args.max_new_tokens,
            "top_p": args.top_p,
            "repetition_penalty": args.repetition_penalty,
            "length_penalty": args.length_penalty,
            "temperature": args.temperature,
        }
        if self.args.temperature == 0.0:
            self.generate_config.update({"do_sample": False})

        if "instructblip-vicuna-7b" in args.model_name.lower():
            self.processor = InstructBlipProcessor.from_pretrained(
                "Salesforce/instructblip-vicuna-7b")
            if self.checkpoint_path:
                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    args.checkpoint_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = InstructBlipForConditionalGeneration.from_pretrained(
                    "Salesforce/instructblip-vicuna-7b",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            self.prompt_format = blip_visual_format
            self.num_visual_tokens = int((self.model.config.vision_config.image_size / self.model.config.vision_config.patch_size) ** 2)
            #self.split_patch = "Answer:"
            self.split_patches = ["<answer> "]
        elif "instructblip-vicuna-13b" in args.model_name.lower():
            self.processor = InstructBlipProcessor.from_pretrained(
                "Salesforce/instructblip-vicuna-13b")
            self.model = InstructBlipForConditionalGeneration.from_pretrained(
                args.checkpoint_path if args.checkpoint_path else "Salesforce/instructblip-vicuna-13b",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            self.prompt_format = blip_visual_format
            self.num_visual_tokens = int((self.model.config.vision_config.image_size / self.model.config.vision_config.patch_size) ** 2)
            #self.split_patch = "Answer:"
            self.split_patches = ["<answer> "]
        elif "llava-1.5-7b" in args.model_name.lower():
            self.processor = LlavaProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
            self.processor.tokenizer.padding_side = "left"
            if self.checkpoint_path:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    args.checkpoint_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-1.5-7b-hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            self.prompt_format = llava_visual_format
            self.split_patches = ["ASSISTANT:"]
        elif "llava-1.5-13b" in args.model_name.lower():
            load_path = args.checkpoint_path if  args.checkpoint_path else "llava-hf/llava-1.5-13b-hf"
            self.processor = LlavaProcessor.from_pretrained(load_path)
            self.processor.tokenizer.padding_side = "right"
            base_model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-13b-hf",
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            if args.checkpoint_path:
                print(f"Loading LoRA weights from {args.checkpoint_path}")
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, load_path)
                self.model = model.merge_and_unload() if getattr(args, "merge_lora", False) else model
            else:
                self.model = base_model
            self.prompt_format = llava_visual_format
            self.split_patches = ["ASSISTANT:"]
        elif "llava-v1.6-34b-hf" in args.model_name.lower():
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-34b-hf", padding_side='left')
            if self.checkpoint_path:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    args.checkpoint_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-v1.6-34b-hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            self.prompt_format = llava_visual_format
            self.split_patches = ["assistant"]
        elif "llava-v1.6-vicuna-7b" in args.model_name.lower():
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-vicuna-7b-hf", padding_side='left')
            if self.checkpoint_path:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    args.checkpoint_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-v1.6-vicuna-7b-hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            self.prompt_format = llava_visual_format
            self.split_patches = ["ASSISTANT:"]
        elif "llava-next-72b-hf" in args.model_name.lower():
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-next-72b-hf", padding_side='left')
            self.processor.pad_token_id = self.processor.tokenizer.eos_token_id
            if self.checkpoint_path:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    args.checkpoint_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True
                )
            else:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-next-72b-hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True
                )
            self.prompt_format = llava_visual_format
            self.split_patches = ["assistant"]
        elif "llava-next-110b-hf" in args.model_name.lower():
            self.processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-next-110b-hf", padding_side='left')
            self.processor.pad_token_id = self.processor.tokenizer.eos_token_id
            if self.checkpoint_path:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    args.checkpoint_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True
                )
            else:
                self.model = LlavaNextForConditionalGeneration.from_pretrained(
                    "llava-hf/llava-next-110b-hf",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    load_in_4bit=True
                )
            self.prompt_format = llava_visual_format
            self.split_patches = ["assistant"]
        elif "qwen2.5-vl-7b" in args.model_name.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
            self.processor.tokenizer.padding_side = "left"
            if self.checkpoint_path:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.checkpoint_path, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-7B-Instruct", 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True
                )
                self.generate_config.update({"max_new_tokens": 100})
            self.split_patches = ["assistant"]
        elif "qwen2.5-vl-3b" in args.model_name.lower():
            from transformers import Qwen2_5_VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
            self.processor.tokenizer.padding_side = "left"
            if self.checkpoint_path:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    args.checkpoint_path, 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True
                )
            else:
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    "Qwen/Qwen2.5-VL-3B-Instruct", 
                    torch_dtype=torch.float16, 
                    low_cpu_mem_usage=True
                )
                self.generate_config.update({"max_new_tokens": 100})
            self.split_patches = ["Answer:", "assistant"]
        elif "qwen2-vl-2b":
            from transformers import Qwen2VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
            self.processor.tokenizer.padding_side = "left"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.checkpoint_path if args.checkpoint_path else "Qwen/Qwen2-VL-2B-Instruct", 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            )
            self.generate_config.update({"max_new_tokens": 100})
            self.split_patches = ["Answer:", "assistant"]
        elif "qwen2-vl-7b" in args.model_name.lower():
            from transformers import Qwen2VLForConditionalGeneration
            self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
            self.processor.tokenizer.padding_side = "left"
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                args.checkpoint_path if args.checkpoint_path else "Qwen/Qwen2-VL-7B-Instruct", 
                torch_dtype=torch.float16, 
                low_cpu_mem_usage=True
            )
            self.generate_config.update({"max_new_tokens": 100})
            self.split_patches = ["Answer:", "assistant"]   
        elif "internvl3-8b" in args.model_name.lower():
            self.processor = None  # InternVL不需要processor，它直接吃 pixel_values
            self.model = AutoModel.from_pretrained(
                args.checkpoint_path if args.checkpoint_path else "InternVL/InternVL3-8B",
                torch_dtype=torch.bfloat16,
                load_in_8bit=False,
                low_cpu_mem_usage=True,
                # use_flash_attn=True,
                trust_remote_code=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True, use_fast=False)
        elif "cogvlm" in args.model_name.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5')
            self.model = AutoModelForCausalLM.from_pretrained(
                'THUDM/cogvlm-chat-hf',
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            self.prompt_format = llava_visual_format
            self.split_patches = ["ASSISTANT:"]
        elif "vila1.5-8b" in args.model_name.lower():
            pass
        else:
            raise ValueError(f"Model {args.model_name} not supported")

        self.model.eval()
        self.model.to(self.device)
    
    @torch.inference_mode()
    def predict(self, context):
        text = context["text"]
        image = context["image"]

        format_text = self.prompt_format.format(system_prompt=self.prompt, user_input=text)

        inputs = self.processor(
            images=image, 
            text=format_text, 
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                **self.generate_config,
            )

        slen = inputs.input_ids.size(1)
        decoded = self.processor.batch_decode(
            outputs[:, slen:],
            skip_special_tokens=True
        )[0].strip()
        answer = decoded.split(self.split_patch)[-1].strip() if self.split_patch in decoded else decoded
        return answer
    
    @torch.inference_mode()
    def chat(self, contexts:list):
        batch_texts = [ctx["text"] for ctx in contexts]
        batch_images = [ctx["image"] for ctx in contexts]
        if "internvl" in self.args.model_name.lower():
            pixel_values_list = []
            num_patches_list = []
            for image in batch_images:
                pixel_values = process_internvl_image(image)
                pixel_values_list.append(pixel_values)
                num_patches_list.append(pixel_values.size(0))
            pixel_values = torch.cat(pixel_values_list, dim=0)  # 拼接成一个大的
            questions = batch_texts

            responses = self.model.batch_chat(
                self.tokenizer,
                pixel_values,
                num_patches_list=num_patches_list,
                questions=questions,
                generation_config=self.generate_config
            )
            
            answers = []
            for response in responses:
                response = output.strip()
                answers.append(response)
                print('The answer is:', answer)
            return answers

        else:
            try:
                messages = [[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": self.prompt},
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                                "min_pixels": 50176,  # 224x224
                                "max_pixels": 451584, # 672x672
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ] for text, image in zip(batch_texts, batch_images)]
                batch_format_texts = [
                    self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                    for message in messages
                ]
            except:
                batch_format_texts = [self.prompt_format.format(system_prompt=self.prompt, user_input=text) for text in batch_texts]

            if "qwen" in self.args.model_name.lower():
                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=batch_format_texts,
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)
            else:
                inputs = self.processor(
                    images=batch_images, 
                    text=batch_format_texts, 
                    return_tensors="pt",
                    padding="longest",
                    truncation=True
                ).to(self.model.device)

            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.generate_config,
                )
                
            answers = []
            for idx, output in enumerate(outputs):
                #slen = length.item() if torch.is_tensor(length) else length
                decoded = self.processor.batch_decode(
                    output[None, :],
                    skip_special_tokens=True
                )[0].strip()
                answer = decoded
                for split_token in self.split_patches:
                    if split_token in decoded:
                        answer = decoded.split(split_token)[-1].strip()
                        break  # 使用第一个匹配的分隔符

                answers.append(answer)
                print('The answer is:', answer)

        return answers

def build_transform(input_size):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    return transform

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = min(target_ratios, key=lambda ratio: abs(aspect_ratio - (ratio[0]/ratio[1])))

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    
    return processed_images

def process_internvl_image(loaded_image, input_size=448, max_num=12):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(loaded_image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values