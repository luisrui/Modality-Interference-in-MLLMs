from .text_interfere import *
from .img_interfere import *

llava_visual_format = """{system_prompt}\nUSER: <image>\n{user_input}\nASSISTANT:"""
cogvlm_format =  """{system_prompt}\nUSER:{user_input}\nASSISTANT:"""
llava_visual_format_ICL = """{system_prompt}\n{icl_context}USER: <image>\n{user_input}\nASSISTANT:"""
llava_visual_format_ICL_reversed = """{system_prompt}\n<image>\n{icl_context}USER: {user_input}\nASSISTANT:"""  
llava_visual_format_train = """{system_prompt}\nUSER: <image>\n{user_input}\nASSISTANT:{answer}"""
llava_example_format = "Example:\nUSER: <image>\n{text}ASSISTANT: {answer}\n"


# blip_visual_format = """{system_prompt}\n{user_input}Answer:"""
# blip_visual_format_train = """{system_prompt}\n{user_input}Answer:{answer}"""
blip_visual_format = """{system_prompt}\n{user_input} <answer> """
blip_visual_format_train = """{system_prompt}\n{user_input} <answer> {answer}"""
blip_visual_format_ICL = """{system_prompt}\n{icl_context}{user_input}Answer:"""
blip_example_format = "Example:\n{text}Answer: {answer}\n"

text_format = """{system_prompt}\nUSER: {user_input}\nASSISTANT:"""

qa_prompt = """You are an expert at question answering. Given the question, please output the answer. No explanation and further question.""" # Mind that you have to give an answer, do not reject answering the question.
qa_context_prompt = """You are an expert at question answering. Given the question and the context of the question, please output the answer. Please provide a concise option, no explanation and further question."""
qa_image_prompt = """You are an expert at question answering. Given the question and the context image of the question, please output the answer. Please provide a concise option, no explanation and further question."""
qa_blend_prompt = """You are an expert at question answering. Given the question, the context of the question and the context image of the question, please output the answer. Please provide a concise option, no explanation and further question."""

qa_vqa_prompt_multi_choices_train = """You are an expert at question answering. Given the question and the context image of the question, please output the answer. Please provide a concise answer, no explanation and further question."""
