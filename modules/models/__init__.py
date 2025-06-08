from .local import *
from .InstructBlip_consistency_regularization import *
from .InstructBlip_ad_attacks import *
from .InstructBlip_unified import *
from .LLaVA_consistency_regularization import *
from .LLaVA_ad_attacks import *
from .LLaVA_unified import *
try:
    from .Qwen_consistency_regularization import *
    from .Qwen_ad_attacks import *
    from .Qwen_unified import *
except:
    pass
from .peft_model import *
from .trainer import *

def get_model(args):
    local_models = ["blip", "llava", "vicuna", "qwen"]
    # for model in api_models:
    #     if model in args.model_name:
    #         return APIChat
    for model in local_models:
        if model in args.model_name.lower():
            return LocalModelChat

def get_model_from_checkpoint(args):
    local_models = ["blip", "llava", "vicuna", "qwen", "internvl"]
    # for model in api_models:
    #     if model in args.model_name:
    #         return APIChat
    for model in local_models:
        if model in args.model_name.lower():
            return LocalModelChat_FromCheckpoint