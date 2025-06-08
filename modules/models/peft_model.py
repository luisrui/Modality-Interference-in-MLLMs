from peft import PeftModel

class PeftModelForLLaVAConsistency(PeftModel):
    def __init__(
        self, 
        model, 
        peft_config, 
        adapter_name: str = "default", 
        **kwargs
    ) -> None:
        super().__init__(model, peft_config, adapter_name=adapter_name, **kwargs)

    def __call__(self, *args, **kwargs):
        ### Ensure that the model is called with origin and augmented samples forward
        return self.forward(*args, **kwargs)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        pixel_values=None,
        labels=None,
        **kwargs,
    ):
        if "origin" in kwargs and "augmented" in kwargs:
            return self.base_model(
                origin=kwargs["origin"],
                augmented=kwargs["augmented"],
                return_dict=True,
            )
        else:
            origin = {
                "input_ids": input_ids,
                "pixel_values": pixel_values,
                "attention_mask": attention_mask,
                "labels": labels,
            }
            return self.base_model(origin=origin, return_dict=True)

