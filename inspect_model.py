from transformers import MoshiForConditionalGeneration

model = MoshiForConditionalGeneration.from_pretrained("kmhf/hf-moshiko")

for name, module in model.named_modules():
    if 'linear' in name.lower() or 'proj' in name.lower():
        print(f"{name}: {type(module)}")