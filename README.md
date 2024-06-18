# Simplismart Task

## Overview
The `MistralModel` class is designed to load and run inference on a transformer-based causal language model. The class supports model quantization to 4-bit precision for efficient memory usage and inference speed. Additionally, it can detect and handle "instruct" models by using a chat template for tokenization and inference.

## Requirements
- `torch`
- `transformers`
- `bitsandbytes`
- `flash-attn`

### Installing Required Packages
To install the required packages, you can use the following commands:

```bash
pip install torch transformers bitsandbytes flash-attn --no-build-isolation
```

## Example Usage

### Normal Usage
```python
from mistral import MistralModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "mistralai/Mistral-7B-v0.1"

model_class = MistralModel(model_name,DEVICE)

prompt = "I am model"

results = model_class.run_inference(prompt)
```
Output:
```
{'generated_outputs': ['I am model obsessed.\n\nMy phone has photos of a tinted mauve blush, 700 lipsticks and a bajillion foundation and concealer swatches. And after seeing Miley Cyrus out and about a few months ago wearing what was seemingly a smudged kohl eyeliner, I thought, "Huh, I want to try that, and look really cool." In reality, I attempted to recreate the \'90s look, but the outcome was much more 2007 Lilo and instead 1991-present, me. I did,'],
 'generated_tokens': tensor([[    1,   315,   837,  2229, 14848,  9740, 28723,    13,    13,  5183,
           4126,   659,  8886,   302,   264,   261, 27700,   290,   581,   333,
            843,  1426, 28725, 28705, 28787, 28734, 28734, 11144,   303,  5446,
            304,   264,   287,  1150, 23202, 13865,   304, 13270, 16351,  1719,
            270,  1927, 28723,  1015,  1024,  6252,   351, 13891, 26832,   381,
            575,   304,   684,   264,  1664,  3370,  3584,  8192,   767,   403,
          20143,   264,   991,   554,  2560,   446, 16674,  1746,   301,  4828,
          28725,   315,  1654, 28725,   345, 28769,  8884, 28725,   315,   947,
            298,  1464,   369, 28725,   304,   913,  1528,  5106,   611,   560,
           6940, 28725,   315, 15335,   298,   937,  2730,   272,   464, 28774,
          28734, 28713,   913, 28725,   562,   272, 14120,   403,  1188,   680,
          28705, 28750, 28734, 28734, 28787,   393, 10630,   304,  3519, 28705,
          28740, 28774, 28774, 28740, 28733, 12497, 28725,   528, 28723,   315,
            863, 28725]], device='cuda:0'),
 'stats': {'total_tokens': 132,
  'inputs_tokens': 4,
  'output_tokens': 128,
  'inference_time': 7.606522083282471,
  'throughput': 17.353528794731055}}
```
### Concurrent Usage
```python
prompts = [prompt]*32
results = model_class.run_inference(prompts)
```
Output (stats):
```
{'total_tokens': 4224,
 'inputs_tokens': 128,
 'output_tokens': 4096,
 'inference_time': 29.28552746772766,
 'throughput': 144.23506643869752}
```