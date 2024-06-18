import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
from torch.cuda.amp import autocast

class MistralModel:
    """
    A class used to load and run inference on a transformer-based causal language model with quantization support.

    Attributes
    ----------
    model_name : str
        The name or path of the pretrained model.
    device : torch.device
        The device on which to run the model (e.g., 'cuda' for GPU).
    tokenizer : AutoTokenizer
        The tokenizer associated with the model.
    stream : torch.cuda.Stream
        The CUDA stream for asynchronous operations.
    model : AutoModelForCausalLM
        The causal language model.
    is_instruct_model : bool
        Flag indicating if the model is an 'instruct' model.
    history : list
        A history of messages for chat-based models (only used if the model is an 'instruct' model).

    Methods
    -------
    warmup(prompts, warmup_steps=5)
        Runs warmup iterations to prime the model and GPU.
    run_inference(prompts, max_new_tokens=128)
        Runs inference on the given prompts and returns generated outputs and performance statistics.
    """
    def __init__(self, model_name: str, device: torch.device):
        """
        Initializes the MistralModel with the specified model and device, and sets up quantization if applicable.

        Parameters
        ----------
        model_name : str
            The name or path of the pretrained model.
        device : torch.device
            The device on which to run the model (e.g., 'cuda' for GPU).
        """
        # specify how to quantize the model
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.stream = torch.cuda.default_stream()
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quantization_config, device_map="auto"
        )
        self.is_instruct_model = "instruct" in model_name.lower()
        
        if self.is_instruct_model:
            self.history = []

    def warmup(self, prompts, warmup_steps=5):
        """
        Runs warmup iterations to prime the model and GPU.

        Parameters
        ----------
        prompts : list of str
            The input prompts to use for warmup.
        warmup_steps : int, optional
            The number of warmup iterations (default is 5).
        """
        # Tokenize the input prompts (batch)
        inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)
        
        # Warmup loop
        for _ in range(warmup_steps):
            with torch.cuda.stream(self.stream):
                with autocast():
                    self.model.generate(**inputs, max_new_tokens=64, do_sample=True)
            torch.cuda.synchronize()

    def run_inference(self, prompts, max_new_tokens = 128):
        """
        Runs inference on the given prompts and returns generated outputs and performance statistics.

        Parameters
        ----------
        prompts : list of str
            The input prompts to run inference on.
        max_new_tokens : int, optional
            The maximum number of new tokens to generate (default is 128).

        Returns
        -------
        dict
            A dictionary containing generated outputs, generated tokens, and performance statistics.
        """
        # Tokenize the input prompts (batch)
        if self.is_instruct_model:
            # Use chat template for instruct models
            self.history.append({"role":"user","content":prompts})
            inputs = self.tokenizer.apply_chat_template(self.history, return_tensors="pt").to(self.device)
        else:
            # Tokenize the input prompts (batch)
            inputs = self.tokenizer(prompts, return_tensors="pt").to(self.device)

        # Measure inference time
        start_time = time.time()

        if self.is_instruct_model:
            with torch.cuda.stream(self.stream):
                with autocast():
                    outputs = self.model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=True)    
        else:
            with torch.cuda.stream(self.stream):
                with autocast():
                    outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True)

        # Wait for the stream to complete
        torch.cuda.synchronize()

        end_time = time.time()

        # Decode the generated texts
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        output_tokens = torch.numel(outputs)
        #tokens_generated = input_tokens + output_tokens
        inference_time = end_time - start_time
        throughput = output_tokens/inference_time

        results = {"generated_outputs":generated_texts,
                   "generated_tokens":outputs,
                  "stats":{"total_tokens":output_tokens,
                          "inputs_tokens":torch.numel(inputs.input_ids),
                          "output_tokens":output_tokens-torch.numel(inputs.input_ids),
                          "inference_time":inference_time,
                          "throughput":throughput}}

        return results