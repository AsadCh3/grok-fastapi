from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
from model import LanguageModelConfig, TransformerConfig, QuantizedWeight8bit as QW8Bit
from runners import InferenceRunner, ModelRunner, sample_from_model

app = FastAPI()

CKPT_PATH = "./checkpoints/"

class PromptRequest(BaseModel):
    prompt: str
    max_len: int = 100
    temperature: float = 0.01

grok_1_model = LanguageModelConfig(
    vocab_size=128 * 1024,
    pad_token=0,
    eos_token=2,
    sequence_len=8192,
    embedding_init_scale=1.0,
    output_multiplier_scale=0.5773502691896257,
    embedding_multiplier_scale=78.38367176906169,
    model=TransformerConfig(
        emb_size=48 * 128,
        widening_factor=8,
        key_size=128,
        num_q_heads=48,
        num_kv_heads=8,
        num_layers=64,
        attn_output_multiplier=0.08838834764831845,
        shard_activations=True,
        num_experts=8,
        num_selected_experts=2,
        data_axis="data",
        model_axis="model",
    ),
)
inference_runner = InferenceRunner(
    pad_sizes=(1024,),
    runner=ModelRunner(
        model=grok_1_model,
        bs_per_device=0.125,
        checkpoint_path=CKPT_PATH,
    ),
    name="local",
    load=CKPT_PATH,
    tokenizer_path="./tokenizer.model",
    local_mesh_config=(1, 8),
    between_hosts_config=(1, 1),
)

@app.on_event("startup")
async def startup_event():
    logging.basicConfig(level=logging.INFO)
    inference_runner.initialize()

@app.post("/generate/")
async def generate_text(prompt_request: PromptRequest):
    try:
        gen = inference_runner.run()
        output = sample_from_model(gen, prompt_request.prompt, max_len=prompt_request.max_len, temperature=prompt_request.temperature)
        return {"prompt": prompt_request.prompt, "output": output}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
