# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors import safe_open
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None


def _resolve_dtype(name: str) -> torch.dtype:
    v = (name or "").strip().lower()
    if v in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if v in {"fp16", "float16", "half"}:
        return torch.float16
    if v in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {name}")


def _resolve_attn(name: str) -> str | None:
    v = (name or "").strip()
    if not v or v.lower() in {"auto", "none", "null"}:
        return None
    return v


def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    parser.add_argument("--speaker_id", type=int, default=3000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--subtalker_loss_weight", type=float, default=0.3)
    parser.add_argument("--attn_implementation", type=str, default="auto")
    parser.add_argument("--torch_dtype", type=str, default="bfloat16")
    parser.add_argument("--log_every_n_steps", type=int, default=10)
    parser.add_argument("--save_every_n_epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speaker_encoder_model_path", type=str, default="",
                        help="Separate model path to load speaker_encoder from, "
                             "when init_model_path lacks one (e.g. CustomVoice models).")
    args = parser.parse_args()

    if args.gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be >= 1")
    if args.log_every_n_steps <= 0:
        raise ValueError("log_every_n_steps must be >= 1")
    if args.save_every_n_epochs <= 0:
        raise ValueError("save_every_n_epochs must be >= 1")
    if args.max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be > 0")
    if args.subtalker_loss_weight < 0:
        raise ValueError("subtalker_loss_weight must be >= 0")

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    accelerator = Accelerator(
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        mixed_precision=str(args.mixed_precision),
        log_with="tensorboard",
    )

    init_model_raw = str(args.init_model_path).strip()
    model_path = Path(init_model_raw).expanduser()
    is_local_style = init_model_raw.startswith(("/", "./", "../", "~"))
    if model_path.exists():
        MODEL_PATH = str(model_path.resolve())
    elif is_local_style:
        raise FileNotFoundError(f"init_model_path not found: {model_path}")
    else:
        # Support HuggingFace repo id (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base).
        from huggingface_hub import snapshot_download

        try:
            MODEL_PATH = snapshot_download(repo_id=init_model_raw)
        except Exception:
            # If a bad/expired token is configured in the environment, public repos can fail.
            MODEL_PATH = snapshot_download(repo_id=init_model_raw, token=False)

    torch_dtype = _resolve_dtype(args.torch_dtype)
    attn = _resolve_attn(args.attn_implementation)

    qwen3tts_kwargs = {"torch_dtype": torch_dtype}
    if attn is not None:
        qwen3tts_kwargs["attn_implementation"] = attn

    qwen3tts = Qwen3TTSModel.from_pretrained(MODEL_PATH, **qwen3tts_kwargs)
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(
        qwen3tts.model.parameters(),
        lr=args.lr,
        weight_decay=float(args.weight_decay),
    )

    # ------------------------------------------------------------------
    # Resolve speaker_encoder: if the main model lacks one (e.g. CustomVoice),
    # optionally load it from a separate Base model.
    # ------------------------------------------------------------------
    _speaker_encoder_fn = getattr(qwen3tts.model, "speaker_encoder", None)
    _se_aux_model = None  # keep alive while needed
    if _speaker_encoder_fn is None or not callable(_speaker_encoder_fn):
        se_path = (args.speaker_encoder_model_path or "").strip()
        if not se_path:
            # Auto-detect: swap CustomVoice → Base variant for speaker_encoder
            se_path = init_model_raw.replace("-CustomVoice", "-Base")
            if se_path == init_model_raw:
                raise RuntimeError(
                    "Model has no speaker_encoder and --speaker_encoder_model_path was not provided."
                )
            accelerator.print(f"[INFO] Auto-loading speaker_encoder from: {se_path}")
        se_model_path = Path(se_path).expanduser()
        if se_model_path.exists():
            se_resolved = str(se_model_path.resolve())
        else:
            from huggingface_hub import snapshot_download
            try:
                se_resolved = snapshot_download(repo_id=se_path)
            except Exception:
                se_resolved = snapshot_download(repo_id=se_path, token=False)
        _se_aux_model = Qwen3TTSModel.from_pretrained(se_resolved, torch_dtype=torch_dtype)
        _speaker_encoder_fn = _se_aux_model.model.speaker_encoder
        if _speaker_encoder_fn is None or not callable(_speaker_encoder_fn):
            raise RuntimeError(f"speaker_encoder still None after loading {se_resolved}")
        accelerator.print("[INFO] speaker_encoder loaded from auxiliary model.")
    # ------------------------------------------------------------------

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    max_steps = int(args.max_steps)
    global_step = 0

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                # Use the resolved speaker_encoder (may be from aux model).
                # If the encoder lives on a different device, move input to it,
                # then move the result back to the training device.
                _train_device = next(iter(model.parameters())).device
                _train_dtype = next(iter(model.parameters())).dtype
                try:
                    _se_param = next(_speaker_encoder_fn.parameters())
                    _se_device = _se_param.device
                    _se_dtype = _se_param.dtype
                except StopIteration:
                    _se_device = _train_device
                    _se_dtype = _train_dtype
                speaker_embedding = _speaker_encoder_fn(
                    ref_mels.to(_se_device).to(_se_dtype)
                ).detach().to(_train_device).to(_train_dtype)
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                # NOTE: 0.6B models can have mismatched embedding dims (e.g. text=2048, codec=1024).
                # Use the model's built-in resize projection when present so finetuning works for both 1.7B and 0.6B.
                raw_text_embedding = model.talker.model.text_embedding(input_text_ids)
                raw_codec_embedding = model.talker.model.codec_embedding(input_codec_ids)

                if raw_text_embedding.shape[-1] != raw_codec_embedding.shape[-1]:
                    if hasattr(model.talker, "text_projection"):
                        raw_text_embedding = model.talker.text_projection(raw_text_embedding)
                    elif hasattr(model.talker, "code_predictor") and hasattr(
                        model.talker.code_predictor, "small_to_mtp_projection"
                    ):
                        raw_codec_embedding = model.talker.code_predictor.small_to_mtp_projection(
                            raw_codec_embedding
                        )
                    else:
                        raise RuntimeError(
                            "Embedding size mismatch and no projection found: "
                            f"text={tuple(raw_text_embedding.shape)} codec={tuple(raw_codec_embedding.shape)}"
                        )

                input_text_embedding = raw_text_embedding * text_embedding_mask
                input_codec_embedding = raw_codec_embedding * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + float(args.subtalker_loss_weight) * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))

                optimizer.step()
                optimizer.zero_grad()

            if step % int(args.log_every_n_steps) == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

            global_step += 1
            if max_steps > 0 and global_step >= max_steps:
                accelerator.print(f"Reached max_steps={max_steps}. Stopping early.")
                break

        if accelerator.is_main_process and (
            ((epoch + 1) % int(args.save_every_n_epochs) == 0) or (epoch + 1 == num_epochs) or (max_steps > 0 and global_step >= max_steps)
        ):
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: int(args.speaker_id)
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][int(args.speaker_id)] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            fallback_path = os.path.join(output_dir, "pytorch_model.bin")
            saved_with_safetensors = False
            try:
                save_file(state_dict, save_path)
                # Guard against partial/corrupt safetensors files (e.g. interrupted writes).
                with safe_open(save_path, framework="pt") as f:
                    _ = f.keys()
                saved_with_safetensors = True
            except Exception as e:
                accelerator.print(
                    f"[WARN] Failed to save `{save_path}` via safetensors: {e}. "
                    f"Falling back to `{fallback_path}`."
                )
                try:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                except Exception:
                    pass

            if not saved_with_safetensors:
                torch.save(state_dict, fallback_path)

            if not os.path.exists(save_path) and not os.path.exists(fallback_path):
                raise RuntimeError(
                    "Checkpoint weights were not written. Expected one of: "
                    f"`{save_path}` or `{fallback_path}`."
                )

        if max_steps > 0 and global_step >= max_steps:
            break

if __name__ == "__main__":
    train()
