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

from qwen_tts import Qwen3TTSTokenizer

BATCH_INFER_NUM = 32

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--tokenizer_model_path", type=str, default="Qwen/Qwen3-TTS-Tokenizer-12Hz")
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_jsonl", type=str, required=True)
    parser.add_argument("--batch_infer_num", type=int, default=BATCH_INFER_NUM)
    args = parser.parse_args()

    tokenizer_12hz = Qwen3TTSTokenizer.from_pretrained(
        args.tokenizer_model_path,
        device_map=args.device,
    )

    batch_size = int(args.batch_infer_num) if int(args.batch_infer_num) > 0 else BATCH_INFER_NUM
    batch_lines = []
    batch_audios = []
    processed = 0

    with open(args.input_jsonl, "r", encoding="utf-8") as fin, open(
        args.output_jsonl, "w", encoding="utf-8"
    ) as fout:
        for raw in fin:
            raw = raw.strip()
            if not raw:
                continue
            line = json.loads(raw)

            batch_lines.append(line)
            batch_audios.append(line["audio"])

            if len(batch_lines) >= batch_size:
                enc_res = tokenizer_12hz.encode(batch_audios)
                for code, obj in zip(enc_res.audio_codes, batch_lines):
                    obj["audio_codes"] = code.cpu().tolist()
                    fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                processed += len(batch_lines)
                if processed % 1000 == 0:
                    print(f"Encoded {processed} audio files...")
                batch_lines.clear()
                batch_audios.clear()

        if len(batch_audios) > 0:
            enc_res = tokenizer_12hz.encode(batch_audios)
            for code, obj in zip(enc_res.audio_codes, batch_lines):
                obj["audio_codes"] = code.cpu().tolist()
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
            processed += len(batch_lines)

    print(f"Done. Encoded {processed} audio files.")

if __name__ == "__main__":
    main()
