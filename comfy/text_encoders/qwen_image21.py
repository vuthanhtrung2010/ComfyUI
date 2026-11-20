import numbers

import torch
import torch.nn.functional as F

import comfy.model_management
import comfy.text_encoders.qwen3vl
from comfy import sd1_clip

VISION_BLOCK = "<|vision_start|><|image_pad|><|vision_end|>"
SYSTEM_PROMPT = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
T2I_TEMPLATE = SYSTEM_PROMPT + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"


class QwenImage21Tokenizer(comfy.text_encoders.qwen3vl.Qwen3VLTokenizer):
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        super().__init__(embedding_directory=embedding_directory, tokenizer_data=tokenizer_data, model_type="qwen3vl_8b")
        self.llama_template = T2I_TEMPLATE

    def tokenize_with_weights(self, text, return_word_ids=False, llama_template=None, images=[], prevent_empty_text=False, keep_vision=False, **kwargs):
        image = kwargs.get("image", None)
        if image is not None and len(images) == 0:
            images = [image[i:i + 1] for i in range(image.shape[0])]
        if llama_template is None and len(images) > 0:
            refs = " ".join("<image{}>{}".format(i + 1, VISION_BLOCK) for i in range(len(images)))
            llama_template = T2I_TEMPLATE.replace("{}", refs + "{}", 1)
        out = super().tokenize_with_weights(text, return_word_ids=return_word_ids, llama_template=llama_template, images=images, prevent_empty_text=prevent_empty_text, **kwargs)
        out["keep_vision"] = keep_vision
        return out


class QwenImage21Qwen3VLClipModel(comfy.text_encoders.qwen3vl.Qwen3VLClipModel):
    def __init__(self, device="cpu", dtype=None, attention_mask=True, model_options={}):
        super().__init__(device=device, dtype=dtype, attention_mask=attention_mask, model_options=model_options, model_type="qwen3vl_8b")
        # last layer without the final RMSNorm: transformers 4.57 hidden_states[-1], which Qwen's results are tuned to (5.x norms it)
        self.layer_norm_hidden_state = False
        self.image_spans = []

    def process_tokens(self, tokens, device):
        embeds, attention_mask, num_tokens, embeds_info = super().process_tokens(tokens, device)
        self.image_spans = [(e["index"], e["size"]) for e in embeds_info if e["type"] == "image"]
        if comfy.model_management.is_device_tpu(device):
            seq_len = embeds.shape[1]
            bucket_len = ((seq_len + 127) // 128) * 128
            if bucket_len > seq_len:
                pad_len = bucket_len - seq_len
                embeds = F.pad(embeds, (0, 0, 0, pad_len))
                attention_mask = F.pad(attention_mask, (0, pad_len))
        return embeds, attention_mask, num_tokens, embeds_info


class QwenImage21TEModel(sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen3vl_8b", clip_model=QwenImage21Qwen3VLClipModel, model_options=model_options)

    def encode_token_weights(self, token_weight_pairs):
        out, pooled, extra = super().encode_token_weights(token_weight_pairs)
        tokens = [t[0] for t in token_weight_pairs["qwen3vl_8b"][0]]
        image_spans = getattr(self, self.clip).image_spans

        # drop the system turn, everything before the second <|im_start|>; positions shift by each image expanded before it
        im_starts, offset, spans = [], 0, iter(image_spans)
        for i, t in enumerate(tokens):
            if isinstance(t, numbers.Integral):
                if t == 151644:
                    im_starts.append(i + offset)
            elif isinstance(t, dict) and t.get("type") == "image":  # a textual embedding is a bare tensor and has no span
                offset += next(spans, (0, 1))[1] - 1
        keep = torch.ones(out.shape[1], dtype=torch.bool)
        keep[:im_starts[1] if len(im_starts) > 1 else 0] = False
        if "attention_mask" in extra:
            keep &= extra["attention_mask"][0].to(device="cpu", dtype=torch.bool)

        # vision tokens are replaced by reference latents in the DiT: drop them and record where each image goes
        # with no latents coming (no vae) they stay, and the image conditions through the text encoder alone
        slots = []
        if not token_weight_pairs.get("keep_vision", False):
            for start, size in image_spans:
                keep[start:start + size] = False
                slots.append(int(keep[:start].sum()))

        out = out[:, keep.to(out.device)]
        extra["attention_mask"] = extra["attention_mask"][:, keep.to(extra["attention_mask"].device)]
        if extra["attention_mask"].sum() == torch.numel(extra["attention_mask"]):
            extra.pop("attention_mask")
        if len(slots) > 0:
            extra["image_slots"] = slots
        return out, pooled, extra

    def encode_token_weights_batch(self, token_weight_pairs_list):
        if len(token_weight_pairs_list) == 1:
            return self.encode_token_weights(token_weight_pairs_list[0])

        to_encode = []
        can_batch = True
        for twp in token_weight_pairs_list:
            sections = twp.get(self.clip_name, [])
            if len(sections) != 1 or twp.get("keep_vision", False):
                can_batch = False
                break
            seq = sections[0]
            if any(not isinstance(t[0], numbers.Integral) or t[1] != 1.0 for t in seq):
                can_batch = False
                break
            to_encode.append([t[0] for t in seq])

        if not can_batch:
            rows, pooled_out = [], None
            for twp in token_weight_pairs_list:
                o, p, _ = self.encode_token_weights(twp)
                rows.append(o[0])
                if pooled_out is None:
                    pooled_out = p
            max_len = max(r.shape[0] for r in rows)
            out = torch.zeros((len(rows), max_len, rows[0].shape[-1]), dtype=rows[0].dtype, device=rows[0].device)
            mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=rows[0].device)
            for idx, r in enumerate(rows):
                out[idx, :r.shape[0]] = r
                mask[idx, :r.shape[0]] = 1
            extra = {"attention_mask": mask} if mask.sum() != mask.numel() else {}
            return out, pooled_out, extra

        clip_model = getattr(self, self.clip)
        res = clip_model.encode(to_encode)
        raw_out, pooled = res[:2]
        raw_mask = res[2].get("attention_mask", None) if len(res) > 2 else None
        out_dev = comfy.model_management.intermediate_device()
        raw_out = raw_out.to(device=out_dev)
        if raw_mask is not None:
            raw_mask = raw_mask.to(device=out_dev, dtype=torch.bool)
        if pooled is not None:
            pooled = pooled.to(device=out_dev)

        rows = []
        for idx, tokens in enumerate(to_encode):
            im_starts = [i for i, t in enumerate(tokens) if t == 151644]
            keep = torch.ones(raw_out.shape[1], dtype=torch.bool, device=out_dev)
            keep[:im_starts[1] if len(im_starts) > 1 else 0] = False
            if raw_mask is not None:
                keep &= raw_mask[idx]
            rows.append(raw_out[idx, keep])

        max_len = max(r.shape[0] for r in rows)
        batched = torch.zeros((len(rows), max_len, raw_out.shape[-1]), dtype=raw_out.dtype, device=out_dev)
        mask = torch.zeros((len(rows), max_len), dtype=torch.long, device=out_dev)
        for idx, r in enumerate(rows):
            batched[idx, :r.shape[0]] = r
            mask[idx, :r.shape[0]] = 1
        extra = {"attention_mask": mask} if mask.sum() != mask.numel() else {}
        return batched, pooled, extra


def te(dtype_llama=None, llama_quantization_metadata=None):
    class QwenImage21TEModel_(QwenImage21TEModel):
        def __init__(self, device="cpu", dtype=None, model_options={}):
            if dtype_llama is not None:
                target_device = model_options.get("load_device", comfy.model_management.text_encoder_device())
                dtype = comfy.model_management.pick_weight_dtype(dtype_llama, dtype, target_device)
            if llama_quantization_metadata is not None:
                model_options = model_options.copy()
                model_options["quantization_metadata"] = llama_quantization_metadata
            super().__init__(device=device, dtype=dtype, model_options=model_options)
    return QwenImage21TEModel_
