import types
import torch
import logging

from comfy.cli_args import args


try:
    import comfy_kitchen as ck
    from comfy_kitchen.backends.eager.quantization import _build_hadamard, _rotate_activation
    from comfy_kitchen.tensor import (
        QuantizedTensor,
        QuantizedLayout,
        TensorCoreFP8Layout as _CKFp8Layout,
        TensorCoreNVFP4Layout as _CKNvfp4Layout,
        TensorCoreConvRotW4A4Layout as _CKTensorCoreConvRotW4A4Layout,
        TensorWiseINT8Layout as _CKTensorWiseINT8Layout,
        AsymW4A8Int8Layout as _CKAsymW4A8Int8Layout,
        register_layout_op,
        register_layout_class,
        get_layout_class,
    )
    _CK_AVAILABLE = True
    if torch.version.cuda is None:
        ck.registry.disable("cuda")
    else:
        cuda_version = tuple(map(int, str(torch.version.cuda).split('.')))
        if cuda_version < (13,):
            ck.registry.disable("cuda")
            logging.warning("WARNING: You need pytorch with cu130 or higher to use optimized CUDA operations.\nWARNING WARNING WARNING\nIf you are on nvidia 20 series and above it is required that you update your pytorch to cu130 or higher.\n")

    # comfy-kitchen picks its accelerated backend on import: the HIP backend registers
    # itself on a supported AMD device and takes dispatch priority there, CUDA on NVIDIA.
    # Triton is an opt-in override, off by default on every platform.
    if args.enable_triton_backend and not args.disable_triton_backend:
        try:
            import triton
            logging.info("Found triton %s. Enabling comfy-kitchen triton backend.", triton.__version__)
        except ImportError as e:
            logging.error(f"Failed to import triton, Error: {e}, the comfy-kitchen triton backend will not be available.")
            ck.registry.disable("triton")
    else:
        ck.registry.disable("triton")
    for k, v in ck.list_backends().items():
        logging.info(f"Found comfy_kitchen backend {k}: {v}")
except ImportError as e:
    logging.error(f"Failed to import comfy_kitchen, Error: {e}, fp8 and fp4 support will not be available.")
    _CK_AVAILABLE = False

    class QuantizedTensor:
        pass

    class _CKFp8Layout:
        pass

    class _CKNvfp4Layout:
        pass

    class _CKTensorWiseINT8Layout:
        pass

    class _CKTensorCoreConvRotW4A4Layout:
        pass

    class _CKAsymW4A8Int8Layout:
        pass

    def register_layout_class(name, cls):
        pass

    def get_layout_class(name):
        return None

    _build_hadamard = None
    _rotate_activation = None

_CK_MXFP8_AVAILABLE = False
if _CK_AVAILABLE:
    try:
        from comfy_kitchen.tensor import TensorCoreMXFP8Layout as _CKMxfp8Layout
        _CK_MXFP8_AVAILABLE = True
    except ImportError:
        logging.warning("comfy_kitchen does not support MXFP8, please update comfy_kitchen.")

if not _CK_MXFP8_AVAILABLE:
    class _CKMxfp8Layout:
        pass

import comfy.float

# ==============================================================================
# FP8 Layouts with Comfy-Specific Extensions
# ==============================================================================

class _TensorCoreFP8LayoutBase(_CKFp8Layout):
    FP8_DTYPE = None  # Must be overridden in subclass

    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if cls.FP8_DTYPE is None:
            raise NotImplementedError(f"{cls.__name__} must define FP8_DTYPE")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if isinstance(scale, str) and scale == "recalculate":
            scale = torch.amax(tensor.abs()).to(dtype=torch.float32) / torch.finfo(cls.FP8_DTYPE).max
            if tensor.dtype not in [torch.float32, torch.bfloat16]:  # Prevent scale from being too small
                tensor_info = torch.finfo(tensor.dtype)
                scale = (1.0 / torch.clamp((1.0 / scale), min=tensor_info.min, max=tensor_info.max))

        if scale is None:
            scale = torch.ones((), device=tensor.device, dtype=torch.float32)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, device=tensor.device, dtype=torch.float32)

        if stochastic_rounding > 0:
            if inplace_ops:
                tensor *= (1.0 / scale).to(tensor.dtype)
            else:
                tensor = tensor * (1.0 / scale).to(tensor.dtype)
            qdata = comfy.float.stochastic_rounding(tensor, dtype=cls.FP8_DTYPE, seed=stochastic_rounding)
        else:
            qdata = ck.quantize_per_tensor_fp8(tensor, scale, cls.FP8_DTYPE)

        params = cls.Params(scale=scale.float(), orig_dtype=orig_dtype, orig_shape=orig_shape)
        return qdata, params


class TensorCoreMXFP8Layout(_CKMxfp8Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"MXFP8 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = comfy.float.stochastic_round_quantize_mxfp8_by_block(tensor, pad_32x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_mxfp8(tensor, pad_32x=needs_padding)

        params = cls.Params(
            scale=block_scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
        )
        return qdata, params


class TensorCoreNVFP4Layout(_CKNvfp4Layout):
    @classmethod
    def quantize(cls, tensor, scale=None, stochastic_rounding=0, inplace_ops=False):
        if tensor.dim() != 2:
            raise ValueError(f"NVFP4 requires 2D tensor, got {tensor.dim()}D")

        orig_dtype = tensor.dtype
        orig_shape = tuple(tensor.shape)

        if scale is None or (isinstance(scale, str) and scale == "recalculate"):
            scale = torch.amax(tensor.abs()) / (ck.float_utils.F8_E4M3_MAX * ck.float_utils.F4_E2M1_MAX)

        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        scale = scale.to(device=tensor.device, dtype=torch.float32)

        padded_shape = cls.get_padded_shape(orig_shape)
        needs_padding = padded_shape != orig_shape

        if stochastic_rounding > 0:
            qdata, block_scale = comfy.float.stochastic_round_quantize_nvfp4_by_block(tensor, scale, pad_16x=needs_padding, seed=stochastic_rounding)
        else:
            qdata, block_scale = ck.quantize_nvfp4(tensor, scale, pad_16x=needs_padding)

        params = cls.Params(
            scale=scale,
            orig_dtype=orig_dtype,
            orig_shape=orig_shape,
            block_scale=block_scale,
        )
        return qdata, params


class TensorCoreFP8E4M3Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e4m3fn


class TensorCoreFP8E5M2Layout(_TensorCoreFP8LayoutBase):
    FP8_DTYPE = torch.float8_e5m2


# Backward compatibility alias - default to E4M3
TensorCoreFP8Layout = TensorCoreFP8E4M3Layout
TensorWiseINT8Layout = _CKTensorWiseINT8Layout
TensorCoreConvRotW4A4Layout = _CKTensorCoreConvRotW4A4Layout
AsymW4A8Int8Layout = _CKAsymW4A8Int8Layout

# ==============================================================================
# Registry
# ==============================================================================

register_layout_class("TensorCoreFP8Layout", TensorCoreFP8Layout)
register_layout_class("TensorCoreFP8E4M3Layout", TensorCoreFP8E4M3Layout)
register_layout_class("TensorCoreFP8E5M2Layout", TensorCoreFP8E5M2Layout)
register_layout_class("TensorCoreNVFP4Layout", TensorCoreNVFP4Layout)
register_layout_class("TensorWiseINT8Layout", _CKTensorWiseINT8Layout)
register_layout_class("TensorCoreConvRotW4A4Layout", _CKTensorCoreConvRotW4A4Layout)
if _CK_MXFP8_AVAILABLE:
    register_layout_class("TensorCoreMXFP8Layout", TensorCoreMXFP8Layout)
register_layout_class("AsymW4A8Int8Layout", _CKAsymW4A8Int8Layout)

QUANT_ALGOS = {
    "float8_e4m3fn": {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E4M3Layout",
    },
    "float8_e5m2": {
        "storage_t": torch.float8_e5m2,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreFP8E5M2Layout",
    },
    "nvfp4": {
        "storage_t": torch.uint8,
        "parameters": {"weight_scale", "weight_scale_2", "input_scale", "pre_quant_scale"},
        "comfy_tensor_layout": "TensorCoreNVFP4Layout",
        "group_size": 16,
    },
}

if _CK_MXFP8_AVAILABLE:
    QUANT_ALGOS["mxfp8"] = {
        "storage_t": torch.float8_e4m3fn,
        "parameters": {"weight_scale", "input_scale"},
        "comfy_tensor_layout": "TensorCoreMXFP8Layout",
        "group_size": 32,
    }

QUANT_ALGOS["int8_tensorwise"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "TensorWiseINT8Layout",
    "quantize_input": False,
}

QUANT_ALGOS["convrot_w4a4"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "TensorCoreConvRotW4A4Layout",
    "quantize_input": False,
}

QUANT_ALGOS["asym_w4a8_int8"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "AsymW4A8Int8Layout",
    "quantize_input": False,
}

# Same layout class and kernels, 6-bit uniform codes: weight is int8 [N, 3K/4], no codebook.
QUANT_ALGOS["w6a8_int8"] = {
    "storage_t": torch.int8,
    "parameters": {"weight_scale"},
    "comfy_tensor_layout": "AsymW4A8Int8Layout",
    "quantize_input": False,
}


def _int8_w8a16_convrot_bf16(x, qdata, scale, h, gs):
    xr = torch.matmul(x.reshape(-1, x.shape[-1] // gs, gs), h).reshape(x.shape)
    return torch.nn.functional.linear(xr, qdata.to(dtype=x.dtype)) * scale.to(dtype=x.dtype).reshape(-1)


def _int8_w8a16_bf16(x, qdata, scale):
    return torch.nn.functional.linear(x, qdata.to(dtype=x.dtype)) * scale.to(dtype=x.dtype).reshape(-1)


_TPU_COMPILED_W8A16_CACHE = {}
_TPU_BACKEND_AVAILABLE = None
_TPU_COMPILED_MAX_SHAPES = 128


def _tpu_backend_available():
    global _TPU_BACKEND_AVAILABLE
    if _TPU_BACKEND_AVAILABLE is None:
        _TPU_BACKEND_AVAILABLE = hasattr(torch, "_dynamo") and "tpu" in torch._dynamo.list_backends()
    return _TPU_BACKEND_AVAILABLE


def _get_tpu_compiled_w8a16_fn(convrot, x_shape, qdata_shape, gs):
    if not _tpu_backend_available():
        return None
    key = (convrot, tuple(x_shape), tuple(qdata_shape), gs)
    fn = _TPU_COMPILED_W8A16_CACHE.get(key)
    if fn is not None:
        return fn
    if len(_TPU_COMPILED_W8A16_CACHE) >= _TPU_COMPILED_MAX_SHAPES:
        return None
    base = _int8_w8a16_convrot_bf16 if convrot else _int8_w8a16_bf16
    tag = f"{base.__name__}_{len(_TPU_COMPILED_W8A16_CACHE)}"
    unique_fn = types.FunctionType(base.__code__.replace(co_name=tag), base.__globals__)
    compiled = torch.compile(unique_fn, backend="tpu")
    _TPU_COMPILED_W8A16_CACHE[key] = compiled
    return compiled


def int8_tensorwise_linear(x, weight, bias=None):
    """Execute full-precision linear on INT8 tensorwise weights by rotating activations (ConvRot) and scaling after F.linear."""
    params = weight._params
    qdata = weight._qdata
    convrot = getattr(params, "convrot", False)
    if x.dtype == torch.bfloat16:
        scale = params.scale if params.scale.device == x.device else params.scale.to(device=x.device)
        gs = params.convrot_groupsize if convrot else 0
        compiled = _get_tpu_compiled_w8a16_fn(convrot, x.shape, qdata.shape, gs) if x.device.type == "tpu" else None
        if convrot:
            h = _build_hadamard(gs, device=x.device, dtype=x.dtype)
            if compiled is not None:
                out = compiled(x, qdata, scale, h, gs)
            else:
                out = _int8_w8a16_convrot_bf16(x, qdata, scale, h, gs)
        else:
            if compiled is not None:
                out = compiled(x, qdata, scale)
            else:
                out = _int8_w8a16_bf16(x, qdata, scale)
    else:
        if convrot:
            gs = params.convrot_groupsize
            h = _build_hadamard(gs, device=x.device, dtype=x.dtype)
            x = _rotate_activation(x, h, gs)
        out = torch.nn.functional.linear(x, qdata.to(dtype=x.dtype))
        scale = params.scale.to(device=out.device, dtype=torch.float32).reshape(-1)
        out = (out.float() * scale).to(dtype=x.dtype)
    if bias is not None:
        out = out + bias
    return out


# ==============================================================================
# Re-exports for backward compatibility
# ==============================================================================

__all__ = [
    "QuantizedTensor",
    "QuantizedLayout",
    "TensorCoreFP8Layout",
    "TensorCoreFP8E4M3Layout",
    "TensorCoreFP8E5M2Layout",
    "TensorCoreNVFP4Layout",
    "TensorCoreConvRotW4A4Layout",
    "TensorWiseINT8Layout",
    "AsymW4A8Int8Layout",
    "QUANT_ALGOS",
    "register_layout_op",
    "int8_tensorwise_linear",
]
