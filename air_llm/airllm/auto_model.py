import importlib
from sys import platform

from transformers import AutoConfig

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from .airllm_llama_mlx import AirLLMLlamaMlx
    from .airllm_qwen35_mlx_fast import AirLLMQwen35Mlx

# Architectures that need a dedicated AirLLM subclass because of a non-standard module layout
# (custom remote-code models). Everything else uses the generic AirLLMBaseModel, which streams any
# standard *ForCausalLM (model.model.layers + lm_head / norm) and lets transformers own the
# forward pass, so newly released architectures work without code changes.
ARCH_OVERRIDES = {
    "ChatGLMModel": "AirLLMChatGLM",
    "ChatGLMForConditionalGeneration": "AirLLMChatGLM",
    "QWenLMHeadModel": "AirLLMQWen",
    "BaichuanForCausalLM": "AirLLMBaichuan",
    "BaiChuanForCausalLM": "AirLLMBaichuan",
    "InternLMForCausalLM": "AirLLMInternLM",
    "KimiK3ForConditionalGeneration": "AirLLMKimiK3",
}

# Qwen3.8 is branded as Qwen3.8 but the released checkpoints intentionally use the
# Qwen3.5-family model classes/configuration internally. MLX-LM implements that family as qwen3_5.
MAC_QWEN35_ARCHITECTURES = {
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5ForCausalLM",
}


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_config(cls, pretrained_model_name_or_path, **kwargs):
        token = kwargs.get("hf_token")
        config_kwargs = {"trust_remote_code": True}
        if token is not None:
            config_kwargs["token"] = token
        return AutoConfig.from_pretrained(pretrained_model_name_or_path, **config_kwargs)

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        config = cls.get_config(pretrained_model_name_or_path, **kwargs)
        architectures = getattr(config, "architectures", None) or []
        arch = architectures[0] if architectures else ""

        cls_name = ARCH_OVERRIDES.get(arch)
        if cls_name is None:
            print(f"using generic AirLLM streaming model for architecture: {arch or 'unknown'}")
            cls_name = "AirLLMBaseModel"
        return "airllm", cls_name

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if is_on_mac_os:
            config = cls.get_config(pretrained_model_name_or_path, **kwargs)
            architectures = getattr(config, "architectures", None) or []
            arch = architectures[0] if architectures else ""

            if arch in MAC_QWEN35_ARCHITECTURES or getattr(config, "model_type", "") == "qwen3_5":
                print(f"using AirLLM Qwen3.5-family MLX streaming backend for: {arch or config.model_type}")
                return AirLLMQwen35Mlx(pretrained_model_name_or_path, *inputs, **kwargs)

            # Preserve AirLLM's historical Mac behavior for all other model families for now.
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        module, class_name = cls.get_module_class(pretrained_model_name_or_path, *inputs, **kwargs)
        module = importlib.import_module(module)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)
