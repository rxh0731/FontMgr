"""深度学习去背景模型的注册、结果校验与缓存键基础设施。"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np


NO_MODEL_ENGINE_ID = "none"
MODEL_OUTPUT_NONE = "none"
MODEL_OUTPUT_PROBABILITY_MASK = "probability_mask"
MODEL_OUTPUT_BINARY_MASK = "binary_mask"
MODEL_OUTPUT_CLEAN_GRAY = "clean_gray"
SUPPORTED_MODEL_OUTPUT_TYPES = frozenset({
    MODEL_OUTPUT_NONE,
    MODEL_OUTPUT_PROBABILITY_MASK,
    MODEL_OUTPUT_BINARY_MASK,
    MODEL_OUTPUT_CLEAN_GRAY,
})


def _require_text(value: object, field_name: str) -> str:
    text = str(value).strip()
    if not text:
        raise ValueError(f"{field_name}不能为空。")
    return text


def _normalize_configuration_value(value: Any, path: str) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"模型配置 {path} 不能包含无穷大或非数字。")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"模型配置 {path} 的键必须是字符串。")
            normalized[key] = _normalize_configuration_value(child, f"{path}.{key}")
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [
            _normalize_configuration_value(child, f"{path}[{index}]")
            for index, child in enumerate(value)
        ]
    raise TypeError(f"模型配置 {path} 包含不支持的类型：{type(value).__name__}。")


def normalize_model_configuration(
    configuration: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """将模型配置复制为键顺序稳定的 JSON 兼容字典。"""
    if configuration is None:
        return {}
    if not isinstance(configuration, Mapping):
        raise TypeError("模型配置必须是字典。")
    normalized = _normalize_configuration_value(configuration, "根")
    if not isinstance(normalized, dict):
        raise TypeError("模型配置必须是字典。")
    return normalized


def model_configuration_hash(configuration: Mapping[str, Any] | None) -> str:
    """计算与字典插入顺序无关的模型配置 SHA-256。"""
    normalized = normalize_model_configuration(configuration)
    payload = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BackgroundModelDescriptor:
    """模型注册表和界面共同使用的稳定描述。"""

    engine_id: str
    display_name: str
    version: str
    output_type: str
    installed: bool
    model_fingerprint: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "engine_id", _require_text(self.engine_id, "处理引擎标识"))
        object.__setattr__(self, "display_name", _require_text(self.display_name, "处理引擎名称"))
        object.__setattr__(self, "version", _require_text(self.version, "处理引擎版本"))
        output_type = _require_text(self.output_type, "模型输出类型")
        if output_type not in SUPPORTED_MODEL_OUTPUT_TYPES:
            raise ValueError(f"不支持的模型输出类型：{output_type}。")
        if self.engine_id == NO_MODEL_ENGINE_ID and output_type != MODEL_OUTPUT_NONE:
            raise ValueError("“无学习模型”的输出类型必须是 none。")
        if self.engine_id != NO_MODEL_ENGINE_ID and output_type == MODEL_OUTPUT_NONE:
            raise ValueError("学习模型必须声明 probability_mask、binary_mask 或 clean_gray 输出。")
        if not isinstance(self.installed, bool):
            raise TypeError("模型安装状态必须是布尔值。")
        object.__setattr__(self, "output_type", output_type)
        object.__setattr__(self, "model_fingerprint", str(self.model_fingerprint).strip())


@dataclass(frozen=True)
class BackgroundModelContext:
    """一次候选分支固定使用的模型及推理配置。"""

    descriptor: BackgroundModelDescriptor
    configuration: Mapping[str, Any] = field(default_factory=dict)
    configuration_hash: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.descriptor, BackgroundModelDescriptor):
            raise TypeError("模型上下文必须包含有效的模型描述。")
        normalized = normalize_model_configuration(self.configuration)
        if self.descriptor.engine_id == NO_MODEL_ENGINE_ID and normalized:
            raise ValueError("“无学习模型”不接受推理配置。")
        object.__setattr__(self, "configuration", MappingProxyType(normalized))
        object.__setattr__(self, "configuration_hash", model_configuration_hash(normalized))

    @property
    def engine_id(self) -> str:
        return self.descriptor.engine_id

    def to_metadata(self) -> dict[str, Any]:
        """生成可直接写入自动优化方案的可追溯信息。"""
        return {
            "标识": self.descriptor.engine_id,
            "名称": self.descriptor.display_name,
            "版本": self.descriptor.version,
            "已安装": self.descriptor.installed,
            "输出类型": self.descriptor.output_type,
            "模型指纹": self.descriptor.model_fingerprint,
            "推理参数": normalize_model_configuration(self.configuration),
            "配置指纹": self.configuration_hash,
        }


@dataclass(frozen=True)
class BackgroundModelInferenceResult:
    """模型输出；数组在创建时复制并设为只读，便于安全复用缓存。"""

    output_type: str
    data: np.ndarray | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = field(init=False)

    def __post_init__(self) -> None:
        output_type = _require_text(self.output_type, "模型输出类型")
        if output_type not in SUPPORTED_MODEL_OUTPUT_TYPES:
            raise ValueError(f"不支持的模型输出类型：{output_type}。")
        object.__setattr__(self, "output_type", output_type)

        if output_type == MODEL_OUTPUT_NONE:
            if self.data is not None:
                raise ValueError("none 类型的推理结果不能包含图像数据。")
            stored_data = None
        else:
            if not isinstance(self.data, np.ndarray):
                raise TypeError("模型推理结果必须是 NumPy 数组。")
            stored_data = np.array(self.data, copy=True)
            object.__setattr__(self, "data", stored_data)

        normalized_metadata = normalize_model_configuration(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(normalized_metadata))
        self.validate()
        if stored_data is not None:
            stored_data.setflags(write=False)
        object.__setattr__(self, "fingerprint", self._build_fingerprint())

    @classmethod
    def no_model(cls) -> "BackgroundModelInferenceResult":
        return cls(MODEL_OUTPUT_NONE)

    def validate(self, expected_shape: tuple[int, int] | None = None) -> None:
        """校验输出的维度、取值范围和可选的原图尺寸。"""
        if self.output_type == MODEL_OUTPUT_NONE:
            if self.data is not None:
                raise ValueError("none 类型的推理结果不能包含图像数据。")
            return
        if not isinstance(self.data, np.ndarray):
            raise TypeError("模型推理结果必须是 NumPy 数组。")
        if self.data.ndim != 2:
            raise ValueError("模型推理结果必须是二维图像。")
        if self.data.size == 0:
            raise ValueError("模型推理结果不能为空。")
        if expected_shape is not None and tuple(self.data.shape) != tuple(expected_shape):
            raise ValueError(
                f"模型输出尺寸 {tuple(self.data.shape)} 与原图尺寸 {tuple(expected_shape)} 不一致。"
            )
        if not (np.issubdtype(self.data.dtype, np.number) or self.data.dtype == np.bool_):
            raise TypeError("模型推理结果必须使用数值或布尔数据类型。")
        if not np.isfinite(self.data).all():
            raise ValueError("模型推理结果不能包含无穷大或非数字。")

        if self.output_type == MODEL_OUTPUT_PROBABILITY_MASK:
            if self.data.dtype == np.bool_:
                raise TypeError("probability_mask 必须使用浮点或数值概率。")
            if float(np.min(self.data)) < 0.0 or float(np.max(self.data)) > 1.0:
                raise ValueError("probability_mask 的取值必须在 0 到 1 之间。")
        elif self.output_type == MODEL_OUTPUT_BINARY_MASK:
            if not np.logical_or(self.data == 0, self.data == 1).all():
                raise ValueError("binary_mask 只能包含 0 和 1。")
        elif self.output_type == MODEL_OUTPUT_CLEAN_GRAY:
            if self.data.dtype == np.bool_:
                raise TypeError("clean_gray 不能使用布尔数据类型。")
            if float(np.min(self.data)) < 0.0 or float(np.max(self.data)) > 255.0:
                raise ValueError("clean_gray 的取值必须在 0 到 255 之间。")

    def _build_fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.output_type.encode("ascii"))
        if self.data is not None:
            contiguous = np.ascontiguousarray(self.data)
            digest.update(contiguous.dtype.str.encode("ascii"))
            digest.update(json.dumps(contiguous.shape, separators=(",", ":")).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()


@runtime_checkable
class BackgroundModelAdapter(Protocol):
    """学习模型适配协议；实现不得修改传入的原图数组。"""

    @property
    def descriptor(self) -> BackgroundModelDescriptor:
        ...

    def infer(
        self,
        source: np.ndarray,
        context: BackgroundModelContext,
    ) -> BackgroundModelInferenceResult:
        ...


class NoLearningModelAdapter:
    """默认处理引擎，不执行模型推理。"""

    descriptor = BackgroundModelDescriptor(
        engine_id=NO_MODEL_ENGINE_ID,
        display_name="无学习模型",
        version="1.0",
        output_type=MODEL_OUTPUT_NONE,
        installed=True,
    )

    def infer(
        self,
        source: np.ndarray,
        context: BackgroundModelContext,
    ) -> BackgroundModelInferenceResult:
        if context.engine_id != NO_MODEL_ENGINE_ID:
            raise ValueError("无学习模型适配器只能处理 none 上下文。")
        return BackgroundModelInferenceResult.no_model()


class BackgroundModelRegistry:
    """线程安全的模型适配器注册表。"""

    def __init__(self, register_default: bool = True) -> None:
        self._adapters: dict[str, BackgroundModelAdapter] = {}
        self._lock = threading.RLock()
        if register_default:
            self.register(NoLearningModelAdapter())

    def register(self, adapter: BackgroundModelAdapter, replace: bool = False) -> None:
        if not isinstance(adapter, BackgroundModelAdapter):
            raise TypeError("模型适配器必须实现 descriptor 和 infer。")
        descriptor = adapter.descriptor
        if not isinstance(descriptor, BackgroundModelDescriptor):
            raise TypeError("模型适配器必须提供有效的模型描述。")
        with self._lock:
            if descriptor.engine_id in self._adapters and not replace:
                raise ValueError(f"处理引擎“{descriptor.engine_id}”已经注册。")
            self._adapters[descriptor.engine_id] = adapter

    def list_descriptors(self, installed_only: bool = False) -> tuple[BackgroundModelDescriptor, ...]:
        with self._lock:
            descriptors = [adapter.descriptor for adapter in self._adapters.values()]
        if installed_only:
            descriptors = [descriptor for descriptor in descriptors if descriptor.installed]
        descriptors.sort(
            key=lambda descriptor: (
                descriptor.engine_id != NO_MODEL_ENGINE_ID,
                descriptor.display_name,
                descriptor.engine_id,
            )
        )
        return tuple(descriptors)

    def get_adapter(self, engine_id: str) -> BackgroundModelAdapter:
        normalized_id = _require_text(engine_id, "处理引擎标识")
        with self._lock:
            adapter = self._adapters.get(normalized_id)
        if adapter is None:
            raise KeyError(f"未注册处理引擎：{normalized_id}。")
        return adapter

    def get_descriptor(self, engine_id: str) -> BackgroundModelDescriptor:
        return self.get_adapter(engine_id).descriptor

    def create_context(
        self,
        engine_id: str = NO_MODEL_ENGINE_ID,
        configuration: Mapping[str, Any] | None = None,
    ) -> BackgroundModelContext:
        return BackgroundModelContext(
            descriptor=self.get_descriptor(engine_id),
            configuration=configuration or {},
        )

    def infer(
        self,
        source: np.ndarray,
        context: BackgroundModelContext,
    ) -> BackgroundModelInferenceResult:
        if not isinstance(source, np.ndarray) or source.ndim not in (2, 3) or source.size == 0:
            raise ValueError("模型输入必须是非空的二维灰度图或三维图像数组。")
        adapter = self.get_adapter(context.engine_id)
        descriptor = adapter.descriptor
        if descriptor != context.descriptor:
            raise ValueError("模型上下文与当前注册的模型版本不一致，请重新选择模型。")
        if not descriptor.installed:
            raise RuntimeError(f"处理引擎“{descriptor.display_name}”尚未安装。")

        result = adapter.infer(source, context)
        if not isinstance(result, BackgroundModelInferenceResult):
            raise TypeError("模型适配器必须返回 BackgroundModelInferenceResult。")
        if result.output_type != descriptor.output_type:
            raise ValueError(
                f"模型声明输出 {descriptor.output_type}，实际返回 {result.output_type}。"
            )
        result.validate(expected_shape=tuple(source.shape[:2]))
        return result


@dataclass(frozen=True)
class InferenceCacheKey:
    source_fingerprint: str
    polarity_version: str
    engine_id: str
    engine_version: str
    output_type: str
    model_fingerprint: str
    configuration_hash: str


def build_inference_cache_key(
    source_fingerprint: str,
    context: BackgroundModelContext,
    polarity_version: str = "1",
) -> InferenceCacheKey:
    """构造模型推理缓存键；模型或输入规范变化时自动隔离。"""
    descriptor = context.descriptor
    return InferenceCacheKey(
        source_fingerprint=_require_text(source_fingerprint, "原图指纹"),
        polarity_version=_require_text(polarity_version, "极性校正规则版本"),
        engine_id=descriptor.engine_id,
        engine_version=descriptor.version,
        output_type=descriptor.output_type,
        model_fingerprint=descriptor.model_fingerprint,
        configuration_hash=context.configuration_hash,
    )


@dataclass(frozen=True)
class CandidateCacheKey:
    library_id: str
    variant_id: str
    source_fingerprint: str
    engine_id: str
    engine_version: str
    output_type: str
    model_fingerprint: str
    configuration_hash: str
    inference_fingerprint: str
    pipeline_version: str


def build_candidate_cache_key(
    library_id: str,
    variant_id: str,
    source_fingerprint: str,
    context: BackgroundModelContext,
    inference_fingerprint: str,
    pipeline_version: str = "1",
) -> CandidateCacheKey:
    """构造基础候选缓存键；探索分支不应复用此键覆盖基础候选。"""
    descriptor = context.descriptor
    return CandidateCacheKey(
        library_id=_require_text(library_id, "字库标识"),
        variant_id=_require_text(variant_id, "字形标识"),
        source_fingerprint=_require_text(source_fingerprint, "原图指纹"),
        engine_id=descriptor.engine_id,
        engine_version=descriptor.version,
        output_type=descriptor.output_type,
        model_fingerprint=descriptor.model_fingerprint,
        configuration_hash=context.configuration_hash,
        inference_fingerprint=_require_text(inference_fingerprint, "推理结果指纹"),
        pipeline_version=_require_text(pipeline_version, "寻优管线版本"),
    )


BACKGROUND_MODEL_REGISTRY = BackgroundModelRegistry()
