"""深度学习去背景模型基础设施回归测试。"""

from __future__ import annotations

import unittest

import numpy as np

from services.background_model_service import (
    BACKGROUND_MODEL_REGISTRY,
    MODEL_OUTPUT_BINARY_MASK,
    MODEL_OUTPUT_CLEAN_GRAY,
    MODEL_OUTPUT_NONE,
    MODEL_OUTPUT_PROBABILITY_MASK,
    NO_MODEL_ENGINE_ID,
    BackgroundModelContext,
    BackgroundModelDescriptor,
    BackgroundModelInferenceResult,
    BackgroundModelRegistry,
    build_candidate_cache_key,
    build_inference_cache_key,
    model_configuration_hash,
    normalize_model_configuration,
)


class BackgroundModelServiceTests(unittest.TestCase):
    """验证默认引擎、模型输出约束及缓存隔离语义。"""

    def test_default_registry_contains_stable_no_model_engine(self) -> None:
        descriptors = BACKGROUND_MODEL_REGISTRY.list_descriptors()

        self.assertGreaterEqual(len(descriptors), 1)
        descriptor = descriptors[0]
        self.assertEqual(descriptor.engine_id, NO_MODEL_ENGINE_ID)
        self.assertEqual(descriptor.display_name, "无学习模型")
        self.assertEqual(descriptor.version, "1.0")
        self.assertEqual(descriptor.output_type, MODEL_OUTPUT_NONE)
        self.assertTrue(descriptor.installed)

    def test_no_model_inference_does_not_change_source(self) -> None:
        registry = BackgroundModelRegistry()
        context = registry.create_context()
        source = np.arange(16, dtype=np.uint8).reshape(4, 4)
        before = source.copy()

        result = registry.infer(source, context)

        np.testing.assert_array_equal(source, before)
        self.assertEqual(result.output_type, MODEL_OUTPUT_NONE)
        self.assertIsNone(result.data)
        self.assertEqual(result.fingerprint, BackgroundModelInferenceResult.no_model().fingerprint)

    def test_context_metadata_is_traceable_and_no_model_rejects_configuration(self) -> None:
        context = BACKGROUND_MODEL_REGISTRY.create_context()

        metadata = context.to_metadata()

        self.assertEqual(metadata["标识"], NO_MODEL_ENGINE_ID)
        self.assertEqual(metadata["名称"], "无学习模型")
        self.assertEqual(metadata["推理参数"], {})
        self.assertEqual(metadata["配置指纹"], model_configuration_hash({}))
        with self.assertRaisesRegex(ValueError, "不接受推理配置"):
            BACKGROUND_MODEL_REGISTRY.create_context(configuration={"阈值": 0.5})

    def test_configuration_hash_is_normalized_and_independent_from_input(self) -> None:
        first = {"设备": "CPU", "参数": {"阈值": np.float32(0.5), "尺寸": (512, 512)}}
        second = {"参数": {"尺寸": [512, 512], "阈值": 0.5}, "设备": "CPU"}

        normalized = normalize_model_configuration(first)

        self.assertEqual(normalized, second)
        self.assertEqual(model_configuration_hash(first), model_configuration_hash(second))
        normalized["参数"]["阈值"] = 0.8
        self.assertEqual(float(first["参数"]["阈值"]), 0.5)

    def test_configuration_rejects_unstable_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "无穷大或非数字"):
            model_configuration_hash({"阈值": float("nan")})
        with self.assertRaisesRegex(TypeError, "键必须是字符串"):
            model_configuration_hash({1: "错误"})  # type: ignore[dict-item]
        with self.assertRaisesRegex(TypeError, "不支持的类型"):
            model_configuration_hash({"集合": {1, 2}})

    def test_all_future_output_types_accept_valid_data_and_copy_input(self) -> None:
        cases = (
            (MODEL_OUTPUT_PROBABILITY_MASK, np.array([[0.0, 0.5], [1.0, 0.2]], dtype=np.float32)),
            (MODEL_OUTPUT_BINARY_MASK, np.array([[0, 1], [1, 0]], dtype=np.uint8)),
            (MODEL_OUTPUT_CLEAN_GRAY, np.array([[0, 128], [255, 30]], dtype=np.uint8)),
        )
        for output_type, source in cases:
            with self.subTest(output_type=output_type):
                result = BackgroundModelInferenceResult(output_type, source)
                source.fill(0)

                self.assertFalse(result.data.flags.writeable)  # type: ignore[union-attr]
                result.validate(expected_shape=(2, 2))
                self.assertNotEqual(result.fingerprint, "")

    def test_model_output_validation_rejects_invalid_type_shape_and_range(self) -> None:
        with self.assertRaisesRegex(ValueError, "不支持的模型输出类型"):
            BackgroundModelInferenceResult("unknown", np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "二维图像"):
            BackgroundModelInferenceResult(MODEL_OUTPUT_BINARY_MASK, np.zeros((2, 2, 1)))
        with self.assertRaisesRegex(ValueError, "0 到 1"):
            BackgroundModelInferenceResult(
                MODEL_OUTPUT_PROBABILITY_MASK,
                np.array([[0.0, 1.2]], dtype=np.float32),
            )
        with self.assertRaisesRegex(ValueError, "只能包含 0 和 1"):
            BackgroundModelInferenceResult(
                MODEL_OUTPUT_BINARY_MASK,
                np.array([[0, 255]], dtype=np.uint8),
            )
        with self.assertRaisesRegex(ValueError, "0 到 255"):
            BackgroundModelInferenceResult(
                MODEL_OUTPUT_CLEAN_GRAY,
                np.array([[-1.0, 255.0]], dtype=np.float32),
            )

    def test_descriptor_rejects_invalid_output_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "输出类型必须是 none"):
            BackgroundModelDescriptor(
                NO_MODEL_ENGINE_ID,
                "错误默认引擎",
                "1.0",
                MODEL_OUTPUT_BINARY_MASK,
                True,
            )
        with self.assertRaisesRegex(ValueError, "学习模型必须声明"):
            BackgroundModelDescriptor("demo", "演示模型", "1.0", MODEL_OUTPUT_NONE, True)

    def test_registry_checks_declared_output_and_original_size(self) -> None:
        registry = BackgroundModelRegistry(register_default=False)
        adapter = _StaticAdapter(
            BackgroundModelDescriptor(
                "probability-demo",
                "概率演示模型",
                "2.0",
                MODEL_OUTPUT_PROBABILITY_MASK,
                True,
                "model-sha256",
            ),
            BackgroundModelInferenceResult(
                MODEL_OUTPUT_PROBABILITY_MASK,
                np.full((3, 4), 0.5, dtype=np.float32),
            ),
        )
        registry.register(adapter)
        context = registry.create_context("probability-demo", {"阈值": 0.5})

        result = registry.infer(np.zeros((3, 4), dtype=np.uint8), context)

        self.assertEqual(result.output_type, MODEL_OUTPUT_PROBABILITY_MASK)
        with self.assertRaisesRegex(ValueError, "与原图尺寸"):
            registry.infer(np.zeros((4, 4), dtype=np.uint8), context)

    def test_cache_keys_are_stable_and_isolate_configuration_and_pipeline(self) -> None:
        descriptor = BackgroundModelDescriptor(
            "probability-demo",
            "概率演示模型",
            "2.0",
            MODEL_OUTPUT_PROBABILITY_MASK,
            True,
            "model-sha256",
        )
        first_context = BackgroundModelContext(descriptor, {"阈值": 0.5, "设备": "CPU"})
        reordered_context = BackgroundModelContext(descriptor, {"设备": "CPU", "阈值": 0.5})
        changed_context = BackgroundModelContext(descriptor, {"阈值": 0.6, "设备": "CPU"})

        first_inference = build_inference_cache_key("source-md5", first_context)
        self.assertEqual(first_inference, build_inference_cache_key("source-md5", reordered_context))
        self.assertNotEqual(first_inference, build_inference_cache_key("source-md5", changed_context))

        first_candidate = build_candidate_cache_key(
            "library-a",
            "variant-a",
            "source-md5",
            first_context,
            "inference-sha256",
        )
        self.assertEqual(
            first_candidate,
            build_candidate_cache_key(
                "library-a",
                "variant-a",
                "source-md5",
                reordered_context,
                "inference-sha256",
            ),
        )
        self.assertNotEqual(
            first_candidate,
            build_candidate_cache_key(
                "library-a",
                "variant-a",
                "source-md5",
                first_context,
                "inference-sha256",
                pipeline_version="2",
            ),
        )


class _StaticAdapter:
    def __init__(
        self,
        descriptor: BackgroundModelDescriptor,
        result: BackgroundModelInferenceResult,
    ) -> None:
        self._descriptor = descriptor
        self._result = result

    @property
    def descriptor(self) -> BackgroundModelDescriptor:
        return self._descriptor

    def infer(
        self,
        _source: np.ndarray,
        _context: BackgroundModelContext,
    ) -> BackgroundModelInferenceResult:
        return self._result


if __name__ == "__main__":
    unittest.main()
