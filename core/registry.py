# registry.py — 算法注册表逻辑（校验、查询、说明生成）

from typing import Any, Optional

from data.registry_store import load_registry


def get_registry() -> dict[str, Any]:
    """获取当前有效的算法注册表。"""
    return load_registry()


def get_layer_groups() -> dict[str, Any]:
    """获取注册表中的分层分组结构。"""
    return get_registry().get("分组", {})


def get_layer_algos(layer_name: str) -> dict[str, Any]:
    """获取指定层所有可选算法。

    参数：
        layer_name: 如 "L1 降噪", "L3 二值化"

    返回：
        {"算法名": {"说明": "...", "参数": {...}}, ...}
    """
    groups = get_layer_groups()
    return groups.get(layer_name, {}).get("算法", {})


def validate_scheme(scheme: dict[str, Any]) -> list[str]:
    """校验一个 V3 方案是否合法，返回错误消息列表。

    参数：
        scheme: {'预处理': {...}, 'L1': {...}, ...}

    返回：
        错误消息列表，空列表表示合法
    """
    errors: list[str] = []
    if not isinstance(scheme, dict):
        errors.append("方案不是字典格式")
        return errors

    groups = get_layer_groups()

    # 校验预处理
    pre = scheme.get("预处理", {})
    if isinstance(pre, dict):
        for key in ("转灰度", "反相", "墨色归一"):
            if key in pre and not isinstance(pre[key], bool):
                errors.append(f"预处理.{key} 应为布尔值")
        if "墨色基准" in pre:
            v = pre["墨色基准"]
            if not isinstance(v, (int, float)) or v < 5 or v > 250:
                errors.append("预处理.墨色基准 超出范围 (5~250)")

    # 校验每一层
    layers = ("L1", "L2", "L3", "L4", "L5")
    group_names = ("L1 降噪", "L2 背景分离", "L3 二值化", "L4 形态清理", "L5 连通域过滤")
    has_algo = False

    for li, layer_key in enumerate(layers):
        cfg = scheme.get(layer_key)
        if cfg is None:
            continue
        if not isinstance(cfg, dict):
            errors.append(f"{layer_key} 配置应为字典")
            continue
        algo_name = cfg.get("算法", "")
        if not algo_name:
            continue
        has_algo = True
        group = groups.get(group_names[li], {})
        available = group.get("算法", {})
        if algo_name not in available:
            errors.append(f"{layer_key} 算法「{algo_name}」不在注册表的 {group_names[li]} 中")
        # 参数校验
        param_defs = available.get(algo_name, {}).get("参数", {})
        user_params = cfg.get("参数", {})
        for pk, pdef in param_defs.items():
            if pk not in user_params:
                continue
            ptype = pdef.get("类型", "")
            val = user_params[pk]
            if ptype == "int" and not isinstance(val, int):
                errors.append(f"{layer_key}.参数.{pk} 应为整数")
            elif ptype == "float" and not isinstance(val, (int, float)):
                errors.append(f"{layer_key}.参数.{pk} 应为数值")
            elif ptype == "bool" and not isinstance(val, bool):
                errors.append(f"{layer_key}.参数.{pk} 应为布尔值")

    if not has_algo:
        errors.append("方案至少需要一层选择算法")

    return errors
