"""保守解码 Photoshop TIFF 中可验证的单图层透明像素。"""

from __future__ import annotations

import struct
from contextlib import ExitStack
from dataclasses import dataclass

from PIL import Image, ImageChops


_IMAGE_SOURCE_DATA_TAG = 37724
_PHOTOSHOP_HEADER = b"Adobe Photoshop Document Data Block\x00"
_REQUIRED_CHANNEL_IDS = frozenset((-1, 0, 1, 2))
_MAX_CANVAS_PIXELS = 16 * 1024 * 1024
_MAX_CANVAS_SIDE = 32_768
_MAX_LAYER_PIXELS = 16 * 1024 * 1024
_MAX_LAYER_SIDE = 32_768
_MAX_CHANNEL_BYTES = 24 * 1024 * 1024
_MAX_LAYER_DATA_BYTES = 4 * _MAX_CHANNEL_BYTES + 64 * 1024


class _UnsupportedLayerData(ValueError):
    """表示图层数据不在当前保守解码范围内。"""


@dataclass(frozen=True)
class _ChannelRecord:
    channel_id: int
    length: int


@dataclass(frozen=True)
class _LayerRecord:
    top: int
    left: int
    bottom: int
    right: int
    channels: tuple[_ChannelRecord, ...]

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def decode_single_layer_rgba(image: Image.Image) -> Image.Image | None:
    """解码严格受限的 Photoshop 单图层 RGBA，不支持时返回 ``None``。

    当前只接受 8 位 RGB TIFF、单个可见普通图层、完整的 Alpha/R/G/B
    四通道，以及未压缩或 PackBits 压缩。解码结果还必须重新合成出与
    TIFF 主图一致的白底图，避免把不完整或含复杂图层效果的数据误用。
    """
    try:
        return _decode_single_layer_rgba(image)
    except (OSError, OverflowError, TypeError, ValueError, struct.error):
        return None


def _decode_single_layer_rgba(image: Image.Image) -> Image.Image:
    tags = getattr(image, "tag_v2", None)
    if tags is None:
        raise _UnsupportedLayerData("图片没有 TIFF 标签。")
    payload_value = tags.get(_IMAGE_SOURCE_DATA_TAG)
    if not isinstance(payload_value, (bytes, bytearray)):
        raise _UnsupportedLayerData("图片没有 Photoshop 图层数据。")
    if len(payload_value) > _MAX_LAYER_DATA_BYTES:
        raise _UnsupportedLayerData("Photoshop 图层数据超过安全上限。")
    if image.mode != "RGB" or int(tags.get(277, 0) or 0) != 3:
        raise _UnsupportedLayerData("只支持 8 位 RGB TIFF 主图。")
    bits_value = tags.get(258, ())
    bits = (
        tuple(int(value) for value in bits_value)
        if isinstance(bits_value, (tuple, list))
        else (int(bits_value),)
    )
    if bits != (8, 8, 8):
        raise _UnsupportedLayerData("只支持每通道 8 位的 TIFF。")

    canvas_width, canvas_height = (int(value) for value in image.size)
    if canvas_width <= 0 or canvas_height <= 0:
        raise _UnsupportedLayerData("TIFF 画布尺寸无效。")
    if max(canvas_width, canvas_height) > _MAX_CANVAS_SIDE:
        raise _UnsupportedLayerData("TIFF 画布边长超过安全上限。")
    if canvas_width * canvas_height > _MAX_CANVAS_PIXELS:
        raise _UnsupportedLayerData("TIFF 画布像素数量超过安全上限。")

    block, byte_order, signature = _find_layer_block(bytes(payload_value))
    layer, data_offset = _parse_single_layer(
        block,
        byte_order,
        signature,
        (canvas_width, canvas_height),
    )
    channel_images: dict[int, Image.Image] = {}
    position = data_offset
    try:
        for channel in layer.channels:
            channel_end = position + channel.length
            if channel_end > len(block):
                raise _UnsupportedLayerData("图层通道长度超出数据块。")
            channel_images[channel.channel_id] = _decode_channel(
                block[position:channel_end],
                layer.width,
                layer.height,
                byte_order,
            )
            position = channel_end
        if position != len(block):
            raise _UnsupportedLayerData("图层通道后存在未识别数据。")

        layer_rgba = Image.merge(
            "RGBA",
            (
                channel_images[0],
                channel_images[1],
                channel_images[2],
                channel_images[-1],
            ),
        )
    finally:
        for channel_image in channel_images.values():
            channel_image.close()

    try:
        canvas = Image.new("RGBA", image.size, (0, 0, 0, 0))
        try:
            canvas.paste(layer_rgba, (layer.left, layer.top))
            alpha = canvas.getchannel("A")
            try:
                alpha_minimum, alpha_maximum = alpha.getextrema()
            finally:
                alpha.close()
            if alpha_minimum >= 255 or alpha_maximum <= 0:
                raise _UnsupportedLayerData("图层没有有效透明背景或可见内容。")
            if not _matches_flattened_white_composite(canvas, image):
                raise _UnsupportedLayerData("图层合成结果与 TIFF 主图不一致。")
        except Exception:
            canvas.close()
            raise
    finally:
        layer_rgba.close()
    return canvas


def _find_layer_block(payload: bytes) -> tuple[bytes, str, bytes]:
    if not payload.startswith(_PHOTOSHOP_HEADER):
        raise _UnsupportedLayerData("Photoshop 图层数据头无效。")
    position = len(_PHOTOSHOP_HEADER)
    while position + 12 <= len(payload):
        signature = payload[position:position + 4]
        key = payload[position + 4:position + 8]
        if signature == b"MIB8":
            byte_order = "<"
            layer_key = b"ryaL"
        elif signature == b"8BIM":
            byte_order = ">"
            layer_key = b"Layr"
        else:
            raise _UnsupportedLayerData("Photoshop 数据块签名不受支持。")
        block_length = struct.unpack_from(
            f"{byte_order}I",
            payload,
            position + 8,
        )[0]
        block_start = position + 12
        block_end = block_start + block_length
        if block_end > len(payload):
            raise _UnsupportedLayerData("Photoshop 数据块长度无效。")
        if key == layer_key:
            return payload[block_start:block_end], byte_order, signature
        position = block_end + (block_length & 1)
    raise _UnsupportedLayerData("未找到 8 位 Photoshop 图层信息。")


def _parse_single_layer(
    block: bytes,
    byte_order: str,
    signature: bytes,
    canvas_size: tuple[int, int],
) -> tuple[_LayerRecord, int]:
    if len(block) < 2:
        raise _UnsupportedLayerData("图层信息为空。")
    layer_count = abs(struct.unpack_from(f"{byte_order}h", block, 0)[0])
    if layer_count != 1:
        raise _UnsupportedLayerData("只支持单图层 Photoshop TIFF。")

    position = 2
    if position + 18 > len(block):
        raise _UnsupportedLayerData("图层记录不完整。")
    top, left, bottom, right = struct.unpack_from(
        f"{byte_order}4i",
        block,
        position,
    )
    position += 16
    canvas_width, canvas_height = canvas_size
    if not (
        0 <= left < right <= canvas_width
        and 0 <= top < bottom <= canvas_height
    ):
        raise _UnsupportedLayerData("图层边界超出 TIFF 画布。")
    layer_width = right - left
    layer_height = bottom - top
    if max(layer_width, layer_height) > _MAX_LAYER_SIDE:
        raise _UnsupportedLayerData("图层边长超过安全上限。")
    if layer_width * layer_height > _MAX_LAYER_PIXELS:
        raise _UnsupportedLayerData("图层像素数量超过安全上限。")

    channel_count = struct.unpack_from(f"{byte_order}H", block, position)[0]
    position += 2
    if channel_count != len(_REQUIRED_CHANNEL_IDS):
        raise _UnsupportedLayerData("图层通道数量不受支持。")
    if position + channel_count * 6 + 16 > len(block):
        raise _UnsupportedLayerData("图层通道记录不完整。")
    channels: list[_ChannelRecord] = []
    for _ in range(channel_count):
        channel_id, channel_length = struct.unpack_from(
            f"{byte_order}hI",
            block,
            position,
        )
        position += 6
        if not 2 <= channel_length <= _MAX_CHANNEL_BYTES:
            raise _UnsupportedLayerData("图层通道长度无效。")
        channels.append(_ChannelRecord(channel_id, channel_length))
    if frozenset(channel.channel_id for channel in channels) != _REQUIRED_CHANNEL_IDS:
        raise _UnsupportedLayerData("图层必须且只能包含 Alpha/R/G/B 通道。")

    blend_signature = block[position:position + 4]
    blend_key = block[position + 4:position + 8]
    expected_blend_key = b"mron" if byte_order == "<" else b"norm"
    if blend_signature != signature or blend_key != expected_blend_key:
        raise _UnsupportedLayerData("只支持普通混合模式。")
    opacity = block[position + 8]
    clipping = block[position + 9]
    flags = block[position + 10]
    filler = block[position + 11]
    extra_length = struct.unpack_from(f"{byte_order}I", block, position + 12)[0]
    position += 16
    if opacity != 255 or clipping != 0 or flags & 0x02 or filler != 0:
        raise _UnsupportedLayerData("图层可见性、不透明度或剪贴设置不受支持。")
    extra_end = position + extra_length
    if extra_length < 8 or extra_end > len(block):
        raise _UnsupportedLayerData("图层附加数据长度无效。")
    mask_length = struct.unpack_from(f"{byte_order}I", block, position)[0]
    if mask_length != 0:
        raise _UnsupportedLayerData("暂不支持单独的图层蒙版。")
    position += 4
    blending_length = struct.unpack_from(f"{byte_order}I", block, position)[0]
    position += 4
    if position + blending_length > extra_end:
        raise _UnsupportedLayerData("图层混合范围长度无效。")
    position = extra_end

    channel_total = sum(channel.length for channel in channels)
    if channel_total != len(block) - position:
        raise _UnsupportedLayerData("图层通道声明长度与数据块不一致。")
    return (
        _LayerRecord(top, left, bottom, right, tuple(channels)),
        position,
    )


def _decode_channel(
    data: bytes,
    width: int,
    height: int,
    byte_order: str,
) -> Image.Image:
    if len(data) < 2:
        raise _UnsupportedLayerData("图层通道数据为空。")
    compression = struct.unpack_from(f"{byte_order}H", data, 0)[0]
    pixel_count = width * height
    if compression == 0:
        if len(data) != 2 + pixel_count:
            raise _UnsupportedLayerData("未压缩通道长度无效。")
        decoded = data[2:]
    elif compression == 1:
        table_end = 2 + height * 2
        if table_end > len(data):
            raise _UnsupportedLayerData("PackBits 行长度表不完整。")
        row_lengths = struct.unpack_from(
            f"{byte_order}{height}H",
            data,
            2,
        )
        if table_end + sum(row_lengths) != len(data):
            raise _UnsupportedLayerData("PackBits 行长度与通道长度不一致。")
        output = bytearray(pixel_count)
        source_position = table_end
        target_position = 0
        for row_length in row_lengths:
            row_end = source_position + row_length
            row = _decode_packbits_row(data[source_position:row_end], width)
            output[target_position:target_position + width] = row
            source_position = row_end
            target_position += width
        decoded = bytes(output)
    else:
        raise _UnsupportedLayerData("只支持未压缩或 PackBits 图层通道。")
    return Image.frombytes("L", (width, height), decoded)


def _decode_packbits_row(data: bytes, width: int) -> bytes:
    output = bytearray()
    position = 0
    while position < len(data):
        header = data[position]
        position += 1
        signed_header = header if header < 128 else header - 256
        if 0 <= signed_header <= 127:
            count = signed_header + 1
            if position + count > len(data) or len(output) + count > width:
                raise _UnsupportedLayerData("PackBits 原样段越界。")
            output.extend(data[position:position + count])
            position += count
        elif -127 <= signed_header <= -1:
            count = 1 - signed_header
            if position >= len(data) or len(output) + count > width:
                raise _UnsupportedLayerData("PackBits 重复段越界。")
            output.extend(bytes((data[position],)) * count)
            position += 1
        # -128 是合法的空操作，占一个控制字节。
    if len(output) != width:
        raise _UnsupportedLayerData("PackBits 解码行宽不一致。")
    return bytes(output)


def _matches_flattened_white_composite(
    decoded_rgba: Image.Image,
    flattened_image: Image.Image,
) -> bool:
    with ExitStack() as stack:
        white = Image.new("RGBA", decoded_rgba.size, "white")
        stack.callback(white.close)
        composite = Image.alpha_composite(white, decoded_rgba)
        stack.callback(composite.close)
        flattened_rgb = flattened_image.convert("RGB")
        stack.callback(flattened_rgb.close)
        composite_rgb = composite.convert("RGB")
        stack.callback(composite_rgb.close)
        difference = ImageChops.difference(composite_rgb, flattened_rgb)
        stack.callback(difference.close)
        extrema = difference.getextrema()
        return all(maximum <= 1 for _minimum, maximum in extrema)
