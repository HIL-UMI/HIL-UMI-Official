# Copyright 2025 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
# Modified for HIL-UMI.

import asyncio
import io
from typing import Optional, Union, cast

import aiohttp
from PIL import Image


def _decode_image_from_bytes(img_bytes: bytes) -> Image.Image:
    with Image.open(io.BytesIO(img_bytes)) as image:
        return image.convert("RGB")


def _decode_image_from_path(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


async def _decode_bytes(idx: int, img_bytes: bytes) -> tuple[int, Image.Image]:
    image = await asyncio.to_thread(_decode_image_from_bytes, img_bytes)
    return idx, image


async def _decode_path(idx: int, path: str) -> tuple[int, Image.Image]:
    image = await asyncio.to_thread(_decode_image_from_path, path)
    return idx, image


async def _fetch_and_decode(
    session: aiohttp.ClientSession, idx: int, url: str
) -> tuple[int, Image.Image]:
    async with session.get(url) as response:
        response.raise_for_status()
        img_bytes = await response.read()
    return await _decode_bytes(idx, img_bytes)


async def process_image_data(
    image_data: list[Union[bytes, str]],
) -> list[Image.Image]:
    if not isinstance(image_data, list):
        raise ValueError("image_data must be a list of bytes or file paths.")

    tasks: list[asyncio.Task[tuple[int, Image.Image]]] = []
    remote_targets: list[tuple[int, str]] = []

    for idx, item in enumerate(image_data):
        if isinstance(item, bytes):
            tasks.append(asyncio.create_task(_decode_bytes(idx, item)))
        elif isinstance(item, str):
            if item.startswith("http://") or item.startswith("https://"):
                remote_targets.append((idx, item))
            else:
                tasks.append(asyncio.create_task(_decode_path(idx, item)))
        else:
            raise ValueError("Input must be bytes or file path string.")

    if remote_targets:
        async with aiohttp.ClientSession() as session:
            tasks.extend(
                asyncio.create_task(_fetch_and_decode(session, idx, url))
                for idx, url in remote_targets
            )

    if not tasks:
        return []

    results = await asyncio.gather(*tasks)
    ordered_images: list[Optional[Image.Image]] = [None] * len(image_data)
    for idx, image in results:
        ordered_images[idx] = image

    if any(img is None for img in ordered_images):
        raise RuntimeError("Failed to decode one or more images.")

    return cast(list[Image.Image], ordered_images)
