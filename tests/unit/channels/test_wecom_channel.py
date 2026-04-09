# -*- coding: utf-8 -*-
# pylint: disable=protected-access
"""Unit tests for WecomChannel (with inkOrCloud aibot SDK)."""
from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from aibot import MediaType
from copaw.app.channels.wecom.channel import WecomChannel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_channel(**overrides: Any) -> WecomChannel:
    """Create a WecomChannel with minimal dummy config."""

    async def _noop_process(_request):
        yield  # pragma: no cover

    defaults = {
        "process": _noop_process,
        "enabled": True,
        "bot_id": "test-bot",
        "secret": "test-secret",
        "bot_prefix": "",
        "media_dir": "",
    }
    defaults.update(overrides)
    return WecomChannel(**defaults)


def _make_content_part(type_value, **attrs):
    """Create a simple mock content part."""
    part = MagicMock()
    part.type = type_value
    for k, v in attrs.items():
        setattr(part, k, v)
    return part


def _make_frame() -> MagicMock:
    return MagicMock(name="ws_frame")


# ---------------------------------------------------------------------------
# start() — verify removed workarounds are gone
# ---------------------------------------------------------------------------


class TestStart:
    def test_no_upload_lock_attribute(self):
        """_upload_lock field must not exist after removing the workaround."""
        ch = _make_channel()
        assert not hasattr(ch, "_upload_lock")

    def test_no_upload_ack_futures_attribute(self):
        """_upload_ack_futures field must not exist after removing the workaround."""
        ch = _make_channel()
        assert not hasattr(ch, "_upload_ack_futures")

    @pytest.mark.asyncio
    async def test_start_does_not_patch_send_heartbeat(self):
        """start() must NOT replace sdk._ws_manager._send_heartbeat."""
        ch = _make_channel()
        mock_client = MagicMock()
        mock_ws_mgr = MagicMock()
        original_hb = MagicMock(name="original_heartbeat")
        mock_ws_mgr._send_heartbeat = original_hb
        mock_client._ws_manager = mock_ws_mgr
        mock_client.connect = AsyncMock()

        with patch("copaw.app.channels.wecom.channel.WSClient", return_value=mock_client), \
             patch("copaw.app.channels.wecom.channel.WSClientOptions"), \
             patch.object(ch, "_run_ws_forever"):
            await ch.start()

        # The SDK's _send_heartbeat must remain untouched
        assert mock_ws_mgr._send_heartbeat is original_hb

    @pytest.mark.asyncio
    async def test_start_does_not_intercept_on_message(self):
        """start() must NOT replace sdk._ws_manager.on_message."""
        ch = _make_channel()
        mock_client = MagicMock()
        mock_ws_mgr = MagicMock()
        original_handler = MagicMock(name="original_on_message")
        mock_ws_mgr.on_message = original_handler
        mock_client._ws_manager = mock_ws_mgr
        mock_client.connect = AsyncMock()

        with patch("copaw.app.channels.wecom.channel.WSClient", return_value=mock_client), \
             patch("copaw.app.channels.wecom.channel.WSClientOptions"), \
             patch.object(ch, "_run_ws_forever"):
            await ch.start()

        assert mock_ws_mgr.on_message is original_handler


# ---------------------------------------------------------------------------
# _send_media_part — IMAGE
# ---------------------------------------------------------------------------


class TestSendMediaPartImage:
    @pytest.mark.asyncio
    async def test_image_reply_via_frame(self, tmp_path):
        ch = _make_channel()
        img = tmp_path / "photo.png"
        img.write_bytes(b"\x89PNG\r\n" + b"0" * 100)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="media123")
        ch._client.reply_image = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.IMAGE, image_url=f"file://{img}")
        frame = _make_frame()

        with patch("copaw.app.channels.wecom.channel.compress_image_for_wecom",
                   return_value=(b"imgdata", "photo.png")):
            await ch._send_media_part("", part, frame)

        ch._client.upload_media.assert_awaited_once()
        call_args = ch._client.upload_media.call_args
        assert call_args.args[1] == "photo.png"
        assert call_args.args[2] == MediaType.Image
        ch._client.reply_image.assert_awaited_once_with(frame, "media123")

    @pytest.mark.asyncio
    async def test_image_send_message_via_chatid(self, tmp_path):
        ch = _make_channel()
        img = tmp_path / "photo.jpg"
        img.write_bytes(b"JFIF" + b"0" * 50)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="mediaxyz")
        ch._client.send_message = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.IMAGE, image_url=f"file://{img}")

        with patch("copaw.app.channels.wecom.channel.compress_image_for_wecom",
                   return_value=(b"imgdata", "photo.jpg")):
            await ch._send_media_part("user123", part, None)

        ch._client.send_message.assert_awaited_once_with(
            "user123", {"msgtype": "image", "image": {"media_id": "mediaxyz"}}
        )


# ---------------------------------------------------------------------------
# _send_media_part — AUDIO (AMR = voice, other = file)
# ---------------------------------------------------------------------------


class TestSendMediaPartAudio:
    @pytest.mark.asyncio
    async def test_amr_uses_voice_type(self, tmp_path):
        ch = _make_channel()
        amr = tmp_path / "voice.amr"
        amr.write_bytes(b"#!AMR" + b"0" * 50)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="voiceid")
        ch._client.reply_voice = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(
            ContentType.AUDIO, data=f"file://{amr}", file_url=""
        )
        await ch._send_media_part("", part, _make_frame())

        call_args = ch._client.upload_media.call_args
        assert call_args.args[2] == MediaType.Voice
        ch._client.reply_voice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_mp3_uses_file_type(self, tmp_path):
        ch = _make_channel()
        mp3 = tmp_path / "audio.mp3"
        mp3.write_bytes(b"ID3" + b"0" * 50)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="fileid")
        ch._client.reply_file = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(
            ContentType.AUDIO, data=f"file://{mp3}", file_url=""
        )
        await ch._send_media_part("", part, _make_frame())

        call_args = ch._client.upload_media.call_args
        assert call_args.args[2] == MediaType.File
        ch._client.reply_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# _send_media_part — VIDEO / FILE
# ---------------------------------------------------------------------------


class TestSendMediaPartVideoFile:
    @pytest.mark.asyncio
    async def test_video_reply(self, tmp_path):
        ch = _make_channel()
        vid = tmp_path / "clip.mp4"
        vid.write_bytes(b"\x00\x00\x00" + b"0" * 100)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="videoid")
        ch._client.reply_video = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.VIDEO, video_url=f"file://{vid}")
        await ch._send_media_part("", part, _make_frame())

        call_args = ch._client.upload_media.call_args
        assert call_args.args[2] == MediaType.Video
        ch._client.reply_video.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_file_reply(self, tmp_path):
        ch = _make_channel()
        doc = tmp_path / "report.pdf"
        doc.write_bytes(b"%PDF" + b"0" * 100)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="fileid2")
        ch._client.reply_file = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.FILE, file_url=f"file://{doc}")
        await ch._send_media_part("", part, _make_frame())

        call_args = ch._client.upload_media.call_args
        assert call_args.args[2] == MediaType.File
        ch._client.reply_file.assert_awaited_once()


# ---------------------------------------------------------------------------
# _send_media_part — Error / edge cases
# ---------------------------------------------------------------------------


class TestSendMediaPartEdgeCases:
    @pytest.mark.asyncio
    async def test_file_not_found_skips_upload(self, tmp_path):
        ch = _make_channel()
        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(
            ContentType.IMAGE,
            image_url="file:///nonexistent/path/img.png",
        )
        await ch._send_media_part("", part, _make_frame())

        ch._client.upload_media.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upload_failure_skips_reply(self, tmp_path):
        ch = _make_channel()
        img = tmp_path / "img.png"
        img.write_bytes(b"\x89PNG" + b"0" * 50)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(side_effect=RuntimeError("upload err"))
        ch._client.reply_image = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.IMAGE, image_url=f"file://{img}")

        with patch("copaw.app.channels.wecom.channel.compress_image_for_wecom",
                   return_value=(b"data", "img.png")):
            await ch._send_media_part("", part, _make_frame())

        ch._client.reply_image.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_client_returns_immediately(self):
        ch = _make_channel()
        ch._client = None

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.IMAGE, image_url="file:///some/img.png")
        # Must not raise
        await ch._send_media_part("", part, _make_frame())

    @pytest.mark.asyncio
    async def test_unknown_content_type_is_ignored(self):
        ch = _make_channel()
        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock()

        part = _make_content_part("unknown_type")
        await ch._send_media_part("chatid", part, _make_frame())

        ch._client.upload_media.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_md5_passed_to_upload(self, tmp_path):
        """upload_media must receive the md5 keyword argument."""
        import hashlib

        ch = _make_channel()
        doc = tmp_path / "file.pdf"
        raw = b"%PDF" + b"x" * 200
        doc.write_bytes(raw)

        ch._client = MagicMock()
        ch._client.upload_media = AsyncMock(return_value="mid")
        ch._client.reply_file = AsyncMock()

        from agentscope_runtime.engine.schemas.agent_schemas import ContentType
        part = _make_content_part(ContentType.FILE, file_url=f"file://{doc}")
        await ch._send_media_part("", part, _make_frame())

        expected_md5 = hashlib.md5(raw).hexdigest()
        call_kwargs = ch._client.upload_media.call_args.kwargs
        assert call_kwargs.get("md5") == expected_md5
