"""Platform profile registry: the data every channel envelope is built from.

One frozen ``PlatformProfile`` per supported platform. Three platforms --
Discord, Slack, Telegram -- carry verified message-length limits matching the
renderer ceilings in ``src/wrapper/contract.py``; every other platform
carries the conservative default pair (1600/1800). No vendor media, reply,
reaction, or action fact is verified in this repository -- there is no
transport evidence here -- so every capability value is ``None`` (unknown),
which the envelope builder resolves to ``False`` with provenance until an
adapter declares otherwise.

This module is data only. Validation and envelope construction live in
``platform_envelope.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

#: Conservative message-length limits for platforms whose real cap has not
#: been verified against the live service. Matches the generic messenger
#: ceiling in ``src/wrapper/contract.py``.
CONSERVATIVE_MAX_RECOMMENDED_CHARS = 1600
CONSERVATIVE_HARD_LIMIT_CHARS = 1800

#: Absolute ceiling on any declared hard limit for a platform without
#: verified limits. Nothing a chat platform accepts legitimately approaches
#: this.
CORE_LIMIT_CAP = 40000

#: The fixed claim boundary every envelope carries: the adapter owns the
#: transport (sending, receiving, rate limits, retries); the core owns the
#: response (what is said, in what order, within the declared limits).
CLAIM_BOUNDARY = "adapter_owns_transport_core_owns_response"

#: Render profiles the core can emit, mirroring ``RENDER_PROFILES`` in
#: ``src/wrapper/contract.py``.
RENDER_PROFILE_LIMITED_MARKDOWN = "limited_markdown"
RENDER_PROFILE_RICH_MARKDOWN = "rich_markdown"
RENDER_PROFILES = (RENDER_PROFILE_LIMITED_MARKDOWN, RENDER_PROFILE_RICH_MARKDOWN)

#: Conservative default render profile. Unknown adapter input normalizes
#: down to this.
DEFAULT_RENDER_PROFILE = RENDER_PROFILE_LIMITED_MARKDOWN

#: Capability groups every profile declares.
CAPABILITY_GROUPS = ("media", "reply", "reactions", "actions")

#: Capability names per group. Every value is ``None`` (unknown): no vendor
#: capability fact is verified in this repository.
_CAPABILITY_NAMES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "media": ("images", "files", "voice", "video", "captions"),
        "reply": ("threads", "quotes", "edit", "delete"),
        "reactions": ("native", "custom_emoji"),
        "actions": ("buttons", "forms", "typing_indicator"),
    }
)


@dataclass(frozen=True)
class PlatformLimits:
    """Message-length limits, keyed exactly as the renderer chunking dict."""

    max_recommended_chars: int
    hard_limit_chars: int


@dataclass(frozen=True)
class PlatformProfile:
    """One platform's static profile.

    ``render_profile`` is the core-side rendering dialect (limited or rich
    Markdown); ``format_family`` names the vendor text format the adapter
    ultimately emits. ``capabilities`` maps group -> capability -> verified
    bool or ``None``; today every value is ``None``.
    """

    platform_id: str
    transport_source: str
    render_profile: str
    format_family: str
    limits: PlatformLimits
    capabilities: Mapping[str, Mapping[str, bool | None]] = field(
        default_factory=dict
    )


def _unknown_capabilities() -> Mapping[str, Mapping[str, bool | None]]:
    """Every declared capability in every group, all unknown."""
    return MappingProxyType(
        {
            group: MappingProxyType({name: None for name in _CAPABILITY_NAMES[group]})
            for group in CAPABILITY_GROUPS
        }
    )


def _profile(
    platform_id: str,
    *,
    transport_source: str = "generic",
    render_profile: str = DEFAULT_RENDER_PROFILE,
    format_family: str = "plain_text",
    max_recommended_chars: int = CONSERVATIVE_MAX_RECOMMENDED_CHARS,
    hard_limit_chars: int = CONSERVATIVE_HARD_LIMIT_CHARS,
) -> PlatformProfile:
    return PlatformProfile(
        platform_id=platform_id,
        transport_source=transport_source,
        render_profile=render_profile,
        format_family=format_family,
        limits=PlatformLimits(
            max_recommended_chars=max_recommended_chars,
            hard_limit_chars=hard_limit_chars,
        ),
        capabilities=_unknown_capabilities(),
    )


#: The exact set of supported platform ids, in declaration order.
PLATFORM_IDS: tuple[str, ...] = (
    "telegram",
    "discord",
    "whatsapp",
    "slack",
    "signal",
    "mattermost",
    "matrix",
    "home_assistant",
    "email",
    "twilio_sms",
    "dingtalk",
    "wecom_websocket",
    "wecom_callback",
    "wechat_ilink",
    "feishu_lark",
    "imessage_bluebubbles",
    "qqbot",
    "tencent_yuanbao",
    "microsoft_teams",
    "line",
    "simplex",
    "api_server",
    "buzz",
)

PLATFORM_PROFILES: Mapping[str, PlatformProfile] = MappingProxyType(
    {
        profile.platform_id: profile
        for profile in (
            _profile(
                "telegram",
                transport_source="telegram",
                format_family="telegram_plain_text",
                max_recommended_chars=3700,
                hard_limit_chars=3900,
            ),
            _profile(
                "discord",
                transport_source="discord",
                format_family="discord_markdown",
                max_recommended_chars=1700,
                hard_limit_chars=1900,
            ),
            _profile("whatsapp", format_family="whatsapp/plain_text"),
            _profile(
                "slack",
                transport_source="slack",
                format_family="slack_mrkdwn",
                max_recommended_chars=2700,
                hard_limit_chars=2900,
            ),
            _profile("signal", format_family="signal/body_ranges"),
            _profile("mattermost", render_profile=RENDER_PROFILE_RICH_MARKDOWN, format_family="mattermost/commonmark"),
            _profile("matrix", render_profile=RENDER_PROFILE_RICH_MARKDOWN, format_family="matrix/matrix_html"),
            _profile("home_assistant", format_family="home_assistant/structured_json"),
            _profile("email", render_profile=RENDER_PROFILE_RICH_MARKDOWN, format_family="email/mime"),
            _profile("twilio_sms", format_family="twilio_sms/plain_text"),
            _profile("dingtalk", format_family="dingtalk/card_json"),
            _profile("wecom_websocket", format_family="wecom/card_json"),
            _profile("wecom_callback", format_family="wecom/card_json"),
            _profile("wechat_ilink", format_family="wechat_ilink/plain_text"),
            _profile("feishu_lark", format_family="feishu_lark/post_card_json"),
            _profile("imessage_bluebubbles", format_family="imessage_bluebubbles/attributed_text"),
            _profile("qqbot", format_family="qqbot/markdown"),
            _profile("tencent_yuanbao", format_family="tencent_yuanbao/plain_text"),
            _profile("microsoft_teams", render_profile=RENDER_PROFILE_RICH_MARKDOWN, format_family="microsoft_teams/adaptive_card"),
            _profile("line", format_family="line/flex_message"),
            _profile("simplex", format_family="simplex/plain_text"),
            _profile("api_server", render_profile=RENDER_PROFILE_RICH_MARKDOWN, format_family="api_server/structured_json"),
            _profile("buzz", transport_source="hermes", render_profile=RENDER_PROFILE_RICH_MARKDOWN, format_family="buzz/markdown"),
        )
    }
)
