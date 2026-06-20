"""
超长记忆上下文增强插件
基于 astrbot_plugin_context_enhancer 大幅重构，新增：
- 按天拆分聊天日志 + LLM 每日摘要（超长记忆）
- 群成员画像（好感度、称呼、性格等长期信息）
- DeepSeek 缓存优化架构（分层注入策略，提高 prefix caching 命中率）
- Token 统计（今日用量 + 缓存命中率）
"""
import traceback
import json
import re
import datetime
import heapq
import itertools
from collections import deque, defaultdict
import os
from typing import Dict, Optional
from asyncio import Lock
import time
import uuid
from dataclasses import dataclass
import asyncio
import aiofiles
import aiofiles.os as aio_os
from aiofiles.os import remove as aio_remove, rename as aio_rename

from astrbot.api.event import filter as event_filter, AstrMessageEvent
from astrbot.api.star import Context, Star, register, StarTools
from astrbot.api import logger, AstrBotConfig
from astrbot.api.provider import ProviderRequest
from astrbot.api.message_components import Plain, At, Image, Face, Reply
from astrbot.api.platform import MessageType

# 导入工具模块
try:
    from .utils.image_caption import ImageCaptionUtils
except ImportError:
    ImageCaptionUtils = None
    # _initialize_utils 方法中会记录详细日志


# 消息类型枚举 - 重命名以避免冲突
class ContextMessageType:
    """消息类型枚举"""
    LLM_TRIGGERED = "llm_triggered"
    NORMAL_CHAT = "normal_chat"
    IMAGE_MESSAGE = "image_message"
    BOT_REPLY = "bot_reply"


# 常量定义 - 避免硬编码
class ContextConstants:
    """插件中使用的常量"""
    MESSAGE_MATCH_TIME_WINDOW = 3
    PROMPT_HEADER = "你正在浏览聊天软件，查看群聊消息。"
    RECENT_CHATS_HEADER = "\n最近的聊天记录:"
    BOT_REPLIES_HEADER = "\n你最近的回复:"
    PROMPT_FOOTER = "请基于以上信息，并严格按照你的角色设定，做出自然且符合当前对话氛围的回复。"


@dataclass
class PluginConfig:
    """统一管理插件配置项"""
    enabled_groups: list
    recent_chats_count: int
    bot_replies_count: int
    collect_bot_replies: bool
    max_images_in_context: int
    enable_image_caption: bool
    image_caption_provider_id: str
    image_caption_prompt: str
    image_caption_timeout: int
    cleanup_interval_seconds: int
    inactive_cleanup_days: int
    command_prefixes: list
    duplicate_check_window_messages: int
    duplicate_check_time_seconds: int
    passive_reply_instruction: str  # 被动回复指令
    active_speech_instruction: str  # 主动发言指令
    # 调试日志配置（将 LLM 请求写入本地文件，用于测试）
    debug_log_llm_requests: bool = False
    # 全局群成员画像配置（记录好感度、称呼、性格等长期信息）
    profile_enabled: bool = True
    profile_update_instruction: str = ""  # 自定义画像更新提示词（空=使用默认）
    # 聊天日志文件配置（用于长记忆 + DeepSeek 缓存优化）
    chat_log_enabled: bool = True
    chat_log_max_chars: int = 16000
    chat_log_days: int = 5                 # 保留几天的完整日志
    auto_cleanup_old_logs: bool = True     # 是否自动删除超过 chat_log_days 的旧日志文件
    # 每日摘要配置
    summary_enabled: bool = True           # 是否启用每日摘要
    summary_provider_id: str = ""          # 摘要专用LLM提供商ID（空=用主提供商）
    summary_prompt: str = ""               # 摘要提示词（空=使用默认）
    summary_max_chars: int = 10000         # 摘要文件超过此长度时压缩
    # Token 统计配置
    token_stats_enabled: bool = True       # 是否启用 token 统计日志


@dataclass
class GroupMessageBuffers:
    """为每个群组管理独立的、按类型划分的消息缓冲区"""
    recent_chats: deque
    bot_replies: deque
    image_messages: deque


@dataclass
class DailyTokenStats:
    """今日 token 统计计数器"""
    date: str = ""  # 当天日期，用于判断是否需要重置
    total_input_other: int = 0  # 今日累计非缓存输入 token
    total_input_cached: int = 0  # 今日累计缓存输入 token
    total_output: int = 0  # 今日累计输出 token
    total_requests: int = 0  # 今日累计请求次数

    def reset(self, date: str) -> None:
        """重置统计数据"""
        self.date = date
        self.total_input_other = 0
        self.total_input_cached = 0
        self.total_output = 0
        self.total_requests = 0

    def add_usage(self, input_other: int, input_cached: int, output: int) -> None:
        """添加一次请求的 token 使用量"""
        self.total_input_other += input_other
        self.total_input_cached += input_cached
        self.total_output += output
        self.total_requests += 1

    @property
    def total_tokens(self) -> int:
        """今日总 token 数"""
        return self.total_input_other + self.total_input_cached + self.total_output

    @property
    def cache_hit_rate(self) -> float:
        """缓存命中率（百分比）"""
        total_input = self.total_input_other + self.total_input_cached
        if total_input == 0:
            return 0.0
        return (self.total_input_cached / total_input) * 100


class GroupMessage:
    """群聊消息的独立数据类，与框架解耦"""
    def __init__(self,
                 message_type: str,
                 sender_id: str,
                 sender_name: str,
                 group_id: str,
                 text_content: str = "",
                 images: Optional[list[str]] = None,
                 message_id: Optional[str] = None,
                 nonce: Optional[str] = None,
                 raw_components: Optional[list] = None):
        self.id = message_id
        self.nonce = nonce
        self.message_type = message_type
        self.timestamp = datetime.datetime.now()
        self.sender_id = sender_id
        self.sender_name = sender_name
        self.group_id = group_id
        self.text_content = text_content
        self.images = images or []
        self.has_image = len(self.images) > 0
        self.image_captions: list[str] = []
        self.raw_components = raw_components or []

    def to_dict(self) -> dict:
        """将消息对象转换为可序列化为 JSON 的字典"""
        # 序列化 raw_components
        serializable_components = []
        for comp in self.raw_components:
            if hasattr(comp, 'to_dict'):
                serializable_components.append(comp.to_dict())
            else:
                # 对于没有 to_dict 方法的组件，尝试转换为字符串
                try:
                    # 修复 #3: 改进对未知组件的序列化处理
                    serializable_components.append({"type": comp.__class__.__name__, "content": str(comp)})
                except Exception:
                    serializable_components.append({"type": "unknown", "content": str(comp)})

        return {
            "id": self.id,
            "nonce": self.nonce,
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "sender_name": self.sender_name,
            "sender_id": self.sender_id,
            "group_id": self.group_id,
            "text_content": self.text_content,
            "has_image": self.has_image,
            "image_captions": self.image_captions,
            "images": self.images,  # 直接存储 URL 列表
            "raw_components": serializable_components
        }

    @classmethod
    def from_dict(cls, data: dict):
        """从字典创建 GroupMessage 对象"""
        # 注意：从字典恢复 raw_components 较为复杂，
        # 这里我们只恢复其字典形式，因为原始对象类型信息已丢失。
        # 如果需要完全恢复，需要一个组件工厂函数。
        # 目前的实现对于数据存储和传输是足够的。
       # 修复 #1: 增强向后兼容性，使用 .get() 并提供默认值
        instance = cls(
           message_type=data.get("message_type", ContextMessageType.NORMAL_CHAT),
           sender_id=data.get("sender_id", "unknown"),
           sender_name=data.get("sender_name", "用户"),
           group_id=data.get("group_id", ""),
           text_content=data.get("text_content", ""),
           images=data.get("images", []),
           message_id=data.get("id"),
           nonce=data.get("nonce"),
           raw_components=data.get("raw_components", [])
        )
        # 时间戳是核心字段，如果缺少则可能无法处理，但仍尝试提供默认值
        timestamp_str = data.get("timestamp")
        instance.timestamp = datetime.datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.datetime.now()
        instance.image_captions = data.get("image_captions", [])
       # has_image 属性需要根据恢复的 images 列表重新计算
        instance.has_image = len(instance.images) > 0
        return instance


@register("context_plus", "秋云", "超长记忆上下文增强插件 - 支持DeepSeek缓存优化", "1.0.0")
class ContextPlus(Star):
    """
    超长记忆上下文增强器

    作者: 秋云

    基于 astrbot_plugin_context_enhancer (原作者: 木有知) 大幅重构

    核心功能:
    - 🧠 超长记忆系统：按天拆分日志 + LLM 每日摘要 + 群成员画像
    - 🚀 DeepSeek 缓存优化：分层注入策略，提高 prefix caching 命中率
    - 📊 Token 统计：每次回复后自动统计今日用量和缓存命中率
    - 🎯 短期上下文增强：自动收集群聊历史和机器人回复记录
    - 🖼️ 图片描述支持（可选）
    - 🛡️ 安全兼容，不覆盖 system_prompt，不干扰其他插件

    技术保证:
    - 不影响 system_prompt，完全兼容人设系统
    - 使用合理优先级，不干扰其他插件
    - 异步处理，不阻塞主流程
    - 完善的错误处理和功能降级
    """
    # 缓冲区大小乘数，用于为 deque 提供额外空间，避免在消息快速增长时频繁丢弃旧消息
    CACHE_LOAD_BUFFER_MULTIPLIER = 2

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context, config)
        self.raw_config = config
        self.config = self._load_plugin_config()
        self._global_lock = asyncio.Lock()
        logger.info("[ContextPlus] 上下文增强器v2.0已初始化")

        # 初始化工具类
        self._initialize_utils()

        # 群聊消息缓存 - 每个群独立存储
        self.group_messages: Dict[str, "GroupMessageBuffers"] = {}
        self.group_locks: defaultdict[str, Lock] = defaultdict(Lock)
        self.group_last_activity: Dict[str, datetime.datetime] = {}
        self.last_cleanup_time = time.time()

        # 今日 token 统计计数器
        self.daily_token_stats = DailyTokenStats(date=datetime.datetime.now().strftime("%Y-%m-%d"))

        # 异步加载持久化的上下文
        # 使用 StarTools.get_data_dir() 自动获取插件专属数据目录
        self.data_dir = StarTools.get_data_dir()
        os.makedirs(self.data_dir, exist_ok=True)
        self.cache_path = os.path.join(self.data_dir, "context_cache.json")

        # 聊天日志目录（用于长记忆文件存储）
        self.chat_log_dir = os.path.join(self.data_dir, "chat_logs")
        os.makedirs(self.chat_log_dir, exist_ok=True)
        
        # 显示当前配置
        logger.info(f"上下文增强器配置加载完成: {self.config}")

    async def _async_init(self):
        """异步初始化部分，例如加载缓存"""
        await self._load_cache_from_file()
        logger.info(f"成功从 {self.cache_path} 异步加载上下文缓存")

    async def terminate(self, context: Context):
        """插件终止时，异步持久化上下文并关闭会话"""
        # 异步持久化上下文
        temp_path = self.cache_path + ".tmp"
        try:
            serializable_data = {}
            for group_id, buffers in self.group_messages.items():
                # 使用 heapq.merge 高效合并已排序的 deques，并立即转换为列表
                all_messages = list(heapq.merge(
                    buffers.recent_chats, buffers.bot_replies, buffers.image_messages, key=lambda msg: msg.timestamp
                ))

                # 在保存前，根据配置裁剪消息列表，防止缓存文件无限增长
                max_messages_to_save = self.config.recent_chats_count + self.config.bot_replies_count
                if len(all_messages) > max_messages_to_save:
                    all_messages = all_messages[-max_messages_to_save:]

                # 序列化
                serializable_data[group_id] = [msg.to_dict() for msg in all_messages]

            # 1. 写入临时文件
            async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(serializable_data, ensure_ascii=False, indent=4))

            # 2. 原子性重命名
            await aio_rename(temp_path, self.cache_path)
            logger.info(f"上下文缓存已成功原子化保存到 {self.cache_path}")

        except Exception as e:
            logger.error(f"[ContextPlus] 异步保存上下文缓存失败: {e}")
        finally:
            # 3. 确保清理临时文件
            if await aio_os.path.exists(temp_path):
                try:
                    await aio_remove(temp_path)
                except Exception as e:
                    logger.error(f"[ContextPlus] 清理临时缓存文件 {temp_path} 失败: {e}")

        # 关闭 aiohttp session
        if self.image_caption_utils and hasattr(self.image_caption_utils, 'close'):
            await self.image_caption_utils.close()
            logger.info("[ContextPlus] ImageCaptionUtils 的 aiohttp session 已关闭。")

    def _load_plugin_config(self) -> PluginConfig:
        """从原始配置加载并填充插件配置类"""
        return PluginConfig(
            enabled_groups=[str(g) for g in self.raw_config.get("enabled_groups", [])],
            recent_chats_count=self.raw_config.get("recent_chats_count", 3),
            bot_replies_count=self.raw_config.get("bot_replies_count", 2),
            max_images_in_context=self.raw_config.get("max_context_images", 4),
            collect_bot_replies=self.raw_config.get("collect_bot_replies", True),
            enable_image_caption=self.raw_config.get("enable_image_caption", True),
            image_caption_provider_id=self.raw_config.get("image_caption_provider_id", ""),
            image_caption_prompt=self.raw_config.get(
                "image_caption_prompt", "请简洁地描述这张图片的主要内容，重点关注与聊天相关的信息"
            ),
            image_caption_timeout=self.raw_config.get("image_caption_timeout", 30),
            cleanup_interval_seconds=self.raw_config.get("cleanup_interval_seconds", 600),
            inactive_cleanup_days=self.raw_config.get("inactive_cleanup_days", 7),
            command_prefixes=self.raw_config.get("command_prefixes", ["/", "!", "！", "#", ".", "。"]),
            duplicate_check_window_messages=self.raw_config.get("duplicate_check_window_messages", 5),
            duplicate_check_time_seconds=self.raw_config.get("duplicate_check_time_seconds", 30),
            passive_reply_instruction=self.raw_config.get("passive_reply_instruction", '现在，群成员 {sender_name} (ID: {sender_id}) 正在对你说话，或者提到了你，TA说："{original_prompt}"\n你需要根据以上聊天记录和你的角色设定，直接回复该用户。（不要回复本消息，这只是个提示）'),
            active_speech_instruction=self.raw_config.get("active_speech_instruction", '以上是最近的聊天记录。现在，你决定主动参与讨论，并想就以下内容发表你的看法："{original_prompt}"\n你需要根据以上聊天记录和你的角色设定，自然地切入对话。（不要回复本消息，这只是个提示）'),
            debug_log_llm_requests=self.raw_config.get("debug_log_llm_requests", False),
            profile_enabled=self.raw_config.get("profile_enabled", True),
            profile_update_instruction=self.raw_config.get("profile_update_instruction", ""),
            chat_log_enabled=self.raw_config.get("chat_log_enabled", True),
            chat_log_max_chars=self.raw_config.get("chat_log_max_chars", 16000),
            chat_log_days=self.raw_config.get("chat_log_days", 5),
            auto_cleanup_old_logs=self.raw_config.get("auto_cleanup_old_logs", True),
            summary_enabled=self.raw_config.get("summary_enabled", True),
            summary_provider_id=self.raw_config.get("summary_provider_id", ""),
            summary_prompt=self.raw_config.get("summary_prompt", ""),
            summary_max_chars=self.raw_config.get("summary_max_chars", 10000),
            token_stats_enabled=self.raw_config.get("token_stats_enabled", True),
        )

    def _initialize_utils(self):
        """初始化工具模块"""
        try:
            if ImageCaptionUtils is not None:
                self.image_caption_utils = ImageCaptionUtils(
                    self.context, self.raw_config
                )
                logger.debug("[ContextPlus] ImageCaptionUtils 初始化成功")
            else:
                self.image_caption_utils = None
                logger.warning("[ContextPlus] ImageCaptionUtils 不可用，将使用基础图片处理")
        except Exception as e:
            logger.error(f"[ContextPlus] 工具类初始化失败: {e}")
            self.image_caption_utils = None

    async def _append_to_chat_log(self, group_id: str, line: str) -> None:
        """追加一行到当前日期的聊天日志文件。

        每天一个独立文件: {group_id}/2026-06-11.log
        新行追加在末尾。文件路径按群组+日期隔离。
        """
        if not self.config.chat_log_enabled:
            return
        try:
            group_dir = self._get_group_log_dir(group_id)
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            log_path = os.path.join(group_dir, f"{today}.log")
            async with aiofiles.open(log_path, "a", encoding="utf-8") as f:
                await f.write(line + "\n")
        except Exception as e:
            logger.error(f"[ContextPlus] 写入聊天日志失败 (群 {group_id}): {e}")

    def _get_group_log_dir(self, group_id: str) -> str:
        """获取群组日志目录，按群组隔离"""
        group_dir = os.path.join(self.chat_log_dir, group_id)
        os.makedirs(group_dir, exist_ok=True)
        return group_dir

    def _get_summary_file_path(self, group_id: str) -> str:
        """获取摘要文件路径"""
        return os.path.join(self._get_group_log_dir(group_id), "_summary.log")

    def _get_profile_file_path(self, group_id: str) -> str:
        """获取全局群成员画像文件路径"""
        return os.path.join(self._get_group_log_dir(group_id), "_profile.md")

    async def _read_profile(self, group_id: str) -> str:
        """读取全局群成员画像文件"""
        if not self.config.profile_enabled:
            return ""
        profile_path = self._get_profile_file_path(group_id)
        if not await aio_os.path.exists(profile_path):
            return ""
        try:
            async with aiofiles.open(profile_path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"[ContextPlus] 读取群成员画像失败 (群 {group_id}): {e}")
            return ""

    async def _write_profile(self, group_id: str, content: str) -> None:
        """写入全局群成员画像文件"""
        if not self.config.profile_enabled:
            return
        try:
            profile_path = self._get_profile_file_path(group_id)
            async with aiofiles.open(profile_path, "w", encoding="utf-8") as f:
                await f.write(content.strip() + "\n")
            logger.info(f"[ContextPlus] 群成员画像已更新 (群 {group_id[:12]}...)")
        except Exception as e:
            logger.error(f"[ContextPlus] 写入群成员画像失败 (群 {group_id}): {e}")

    async def _append_to_summary_file(self, group_id: str, summary_line: str) -> None:
        """追加一行到摘要文件"""
        try:
            summary_path = self._get_summary_file_path(group_id)
            async with aiofiles.open(summary_path, "a", encoding="utf-8") as f:
                await f.write(summary_line + "\n")
        except Exception as e:
            logger.error(f"[ContextPlus] 写入摘要文件失败 (群 {group_id}): {e}")

    async def _read_summary_file(self, group_id: str) -> str:
        """读取摘要文件全部内容"""
        summary_path = self._get_summary_file_path(group_id)
        if not await aio_os.path.exists(summary_path):
            return ""
        try:
            async with aiofiles.open(summary_path, "r", encoding="utf-8") as f:
                return await f.read()
        except Exception as e:
            logger.error(f"[ContextPlus] 读取摘要文件失败 (群 {group_id}): {e}")
            return ""

    async def _read_chat_log_last_lines(self, group_id: str) -> str:
        """读取聊天日志文件的内容，返回格式化字符串。

        缓存优化型读取策略（DeepSeek prefix caching）：
        1. 摘要文件 (_summary.log) → 始终加载，最前面，长期稳定
        2. 当天日志 → 始终加载完整（末尾，唯一每天变化的部分）
        3. 昨天 → 前天 → ... → 从昨天往前，逐天尝试加入历史详细日志，
           累计（当天不计入预算）不超过 chat_log_max_chars 字符。
           一旦某天超限，该天及之前所有天都跳过（只用摘要提供记忆）。

        这样，每天凌晨跨天时前缀只变化一次（新增"昨天"日志块），
        之后全天前缀稳定 → 高缓存命中率。
        """
        if not self.config.chat_log_enabled:
            return ""
        parts = []

        # 1. 每日摘要文件（始终加载，不计入字符预算，最前面 → 最稳定缓存块）
        if self.config.summary_enabled:
            summary_content = await self._read_summary_file(group_id)
            if summary_content and summary_content.strip():
                parts.append(f"<historical_summary>\n{summary_content.strip()}\n</historical_summary>")

        # 2. 全局群成员画像（紧随摘要之后。摘要天级变化，画像按需更新，
        #    两者独立变化互不影响缓存。放在历史日志之前确保前缀稳定。）
        profile_content = await self._read_profile(group_id)
        if profile_content and profile_content.strip():
            parts.append(f"<group_profile>\n{profile_content.strip()}\n</group_profile>")

        # 3. 当天日志必须全部加载（不计入预算，放在末尾）
        today = datetime.datetime.now()
        today_str = today.strftime("%Y-%m-%d")
        today_log_path = os.path.join(self._get_group_log_dir(group_id), f"{today_str}.log")
        today_formatted = ""
        if await aio_os.path.exists(today_log_path):
            try:
                async with aiofiles.open(today_log_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                if content and content.strip():
                    today_formatted = f"<chat_logs_{today_str}>\n{content.strip()}\n</chat_logs_{today_str}>"
            except Exception as e:
                logger.error(f"[ContextPlus] 读取当天日志失败 (群 {group_id}): {e}")

        # 4. 从昨天开始往前遍历，累计预算内尽可能加载历史日志
        # 预算控制阶段：从新到旧排列，优先加载最近的日志
        daily_logs = []  # 从新到旧排列（用于预算控制）
        accumulated_size = 0  # 历史详细日志累计大小（不含当天）

        for day_offset in range(1, self.config.chat_log_days):  # 1=昨天, 2=前天, ...
            date_str = (today - datetime.timedelta(days=day_offset)).strftime("%Y-%m-%d")
            log_path = os.path.join(self._get_group_log_dir(group_id), f"{date_str}.log")
            if not await aio_os.path.exists(log_path):
                continue
            try:
                async with aiofiles.open(log_path, "r", encoding="utf-8") as f:
                    content = await f.read()
                if not content or not content.strip():
                    continue
                formatted = f"<chat_logs_{date_str}>\n{content.strip()}\n</chat_logs_{date_str}>"
                # 如果加上这一天会超限，则跳过该天及之前所有天
                if accumulated_size + len(formatted) > self.config.chat_log_max_chars:
                    break
                daily_logs.append(formatted)
                accumulated_size += len(formatted)
            except Exception as e:
                logger.error(f"[ContextPlus] 读取日志失败 (群 {group_id}, 日期 {date_str}): {e}")

        # 5. 拼装：摘要 → 画像 → 历史日志（时间正序）→ 当天日志
        summary_part = "\n\n".join(parts) if parts else ""
        # 反转daily_logs，使其从旧到新排列（时间正序，便于AI理解）
        history_part = "\n\n".join(reversed(daily_logs)) if daily_logs else ""

        result_parts = [p for p in [summary_part, history_part, today_formatted] if p]
        if not result_parts:
            return ""

        return "\n\n".join(result_parts)

    async def _summarize_previous_day(self, group_id: str, date_str: str, chat_text: str) -> str:
        """调用 LLM 对指定日期的聊天记录生成摘要。

        过滤日常吹水，只保留有思考、有信息的内容。
        """
        if not chat_text or not chat_text.strip():
            return ""

        default_prompt = (
            "你是一个群聊记录摘要生成器。请分析以下聊天记录，提取出重要的信息。\n\n"
            "要求：\n"
            "1. 过滤掉日常打招呼（早安/晚安/吃了没等）、表情包、纯水聊\n"
            "2. 保留有实质内容的讨论、决策、问题解答、重要事件\n"
            "3. 每条摘要保持简洁，一句话到两句话\n"
            "4. 如果当天没有重要内容，返回空字符串\n"
            "5. 不要添加额外的解释，直接输出摘要内容\n\n"
        )
        summary_instruction = self.config.summary_prompt if self.config.summary_prompt else default_prompt

        prompt = f"{summary_instruction}\n\n日期: {date_str}\n群聊记录:\n{chat_text}"

        try:
            # 获取摘要用的 provider
            if self.config.summary_provider_id:
                provider = self.context.get_provider_by_id(self.config.summary_provider_id)
            else:
                provider = self.context.get_using_provider()
            if not provider:
                logger.warning(f"[ContextPlus] 未找到摘要提供商，跳过日期 {date_str} 的总结")
                return ""

            llm_resp = await provider.text_chat(prompt=prompt, contexts=[])
            summary = (llm_resp.completion_text or "").strip()
            if summary:
                logger.info(f"[ContextPlus] 日期 {date_str} (群 {group_id[:12]}...) 摘要已生成: {summary[:100]}...")
                return summary
            return ""
        except Exception as e:
            logger.error(f"[ContextPlus] 生成日期 {date_str} 摘要失败 (群 {group_id}): {e}")
            return ""

    async def _compress_summary_file(self, group_id: str) -> None:
        """当摘要文件超过 summary_max_chars 时，压缩最旧的条目。

        策略：让 LLM 合并最早的一半条目，保留最新一天的内容不变。
        """
        summary_path = self._get_summary_file_path(group_id)
        if not await aio_os.path.exists(summary_path):
            return
        try:
            async with aiofiles.open(summary_path, "r", encoding="utf-8") as f:
                content = await f.read()
            if len(content) <= self.config.summary_max_chars:
                return

            lines = content.strip().splitlines()
            if len(lines) <= 3:
                return  # 太少条目，不压缩

            # 保留最新的 1 条，压缩剩下的
            keep_newest = 1
            to_compress = lines[:-keep_newest]
            untouched = lines[-keep_newest:]

            compress_text = "\n".join(to_compress)
            compress_prompt = (
                "你是一个群聊摘要压缩器。请将以下多条每日群聊摘要合并成更简洁的版本。\n\n"
                "要求：\n"
                "1. 保留所有重要信息点，不要丢失关键内容\n"
                "2. 合并相似的讨论主题\n"
                "3. 输出比原来更简洁但信息完整的版本\n"
                "4. 每条前面保留日期标记 [YYYY-MM-DD]\n"
                "5. 直接输出压缩结果，不要添加任何解释\n\n"
                f"需要压缩的摘要：\n{compress_text}"
            )

            if self.config.summary_provider_id:
                provider = self.context.get_provider_by_id(self.config.summary_provider_id)
            else:
                provider = self.context.get_using_provider()
            if not provider:
                logger.warning(f"[ContextPlus] 未找到摘要提供商，跳过摘要压缩")
                return

            llm_resp = await provider.text_chat(prompt=compress_prompt, contexts=[])
            compressed = (llm_resp.completion_text or "").strip()
            if not compressed:
                return

            new_content = compressed + "\n" + "\n".join(untouched)
            async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
                await f.write(new_content.strip() + "\n")
            logger.info(f"[ContextPlus] 摘要文件已压缩 (群 {group_id[:12]}...): {len(content)} → {len(new_content)} 字符")
        except Exception as e:
            logger.error(f"[ContextPlus] 压缩摘要文件失败 (群 {group_id}): {e}")

    async def _maybe_summarize_yesterday(self, group_id: str) -> bool:
        """检测是否需要总结昨天，如果需要则执行。

        条件：
        - 启用摘要功能
        - 昨天的日志文件存在
        - 摘要文件中尚未包含昨天的日期标记

        Returns:
            True 表示成功生成了摘要
        """
        if not self.config.summary_enabled or not self.config.chat_log_enabled:
            return False

        yesterday = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        group_dir = self._get_group_log_dir(group_id)
        yesterday_log = os.path.join(group_dir, f"{yesterday}.log")

        if not await aio_os.path.exists(yesterday_log):
            return False

        # 检查摘要文件是否已包含昨天
        summary_content = await self._read_summary_file(group_id)
        if yesterday in summary_content:
            return False  # 已总结过

        # 读取昨天的日志
        try:
            async with aiofiles.open(yesterday_log, "r", encoding="utf-8") as f:
                chat_text = await f.read()
        except Exception as e:
            logger.error(f"[ContextPlus] 读取昨日日志失败 (群 {group_id}): {e}")
            return False

        if not chat_text.strip():
            return False

        # 生成摘要
        summary = await self._summarize_previous_day(group_id, yesterday, chat_text)
        if summary:
            summary_line = f"[{yesterday} 总结] {summary}"
            await self._append_to_summary_file(group_id, summary_line)

            # 超过长度时压缩
            await self._compress_summary_file(group_id)

            # 摘要生成后，清理超过 chat_log_days 的旧日志文件
            if self.config.auto_cleanup_old_logs:
                await self._cleanup_old_log_files_for_group(group_id)

            # 摘要更新后顺带更新群成员画像（也基于最近聊天记录分析）
            if self.config.profile_enabled:
                asyncio.create_task(self._maybe_update_profile(group_id))

            return True
        return False

    async def _maybe_update_profile(self, group_id: str) -> bool:
        """定期检测并自动更新全局群成员画像。

        条件：
        - 启用画像功能
        - 画像文件不存在（首次创建）或已有画像超过 24 小时未更新

        Returns:
            True 表示成功更新了画像
        """
        if not self.config.profile_enabled or not self.config.chat_log_enabled:
            return False

        profile_path = self._get_profile_file_path(group_id)
        profile_exists = await aio_os.path.exists(profile_path)

        # 检查是否需要更新：不存在则创建，存在则看是否超过 24 小时
        if profile_exists:
            try:
                stat = await aio_os.stat(profile_path)
                last_modified = datetime.datetime.fromtimestamp(stat.st_mtime)
                hours_since_update = (datetime.datetime.now() - last_modified).total_seconds() / 3600
                if hours_since_update < 24:
                    return False  # 24 小时内刚更新过，跳过
            except Exception:
                pass

        # 读取聊天记录 + 已有画像
        chat_log_content = await self._read_chat_log_last_lines(group_id)
        existing_profile = await self._read_profile(group_id)

        if not chat_log_content and not existing_profile:
            return False  # 既无聊天记录也无已有画像，没法分析

        # 获取 LLM provider
        if self.config.summary_provider_id:
            provider = self.context.get_provider_by_id(self.config.summary_provider_id)
        else:
            provider = self.context.get_using_provider()
        if not provider:
            logger.warning(f"[ContextPlus] 未找到提供商，跳过画像自动更新 (群 {group_id[:12]}...)")
            return False

        # 构建提示词
        default_instruction = (
            "你是群成员画像分析器。请分析以下聊天记录，提取每个活跃群成员的画像信息。\n\n"
            "要求：\n"
            "1. 对每个活跃成员，输出格式为：\n"
            "   【昵称】好感度: N/10 | 称呼偏好: xxx | 性格特点: xxx | 兴趣爱好: xxx | 注意事项: xxx\n"
            "2. 好感度根据该成员与 AI 互动时的友善程度打分（初始为 5）\n"
            "3. 称呼偏好：记录该成员喜欢怎么被称呼\n"
            "4. 性格特点：简洁概括\n"
            "5. 兴趣爱好：从聊天内容中推断\n"
            "6. 注意事项：记录该成员不喜欢什么、雷区等\n"
            "7. 如果已有画像，在已有基础上更新，不要丢失已有信息\n"
            "8. 只输出画像内容，不要额外解释\n"
        )
        instruction = self.config.profile_update_instruction if self.config.profile_update_instruction else default_instruction

        prompt = f"{instruction}\n\n"
        if existing_profile:
            prompt += f"已有的画像（请在此基础上更新）:\n{existing_profile}\n\n"
        prompt += f"聊天记录:\n{chat_log_content[:30000]}"

        try:
            llm_resp = await provider.text_chat(prompt=prompt, contexts=[])
            new_profile = (llm_resp.completion_text or "").strip()
            if new_profile:
                await self._write_profile(group_id, new_profile)
                logger.info(f"[ContextPlus] 群成员画像自动更新完成 (群 {group_id[:12]}...)")
                return True
            else:
                logger.warning(f"[ContextPlus] 画像自动更新返回为空 (群 {group_id[:12]}...)")
                return False
        except Exception as e:
            logger.error(f"[ContextPlus] 画像自动更新失败 (群 {group_id[:12]}...): {e}")
            return False

    def _get_or_create_lock(self, group_id: str) -> Lock:
        return self.group_locks[group_id]

    async def _load_cache_from_file(self):
        """从文件异步加载缓存"""
        if not await aio_os.path.exists(self.cache_path):
            return
        try:
            async with aiofiles.open(self.cache_path, "r", encoding="utf-8") as f:
                content = await f.read()
                if content: # 确保文件内容不为空
                    data = json.loads(content)
                    self.group_messages = self._load_group_messages_from_dict(data)
                    logger.info(f"[ContextPlus] 成功从 {self.cache_path} 异步加载上下文缓存。")
                else:
                    logger.info(f"[ContextPlus] 缓存文件 {self.cache_path} 为空，跳过加载。")
        except Exception as e:
            logger.error(f"[ContextPlus] 异步加载上下文缓存失败: {e}")

    def _load_group_messages_from_dict(
        self, data: Dict[str, list]
    ) -> Dict[str, "GroupMessageBuffers"]:
        """从字典加载群组消息到新的多缓冲区结构"""
        group_buffers_map = {}

        for group_id, msg_list in data.items():
            # 为每个群组创建独立的缓冲区
            buffers = self._create_new_group_buffers()

            for msg_data in msg_list:
                try:
                    msg = GroupMessage.from_dict(msg_data)
                    # 根据消息类型和内容分发到对应的 deque
                    if msg.message_type == ContextMessageType.BOT_REPLY:
                        buffers.bot_replies.append(msg)
                    elif msg.has_image:
                        buffers.image_messages.append(msg)
                    else:
                        buffers.recent_chats.append(msg)
                except Exception as e:
                    logger.warning(f"[ContextPlus] 从字典转换并分发消息失败 (群 {group_id}): {e}")
            group_buffers_map[group_id] = buffers
        return group_buffers_map

    def _create_new_group_buffers(self) -> "GroupMessageBuffers":
        """创建一个新的 GroupMessageBuffers 实例，并根据配置初始化 deques"""
        # 为每个 deque 设置独立的 maxlen，并增加一定的缓冲空间
        return GroupMessageBuffers(
            recent_chats=deque(maxlen=self.config.recent_chats_count * self.CACHE_LOAD_BUFFER_MULTIPLIER),
            bot_replies=deque(maxlen=self.config.bot_replies_count * self.CACHE_LOAD_BUFFER_MULTIPLIER),
            image_messages=deque(maxlen=self.config.max_images_in_context * self.CACHE_LOAD_BUFFER_MULTIPLIER)
        )

    async def _get_or_create_group_buffers(self, group_id: str) -> "GroupMessageBuffers":
        """获取或创建群聊的消息缓冲区集合"""
        current_dt = datetime.datetime.now()

        # 更新活动时间
        self.group_last_activity[group_id] = current_dt

        # 基于时间的缓存清理
        now = time.time()
        if now - self.last_cleanup_time > self.config.cleanup_interval_seconds:
            await self._cleanup_inactive_groups(current_dt)
            self.last_cleanup_time = now

        if group_id not in self.group_messages:
            async with self._global_lock:
                # 双重检查，防止在等待锁期间其他协程已创建
                if group_id not in self.group_messages:
                    self.group_messages[group_id] = self._create_new_group_buffers()
        return self.group_messages[group_id]

    async def _cleanup_inactive_groups(self, current_time: datetime.datetime):
        """清理超过配置天数未活跃的群组缓存，并清理旧日志文件
        
        清理策略：
        1. 先清理数据缓存（group_messages, group_last_activity）
        2. 延迟清理锁对象（等待所有线程释放锁后再删除）
        """
        logger.info("开始清理不活跃群组...")
        inactive_threshold = datetime.timedelta(
            days=self.config.inactive_cleanup_days
        )
        inactive_groups = []

        # 这个循环是安全的，因为它只读取 self.group_last_activity
        for group_id, last_activity in list(self.group_last_activity.items()):
            if current_time - last_activity > inactive_threshold:
                inactive_groups.append(group_id)

        if inactive_groups:
            logger.info(f"准备清理 {len(inactive_groups)} 个不活跃的群组上下文缓存...")
            
            # 第一步：清理数据缓存（不清理锁）
            async with self._global_lock:
                for group_id in inactive_groups:
                    self.group_messages.pop(group_id, None)
                    self.group_last_activity.pop(group_id, None)
            
            # 第二步：延迟清理锁对象（等待所有线程释放锁）
            # 等待1秒，确保所有持有锁的线程完成操作并释放锁
            await asyncio.sleep(1.0)
            
            async with self._global_lock:
                for group_id in inactive_groups:
                    self.group_locks.pop(group_id, None)
            
            logger.info(f"不活跃群组上下文缓存清理完毕，共清理 {len(inactive_groups)} 个。")
        else:
            logger.info("没有不活跃的群组需要清理。")

        # 清理超过 chat_log_days 的旧日志文件
        if self.config.auto_cleanup_old_logs and self.config.chat_log_enabled:
            await self._cleanup_all_old_log_files(current_time)

    async def _cleanup_all_old_log_files(self, current_time: datetime.datetime):
        """清理所有群组超过 chat_log_days 的旧日志文件

        只删除 YYYY-MM-DD.log 格式的文件，保留 _summary.log 和 _profile.md
        """
        cutoff_date = current_time - datetime.timedelta(days=self.config.chat_log_days)

        deleted_count = 0
        try:
            if not await aio_os.path.exists(self.chat_log_dir):
                return

            # 遍历所有群组目录
            for group_dir_name in os.listdir(self.chat_log_dir):
                group_dir = os.path.join(self.chat_log_dir, group_dir_name)
                if not os.path.isdir(group_dir):
                    continue

                # 遍历群组目录下的日志文件
                for file_name in os.listdir(group_dir):
                    # 只删除日期格式的日志文件（YYYY-MM-DD.log）
                    if not file_name.endswith(".log") or file_name.startswith("_"):
                        continue

                    # 解析日期
                    date_str = file_name.replace(".log", "")
                    try:
                        file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                        if file_date < cutoff_date:
                            file_path = os.path.join(group_dir, file_name)
                            await aio_remove(file_path)
                            deleted_count += 1
                            logger.debug(f"[ContextPlus] 已删除旧日志文件: {file_path}")
                    except ValueError:
                        # 文件名不是日期格式，跳过
                        continue

            if deleted_count > 0:
                logger.info(f"[ContextPlus] 已清理 {deleted_count} 个超过 {self.config.chat_log_days} 天的旧日志文件")
        except Exception as e:
            logger.error(f"[ContextPlus] 清理旧日志文件失败: {e}")

    async def _cleanup_old_log_files_for_group(self, group_id: str) -> None:
        """清理指定群组超过 chat_log_days 的旧日志文件

        只删除 YYYY-MM-DD.log 格式的文件，保留 _summary.log 和 _profile.md
        在摘要生成后调用，只清理当前群组
        """
        cutoff_date = datetime.datetime.now() - datetime.timedelta(days=self.config.chat_log_days)

        deleted_count = 0
        try:
            group_dir = self._get_group_log_dir(group_id)
            if not await aio_os.path.exists(group_dir):
                return

            # 遍历群组目录下的日志文件
            for file_name in os.listdir(group_dir):
                # 只删除日期格式的日志文件（YYYY-MM-DD.log）
                if not file_name.endswith(".log") or file_name.startswith("_"):
                    continue

                # 解析日期
                date_str = file_name.replace(".log", "")
                try:
                    file_date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                    if file_date < cutoff_date:
                        file_path = os.path.join(group_dir, file_name)
                        await aio_remove(file_path)
                        deleted_count += 1
                        logger.debug(f"[ContextPlus] 已删除旧日志文件: {file_path}")
                except ValueError:
                    # 文件名不是日期格式，跳过
                    continue

            if deleted_count > 0:
                logger.info(f"[ContextPlus] 群组 {group_id[:12]}... 已清理 {deleted_count} 个超过 {self.config.chat_log_days} 天的旧日志文件")
        except Exception as e:
            logger.error(f"[ContextPlus] 清理群组 {group_id} 旧日志文件失败: {e}")

    def is_chat_enabled(self, event: AstrMessageEvent) -> bool:
        """检查当前聊天是否启用增强功能"""
        if event.get_message_type() == MessageType.FRIEND_MESSAGE:
            return True  # 简化版本默认启用私聊
        
        group_id = event.get_group_id()
        logger.debug(f"[ContextPlus] 群聊启用检查: 群ID={group_id}, 启用列表={self.config.enabled_groups}")
        
        # 如果启用列表为空，则对所有群组生效；否则，检查 group_id 是否在列表中
        return not self.config.enabled_groups or group_id in self.config.enabled_groups

    @event_filter.platform_adapter_type(event_filter.PlatformAdapterType.ALL)
    async def on_message(self, event: AstrMessageEvent):
        """监听所有消息，进行分类和存储"""
        start_time = time.monotonic()
        group_id = event.get_group_id()
        if event.get_message_type() == MessageType.GROUP_MESSAGE and not group_id:
            logger.warning("[ContextPlus] 事件缺少 group_id，无法处理。")
            return
        
        try:
            if not self.is_chat_enabled(event):
                return

            # 检查是否是 reset 命令
            message_text = (event.message_str or "").strip()
            if message_text.lower() in ["reset", "new"]:
                await self.handle_clear_context_command(event)
                return

            if event.get_message_type() == MessageType.GROUP_MESSAGE:
                await self._handle_group_message(event)

        except Exception as e:
            logger.error(f"[ContextPlus] 处理消息时发生错误: {e}")
            logger.error(f"[ContextPlus] {traceback.format_exc()}")
        finally:
            duration = (time.monotonic() - start_time) * 1000
            logger.debug(f"[Profiler] on_message for group {group_id} took: {duration:.2f} ms")

    def _extract_user_info_from_event(self, event: AstrMessageEvent) -> tuple[str, str]:
        """
        从事件中提取用户ID和昵称的统一方法
        返回: (sender_name, sender_id)
        """
        # 1. 优先使用标准方法
        sender_name = event.get_sender_name()
        sender_id = event.get_sender_id()

        # 2. 如果标准方法失败，尝试从 message_obj.sender 获取
        if not sender_name or not sender_id:
            message_obj = getattr(event, 'message_obj', None)
            if message_obj and hasattr(message_obj, 'sender') and message_obj.sender:
                sender = message_obj.sender
                if not sender_name and hasattr(sender, 'nickname'):
                    sender_name = sender.nickname
                if not sender_id and hasattr(sender, 'user_id'):
                    sender_id = str(sender.user_id)

        # 3. 如果仍然失败，尝试从原始事件数据中获取 (兼容性)
        if not sender_name or not sender_id:
            raw_event = getattr(event, 'raw_event', None)
            if raw_event and isinstance(raw_event.get("sender"), dict):
                raw_sender = raw_event["sender"]
                if not sender_name:
                    sender_name = raw_sender.get("card") or raw_sender.get("nickname")
                if not sender_id:
                    sender_id = str(raw_sender.get("user_id") or raw_sender.get("id"))

        # 4. 最后使用后备值
        return sender_name or "用户", sender_id or "unknown"

    async def _get_image_captions(self, images: list[str]) -> list[str]:
        """获取图片描述的辅助函数"""
        if not self.config.enable_image_caption or not self.image_caption_utils:
            return ["图片"] * len(images)

        tasks = []
        for image_url in images:
            if image_url:
                task = self.image_caption_utils.generate_image_caption(
                    image_url,
                    timeout=self.config.image_caption_timeout,
                    provider_id=self.config.image_caption_provider_id or None,
                    custom_prompt=self.config.image_caption_prompt,
                )
                tasks.append(task)

        captions = []
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results:
                if isinstance(res, Exception):
                    logger.debug(f"[ContextPlus] 生成图片描述失败: {res}")
                    captions.append("图片内容未知")
                else:
                    captions.append(res or "图片内容未知")
        
        return captions

    async def _create_group_message_from_event(self, event: AstrMessageEvent, message_type: str) -> GroupMessage:
        """从事件创建 GroupMessage 实例，并在检测到图片时异步获取描述"""
        text_content_parts = []
        images = []
        
        message_obj = getattr(event, 'message_obj', None)
        raw_components = message_obj.message if message_obj and hasattr(message_obj, 'message') else []

        for comp in raw_components:
            if isinstance(comp, Plain):
                text_content_parts.append(comp.text)
            elif isinstance(comp, At):
                text_content_parts.append(f"@{comp.qq}")
            elif isinstance(comp, Face):
                text_content_parts.append(f"[表情]")
            elif isinstance(comp, Reply):
                text_content_parts.append(f"[引用了 {comp.sender_nickname} 的消息]")
            elif isinstance(comp, Image):
                image_url = getattr(comp, "url", None) or getattr(comp, "file", None)
                if image_url:
                    images.append(image_url)

        if images:
            captions = await self._get_image_captions(images)
            text_content_parts.append(f"[Image: {'; '.join(captions)}]")

        final_sender_name, final_sender_id = self._extract_user_info_from_event(event)

        return GroupMessage(
            message_type=message_type,
            sender_id=final_sender_id,
            sender_name=final_sender_name,
            group_id=event.get_group_id(),
            text_content="".join(text_content_parts).strip(),
            images=images,
            message_id=getattr(event, 'id', None) or (message_obj and getattr(message_obj, 'id', None)),
            nonce=getattr(event, '_context_plus_nonce', None),
            raw_components=raw_components
        )

    async def _handle_group_message(self, event: AstrMessageEvent):
        """处理群聊消息"""
        # 现在 create 方法是 async 的，需要 await
        group_msg = await self._create_group_message_from_event(event, "")  # 临时创建以检查内容
        if not group_msg.text_content and not group_msg.has_image: # 检查 has_image 以防万一
            logger.debug("[ContextPlus] 消息为空（无文本无图片），跳过处理。")
            return

        try:
            if self._is_bot_message(event):
                logger.debug("[ContextPlus] 收集到机器人自己的消息，用于保持上下文完整性。")

            message_type = self._classify_message(event)
            group_msg.message_type = message_type # 更新消息类型

            # 获取或创建该群组的缓冲区集合
            buffers = await self._get_or_create_group_buffers(group_msg.group_id)
            lock = self._get_or_create_lock(group_msg.group_id)

            async with lock:
                # 根据消息类型和内容，将其放入对应的 deque
                target_deque = None
                if message_type == ContextMessageType.BOT_REPLY:
                    target_deque = buffers.bot_replies
                # 图片消息现在作为普通聊天处理，因为内容已是文本
                else: # NORMAL_CHAT or LLM_TRIGGERED
                    target_deque = buffers.recent_chats

                # 🚨 防重复机制：检查是否已存在相同消息
                if not self._is_duplicate_message(target_deque, group_msg):
                    target_deque.append(group_msg)
                    logger.debug(
                        f"收集群聊消息 [{message_type}] (群组: {group_msg.group_id}): {group_msg.sender_name} - {group_msg.text_content[:50]}..."
                    )
                else:
                    logger.debug(
                        f"[ContextPlus] 跳过重复消息: {group_msg.sender_name} - {group_msg.text_content[:30]}..."
                    )

            # 追加到聊天日志文件（用于长记忆 + 缓存优化）
            # 只在非重复消息时写入
            if group_msg.text_content:
                timestamp_str = group_msg.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                log_line = f"[{timestamp_str}] {group_msg.sender_name}: {group_msg.text_content}"
                await self._append_to_chat_log(group_msg.group_id, log_line)

        except Exception as e:
            logger.error(f"[ContextPlus] 处理群聊消息时发生错误: {e}")

    def _is_duplicate_message(self, target_deque: deque, new_msg: GroupMessage) -> bool:
        """检查消息是否已存在于目标缓冲区（防重复）"""
        # 如果新消息包含图片，则不视为重复，以确保图片总能被处理
        if new_msg.has_image:
            return False
            
        # 检查最近N条消息即可，避免性能问题
        start_index = max(0, len(target_deque) - self.config.duplicate_check_window_messages)
        recent_messages = list(itertools.islice(target_deque, start_index, len(target_deque)))

        for existing_msg in recent_messages:
            # 重复判断条件：
            # 1. 相同发送者
            # 2. 相同文本内容
            # 3. 时间差在指定窗口内
            if (
                existing_msg.sender_id == new_msg.sender_id and
                existing_msg.text_content == new_msg.text_content and
                abs((new_msg.timestamp - existing_msg.timestamp).total_seconds()) < self.config.duplicate_check_time_seconds
            ):
                return True

        return False

    def _is_bot_message(self, event: AstrMessageEvent) -> bool:
        """检查是否是机器人自己发送的消息"""
        try:
            # 获取机器人自身ID
            bot_id = event.get_self_id()
            sender_id = event.get_sender_id()

            # 如果发送者ID等于机器人ID，则是机器人自己的消息
            return bool(bot_id and sender_id and str(sender_id) == str(bot_id))
        except (AttributeError, KeyError) as e:
            logger.warning(f"[ContextPlus] 检查机器人消息时出错（可能是不支持的事件类型或数据结构）: {e}")
            return False

    def _classify_message(self, event: AstrMessageEvent) -> str:
        """
        分类消息类型，区分直接触发和间接触发。
        新的逻辑流程:
        1. 直接触发 (用户@或指令) -> LLM_TRIGGERED (被动响应)
        2. 间接触发 (wakepro等) -> NORMAL_CHAT (主动发言)
        3. 其他按原逻辑处理
        """
        # 🤖 首先检查是否是机器人自己的消息
        if self._is_bot_message(event) and self.config.bot_replies_count > 0:
            return ContextMessageType.BOT_REPLY

        # 1. 检查是否为用户直接触发
        if self._is_directly_triggered(event):
            # 附加一个唯一标识符，用于后续精确匹配
            setattr(event, '_context_plus_nonce', uuid.uuid4().hex)
            return ContextMessageType.LLM_TRIGGERED

        # 2. 检查是否为间接触发（例如被 wakepro 唤醒）
        # 根据新逻辑，这种情况被视为普通聊天，以体现“主动发言”的角色扮演
        if self._is_indirectly_triggered(event):
            return ContextMessageType.NORMAL_CHAT

        # 3. 如果不是间接触发，也不是机器人自己的消息，那它就是一次需要LLM响应的普通消息
        return ContextMessageType.NORMAL_CHAT


    def _is_at_triggered(self, event: AstrMessageEvent) -> bool:
        """检查消息是否通过@机器人触发"""
        bot_id = event.get_self_id()
        if not bot_id:
            return False

        # 检查消息组件
        if event.message_obj and event.message_obj.message:
            for comp in event.message_obj.message:
                if isinstance(comp, At) and (
                    str(comp.qq) == str(bot_id) or comp.qq == "all"
                ):
                    return True
        
        # 检查纯文本
        message_text = event.message_str or ""
        # 使用正则表达式确保 @<bot_id> 是一个独立的词
        pattern = rf'(^|\s)@{re.escape(str(bot_id))}($|\s)'
        if re.search(pattern, message_text):
            return True

        return False

    def _is_keyword_triggered(self, event: AstrMessageEvent) -> bool:
        """检查消息是否通过命令前缀触发"""
        message_text = (event.message_str or "").lower().strip()
        if not message_text:
            return False

        return any(
            message_text.startswith(prefix)
            for prefix in self.config.command_prefixes
        )

    def _is_directly_triggered(self, event: AstrMessageEvent) -> bool:
        """
        检查消息是否由用户直接触发（@机器人或使用命令词）。
        这代表了最明确的用户交互意图。
        """
        return self._is_at_triggered(event) or self._is_keyword_triggered(event)

    def _is_indirectly_triggered(self, event: AstrMessageEvent) -> bool:
        """
        检查消息是否由间接方式触发（如 wakepro 插件的智能唤醒）。
        这通常不被视为用户直接的对话意图。
        """
        return getattr(event, "is_wake", False) or getattr(
            event, "is_at_or_wake_command", False
        )

    @event_filter.on_llm_request(priority=100)
    async def on_llm_request(self, event: AstrMessageEvent, request: ProviderRequest):
        """
        LLM请求时提供上下文增强。

        架构说明（v2.1 - 缓存优化版）：
        - 聊天日志（长记忆）：追加到 request.system_prompt 末尾
          → 作为 prompt 前缀，利用 DeepSeek prefix caching
          → 文件追加模式，日志前面大部分内容不变，缓存命中率高
        - 短期上下文 + 指令：保留在 request.prompt 中
          → 放在消息末尾，不影响前缀缓存
        """
        start_time = time.monotonic()
        group_id = event.get_group_id()
        if event.get_message_type() == MessageType.GROUP_MESSAGE and not group_id:
            logger.warning(f"[ContextPlus] LLM 请求事件缺少 group_id，无法增强上下文。")
            return
            
        try:
            # 1. 检查是否需要增强
            if not self._should_enhance_context(event, request):
                return

            # 2. 获取群聊历史记录
            group_id = event.get_group_id()
            buffers = await self._get_or_create_group_buffers(group_id)

            # 3. 检测跨天并生成昨日摘要（异步，不阻塞LLM请求）
            #    摘要写入 _summary.log，同时顺带更新群成员画像，下次请求自动生效
            if self.config.summary_enabled:
                asyncio.create_task(self._maybe_summarize_yesterday(group_id))

            # 4. 读取聊天日志文件（长记忆 + 每日摘要，用于缓存优化）
            chat_log_content = await self._read_chat_log_last_lines(group_id)

            # 5. 确定场景（被动回复 vs 主动发言）
            lock = self._get_or_create_lock(group_id)
            async with lock:
                # 合并所有消息用于查找触发消息和短期上下文
                all_messages = list(heapq.merge(
                    buffers.recent_chats, buffers.bot_replies, buffers.image_messages,
                    key=lambda x: x.timestamp
                ))

                triggering_message, scene = self._find_triggering_message_from_event(all_messages, event)

                # 构建短期上下文增强和指令
                context_enhancement, image_urls_for_context = self._build_context_enhancement(
                    all_messages, request.prompt, triggering_message, scene, event
                )

            # 6. 注入到请求
            # 聊天日志 + 历史摘要 → system_prompt 末尾（利用前缀缓存）
            # 指令 + 短期上下文 → prompt（消息末尾，不影响缓存）
            original_prompt = request.prompt  # 保存原始 prompt，用于调试日志
            self._inject_context_into_request(
                request, context_enhancement, image_urls_for_context, chat_log_content
            )

            # 6.1 调试日志：将注入后的完整请求写入本地文件（开关控制）
            await self._log_llm_request_debug(
                group_id, event, request, scene, original_prompt
            )

            # 7. 清空 conversation.history（多轮对话历史）
            # 聊天日志文件已经包含了所有群聊消息+机器人回复，
            # 不再需要将历史多轮对话也发送到 LLM。
            # 此举大幅减少了每次请求的 token 消耗，
            # 且让 prompt 前缀更加固定 → 缓存命中率更高。
            # 注意：req.conversation 保留，不影响 _save_to_history 的保存逻辑。
            if self.config.chat_log_enabled:
                old_len = len(request.contexts) if request.contexts else 0
                request.contexts = []
                if old_len > 0:
                    logger.debug(
                        f"[ContextPlus] 已清空对话历史 "
                        f"(共 {old_len} 条)，由聊天日志文件替代。"
                    )

        except Exception as e:
            logger.error(f"[ContextPlus] 上下文增强时发生错误: {e}")
            logger.error(f"[ContextPlus] {traceback.format_exc()}")
        finally:
            duration = (time.monotonic() - start_time) * 1000
            logger.debug(f"[Profiler] on_llm_request for group {group_id} took: {duration:.2f} ms")

    def _should_enhance_context(self, event: AstrMessageEvent, request: ProviderRequest) -> bool:
        """检查是否应执行上下文增强"""
        return (
            not hasattr(request, '_context_enhanced') and
            self.is_chat_enabled(event) and
            event.get_message_type() == MessageType.GROUP_MESSAGE
        )

    def _extract_messages_for_context(self, sorted_messages: list[GroupMessage]) -> dict:
        """从已排序的合并消息列表中提取和筛选数据"""
        max_chats = self.config.recent_chats_count
        max_bot_replies = self.config.bot_replies_count

        # 使用列表推导式和 islice 高效筛选和截取
        bot_replies = [
            f"你回复了: {msg.text_content}"
            for msg in itertools.islice(
                (m for m in reversed(sorted_messages) if m.message_type == ContextMessageType.BOT_REPLY),
                max_bot_replies
            )
        ]
        
        recent_chats = [
            f"{msg.sender_name}: {msg.text_content}"
            for msg in itertools.islice(
                (m for m in reversed(sorted_messages) if m.message_type != ContextMessageType.BOT_REPLY and m.text_content),
                max_chats
            )
        ]

        # 反转以恢复时序
        return {
            "recent_chats": list(reversed(recent_chats)),
            "bot_replies": list(reversed(bot_replies)),
        }

    def _build_context_enhancement(
        self,
        sorted_messages: list[GroupMessage],
        original_prompt: str,
        triggering_message: Optional[GroupMessage],
        scene: str,
        event: AstrMessageEvent,
    ) -> tuple[str, list[str]]:
        """
        构建要追加到原始提示词的增强内容和图片URL列表。
        返回一个元组: (增强内容字符串, 图片URL列表)
        """
        extracted_data = self._extract_messages_for_context(sorted_messages)

        # 提取图片URL
        image_urls = []
        for msg in sorted_messages:
            if msg.images:
                image_urls.extend(msg.images)
        
        # 限制图片数量
        if len(image_urls) > self.config.max_images_in_context:
            image_urls = image_urls[-self.config.max_images_in_context:]


        # 构建历史聊天记录部分
        history_parts = [ContextConstants.PROMPT_HEADER]
        history_parts.extend(self._format_recent_chats_section(extracted_data["recent_chats"]))
        history_parts.extend(self._format_bot_replies_section(extracted_data["bot_replies"]))
        context_str = "\n".join(part for part in history_parts if part)

        # 根据场景选择并格式化指令
        instruction_prompt = self._format_situation_instruction(
            original_prompt, triggering_message, scene, event
        )

        # 组合成最终的增强内容
        final_enhancement = f"{context_str}\n\n{instruction_prompt}"
        
        return final_enhancement, image_urls

    def _inject_context_into_request(
        self, request: ProviderRequest, context_enhancement: str, image_urls: list[str],
        chat_log_content: str = ""
    ):
        """将聊天日志、短期上下文和图片URL注入到 ProviderRequest 对象中。

        分层注入策略（针对 DeepSeek prefix caching 优化）：
        1. 聊天日志（长记忆）→ request.system_prompt 末尾
           → 位于 prompt 最前面，日志追加模式使前缀稳定 → 高缓存命中率
        2. 短期上下文 + 指令 → request.prompt（保留原替换逻辑）
           → 位于消息末尾，不影响前缀缓存
        """
        # 1. 聊天日志追加到 system_prompt 末尾（利用前缀缓存）
        # 注意：不加 <chat_logs> 外层包装，保持与 heartflow _read_chat_logs_for_judge
        # 返回的格式一致，使两者共享相同的 system_prompt 前缀 → 共用 DeepSeek 缓存。
        if chat_log_content:
            chat_block = f"\n\n{chat_log_content}"
            if request.system_prompt:
                request.system_prompt += chat_block
            else:
                request.system_prompt = chat_block
            logger.debug(
                f"[ContextPlus] 聊天日志已追加到 system_prompt，"
                f"日志长度: {len(chat_log_content)} 字符, "
                f"system_prompt总长度: {len(request.system_prompt)} 字符"
            )

        # 2. 短期上下文 + 指令 → 保持 request.prompt 替换
        # （放在消息末尾，不影响前缀缓存）
        if context_enhancement:
            request.prompt = context_enhancement
            setattr(request, '_context_enhanced', True)
            logger.debug(f"[ContextPlus] 短期上下文注入完成，新prompt长度: {len(request.prompt)}")

        if image_urls:
            if not hasattr(request, 'image_urls') or request.image_urls is None:
                request.image_urls = []
            request.image_urls.extend(image_urls)
            logger.debug(f"[ContextPlus] 向请求中追加了 {len(image_urls)} 张图片URL。")

    async def _log_llm_request_debug(
        self, group_id: str, event: AstrMessageEvent, request: ProviderRequest,
        scene: str, original_prompt: str
    ) -> None:
        """将 LLM 请求的完整信息写入调试日志文件（开关控制）。

        调试日志文件位置: {data_dir}/debug_llm_requests/{group_id}.log
        每次写入包含时间戳、场景、发送者、system_prompt、prompt 等关键信息。
        """
        if not self.config.debug_log_llm_requests:
            return
        try:
            debug_dir = os.path.join(self.data_dir, "debug_llm_requests")
            os.makedirs(debug_dir, exist_ok=True)
            log_path = os.path.join(debug_dir, f"{group_id}.log")

            sender_name = event.get_sender_name() or "未知"
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # 构建日志内容
            separator = "=" * 72
            log_entry = (
                f"{separator}\n"
                f"时间: {timestamp}\n"
                f"群组: {group_id}\n"
                f"发送者: {sender_name}\n"
                f"场景: {scene}\n"
                f"原始消息: {original_prompt}\n"
                f"--- system_prompt ---\n"
                f"{request.system_prompt or '(空)'}\n"
                f"--- prompt ---\n"
                f"{request.prompt or '(空)'}\n"
                f"--- 图片URL ---\n"
                f"{json.dumps(getattr(request, 'image_urls', []), ensure_ascii=False) if getattr(request, 'image_urls', None) else '(无)'}\n"
                f"{separator}\n\n"
            )

            async with aiofiles.open(log_path, "a", encoding="utf-8") as f:
                await f.write(log_entry)

            logger.info(f"[ContextPlus] 调试日志已写入: {log_path}")
        except Exception as e:
            logger.error(f"[ContextPlus] 写入 LLM 请求调试日志失败: {e}")

    def _find_triggering_message_from_event(self, sorted_messages: list[GroupMessage], llm_request_event: AstrMessageEvent) -> tuple[Optional[GroupMessage], str]:
        """
        在 on_llm_request 事件中，从已排序的合并消息列表中根据 nonce 精确查找触发 LLM 调用的消息，并判断场景。
        """
        nonce = getattr(llm_request_event, '_context_plus_nonce', None)

        if not nonce:
            logger.debug(f"[ContextPlus] 事件中未找到 nonce (群组: {llm_request_event.get_group_id()})，判定为'主动发言'")
            return None, "主动发言"

        # 使用 next() 和生成器表达式更高效地查找
        trigger_message = next((msg for msg in reversed(sorted_messages) if msg.nonce == nonce), None)

        if trigger_message:
            logger.debug(f"通过 nonce 成功匹配到触发消息 (群组: {llm_request_event.get_group_id()})，判定为'被动回复'")
        else:
            logger.warning(f"持有 nonce 但在缓冲区中未找到匹配的触发消息 (群组: {llm_request_event.get_group_id()})。仍判定为'被动回复'场景。")
            
        return trigger_message, "被动回复"

    def _format_recent_chats_section(self, recent_chats: list) -> list:
        """格式化最近的聊天记录部分"""
        if not recent_chats:
            return []
        return [ContextConstants.RECENT_CHATS_HEADER] + recent_chats

    def _format_bot_replies_section(self, bot_replies: list) -> list:
        """格式化机器人回复部分"""
        if not bot_replies:
            return []
        return [ContextConstants.BOT_REPLIES_HEADER] + bot_replies

    def _format_situation_instruction(
        self,
        original_prompt: str,
        triggering_message: Optional[GroupMessage],
        scenario: str,
        event: AstrMessageEvent,
    ) -> str:
        """根据场景格式化指令性提示词"""
        if scenario == "被动回复":
            # 修复 #2: 即使 triggering_message 为 None，也使用被动回复模板
            instruction = self.config.passive_reply_instruction

            # 优先从 triggering_message 获取用户信息，如果为空则从当前事件获取
            if triggering_message:
                sender_name = triggering_message.sender_name
                sender_id = triggering_message.sender_id
            else:
                # 使用统一的用户信息提取方法
                sender_name, sender_id = self._extract_user_info_from_event(event)

            return instruction.format(
                sender_name=sender_name,
                sender_id=sender_id,
                original_prompt=original_prompt,
            )
        else:
            # 默认为主动发言
            instruction = self.config.active_speech_instruction
            return instruction.format(
                original_prompt=original_prompt
            )

    @event_filter.on_llm_response(priority=100)
    async def on_llm_response(self, event: AstrMessageEvent, resp):
        """记录机器人的回复内容并统计 token 使用量"""
        try:
            # Token 统计功能
            if self.config.token_stats_enabled:
                await self._update_and_log_token_stats(resp)

            if event.get_message_type() == MessageType.GROUP_MESSAGE:
                group_id = event.get_group_id()

                # 获取回复文本
                response_text = ""
                if hasattr(resp, "completion_text"):
                    response_text = resp.completion_text
                elif hasattr(resp, "text"):
                    response_text = resp.text
                else:
                    response_text = str(resp)

                # 创建机器人回复记录
                bot_name = self.raw_config.get("name", "助手")
                bot_reply = GroupMessage(
                    message_type=ContextMessageType.BOT_REPLY,
                    sender_id=event.get_self_id(),
                    sender_name=bot_name,
                    group_id=group_id,
                    text_content=response_text[:1000]
                )

                buffers = await self._get_or_create_group_buffers(group_id)
                lock = self._get_or_create_lock(group_id)
                async with lock:
                    buffers.bot_replies.append(bot_reply)

                # 同时追加到聊天日志文件（记录bot回复，保持上下文完整性）
                if response_text and self.config.chat_log_enabled:
                    timestamp_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    log_line = f"[{timestamp_str}] {bot_name}: {response_text[:200]}"
                    await self._append_to_chat_log(group_id, log_line)

                logger.debug(f"[ContextPlus] 记录机器人回复: {response_text[:50]}...")

        except Exception as e:
            logger.error(f"[ContextPlus] 记录机器人回复时发生错误: {e}")

    async def _update_and_log_token_stats(self, resp) -> None:
        """更新今日 token 统计并在日志中输出"""
        try:
            # 检查日期是否需要重置
            today = datetime.datetime.now().strftime("%Y-%m-%d")
            if self.daily_token_stats.date != today:
                self.daily_token_stats.reset(today)

            # 获取当前请求的 token 使用信息
            usage = getattr(resp, "usage", None)
            if usage is None:
                return

            # 提取 token 数量
            input_other = getattr(usage, "input_other", 0) or 0
            input_cached = getattr(usage, "input_cached", 0) or 0
            output = getattr(usage, "output", 0) or 0

            # 更新今日统计
            self.daily_token_stats.add_usage(input_other, input_cached, output)

            # 计算当前请求的缓存命中率
            current_total_input = input_other + input_cached
            current_cache_rate = 0.0
            if current_total_input > 0:
                current_cache_rate = (input_cached / current_total_input) * 100

            # 输出日志
            logger.info(
                f"📊 Token 统计 | "
                f"本次: 输入={input_other + input_cached} (缓存={input_cached}, {current_cache_rate:.1f}%) | "
                f"输出={output} | "
                f"今日累计: {self.daily_token_stats.total_tokens} tokens "
                f"(缓存命中率={self.daily_token_stats.cache_hit_rate:.1f}%) | "
                f"请求次数: {self.daily_token_stats.total_requests}"
            )

        except Exception as e:
            logger.error(f"[ContextPlus] Token 统计时发生错误: {e}")

    async def clear_context_cache(self, group_id: Optional[str] = None):
        """
        清空上下文缓存。
        如果提供了 group_id，则只清空该群组的缓存。
        否则，清空所有群组的缓存。
        """
        try:
            if group_id:
                if group_id in self.group_messages:
                    lock = self._get_or_create_lock(group_id)
                    async with lock:
                        # 使用 pop 安全地移除并返回条目，如果键不存在则返回 None，避免错误
                        self.group_messages.pop(group_id, None)
                        self.group_locks.pop(group_id, None)
                        self.group_last_activity.pop(group_id, None)
                    logger.info(f"[ContextPlus] 已为群组 {group_id} 清理上下文缓存。")
            else:
                async with self._global_lock:
                    self.group_messages.clear()
                self.group_last_activity.clear()
                logger.info("[ContextPlus] 内存中的所有上下文缓存已清空。")
                if await aio_os.path.exists(self.cache_path):
                    await aio_remove(self.cache_path)
                    logger.info(f"[ContextPlus] 持久化缓存文件 {self.cache_path} 已异步删除。")

        except Exception as e:
            logger.error(f"[ContextPlus] 清空上下文缓存时发生错误: {e}")

    @event_filter.command("reset", "new", description="清空当前群聊的上下文缓存")
    async def handle_clear_context_command(self, event: AstrMessageEvent):
        """处理 reset 和 new 命令，清空特定群组的上下文缓存"""
        group_id = event.get_group_id()
        if group_id:
            logger.info(f"收到为群组 {group_id} 清空上下文的命令...")
            await self.clear_context_cache(group_id=group_id)
        else:
            logger.warning("[ContextPlus] 无法获取 group_id，无法执行定向清空操作。")

    @event_filter.command("profile", description="查看当前群聊的全局群成员画像")
    async def handle_profile_command(self, event: AstrMessageEvent):
        """查看全局群成员画像"""
        group_id = event.get_group_id()
        if not group_id:
            event.set_result(event.plain_result("❌ 请在群聊中使用此命令"))
            return
        if not self.config.profile_enabled:
            event.set_result(event.plain_result("❌ 全局画像功能已关闭"))
            return

        profile = await self._read_profile(group_id)
        if profile and profile.strip():
            event.set_result(event.plain_result(f"📋 当前群成员画像:\n\n{profile.strip()}"))
        else:
            event.set_result(event.plain_result("📋 暂无群成员画像。使用 /profile_update 让 AI 根据聊天记录生成。"))

    @event_filter.permission_type(event_filter.PermissionType.ADMIN)
    @event_filter.command("profile_update", description="让 AI 分析聊天记录，更新全局群成员画像")
    async def handle_profile_update_command(self, event: AstrMessageEvent):
        """让 LLM 分析聊天记录，自动生成/更新全局群成员画像"""
        group_id = event.get_group_id()
        if not group_id:
            event.set_result(event.plain_result("❌ 请在群聊中使用此命令"))
            return
        if not self.config.profile_enabled:
            event.set_result(event.plain_result("❌ 全局画像功能已关闭"))
            return

        await event.set_result(event.plain_result("🔄 正在分析聊天记录，生成群成员画像..."))

        try:
            # 读取最近几天的聊天记录作为分析素材
            chat_log_content = await self._read_chat_log_last_lines(group_id)

            # 读取已有画像（用于增量更新）
            existing_profile = await self._read_profile(group_id)

            # 获取 LLM provider
            provider = self.context.get_using_provider()
            if not provider:
                event.set_result(event.plain_result("❌ 无法获取 LLM 提供商"))
                return

            # 构建提示词
            default_instruction = (
                "你是群成员画像分析器。请分析以下聊天记录，提取每个活跃群成员的画像信息。\n\n"
                "要求：\n"
                "1. 对每个活跃成员，输出格式为：\n"
                "   【昵称】好感度: N/10 | 称呼偏好: xxx | 性格特点: xxx | 兴趣爱好: xxx | 注意事项: xxx\n"
                "2. 好感度根据该成员与 AI 互动时的友善程度打分（初始为 5）\n"
                "3. 称呼偏好：记录该成员喜欢怎么被称呼\n"
                "4. 性格特点：简洁概括，如「开朗/话少/技术宅/爱开玩笑」\n"
                "5. 兴趣爱好：从聊天内容中推断\n"
                "6. 注意事项：记录该成员不喜欢什么、雷区等\n"
                "7. 如果已有画像，在已有基础上更新，不要丢失已有信息\n"
                "8. 只输出画像内容，不要额外解释\n"
            )
            instruction = self.config.profile_update_instruction if self.config.profile_update_instruction else default_instruction

            prompt = f"{instruction}\n\n"
            if existing_profile:
                prompt += f"已有的画像（请在此基础上更新）:\n{existing_profile}\n\n"
            prompt += f"聊天记录:\n{chat_log_content[:30000]}"  # 取最近一段够了

            llm_resp = await provider.text_chat(prompt=prompt, contexts=[])
            new_profile = (llm_resp.completion_text or "").strip()

            if new_profile:
                await self._write_profile(group_id, new_profile)
                event.set_result(event.plain_result(f"✅ 群成员画像已更新:\n\n{new_profile}"))
            else:
                event.set_result(event.plain_result("❌ 生成画像失败，LLM 返回为空"))
        except Exception as e:
            logger.error(f"[ContextPlus] 生成群成员画像失败: {e}")
            event.set_result(event.plain_result(f"❌ 生成画像失败: {e}"))
