import asyncio
import glob
import os
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, Optional


@dataclass
class InterfaceConfig:
    kws_text: str
    kws_enabled: bool
    sv_enabled: bool
    enroll_dir: str


@dataclass
class StateBridge:
    set_processing: Callable[[bool], None]
    clear_segments: Callable[[], None]
    get_flag_kws: Callable[[], int]
    set_flag_kws: Callable[[int], None]
    get_flag_sv_enroll: Callable[[], int]
    set_flag_sv_enroll: Callable[[int], None]
    set_temp_register_name: Callable[[str], None]


class InterfaceLayer:
    """接口层：仅负责音频输入输出和设备侧交互。"""

    def __init__(
        self,
        *,
        audio_task_queue,
        tts_text_queue,
        asr_func: Callable[[str], str],
        extract_pinyin_func: Callable[[str], str],
        speaker_identify_func: Callable[[str], tuple],
        play_wakeup_func: Callable[[str], Awaitable[None]],
        wakeup_file: str,
        config: InterfaceConfig,
        state: StateBridge,
    ):
        self.audio_task_queue = audio_task_queue
        self.tts_text_queue = tts_text_queue
        self.asr_func = asr_func
        self.extract_pinyin_func = extract_pinyin_func
        self.speaker_identify_func = speaker_identify_func
        self.play_wakeup_func = play_wakeup_func
        self.wakeup_file = wakeup_file
        self.config = config
        self.state = state

    async def next_audio(self) -> str:
        while True:
            try:
                return self.audio_task_queue.get_nowait()
            except Exception:
                await asyncio.sleep(0.1)

    def has_enrolled_users(self) -> bool:
        users = glob.glob(os.path.join(self.config.enroll_dir, "*.wav"))
        return len(users) > 0

    async def transcribe(self, audio_path: str) -> str:
        loop = asyncio.get_event_loop()
        # 这里的await会暂时挂起这个任务，等到asr_func执行完毕并返回结果后才继续执行后面的代码
        return await loop.run_in_executor(None, self.asr_func, audio_path)  # 这里会把音频转写成文本，这个ASR是同步的，因此放到线程池中运行

    async def maybe_handle_wakeup(self, raw_text: str) -> bool:
        if not self.config.kws_enabled:
            return False

        pinyin_text = self.extract_pinyin_func(raw_text)
        if self.config.kws_text in pinyin_text:
            self.state.set_flag_kws(1)
            await self.play_wakeup_func(self.wakeup_file)
            return True

        if not self.state.get_flag_kws():
            return True

        return False

    async def verify_speaker(self, audio_path: str) -> Optional[str]:
        if not self.config.sv_enabled:
            return "Guest"

        loop = asyncio.get_event_loop()
        user, _score = await loop.run_in_executor(None, self.speaker_identify_func, audio_path)
        if user == "Unknown":
            await self.tts_text_queue.put("抱歉，我没听出你是谁。")
            self.state.set_flag_kws(0)
            return None
        return user

    async def speak(self, text: str, t0_ref: Optional[float] = None):
        if text and text.strip():
            await self.tts_text_queue.put((text, t0_ref))

    async def speak_plain(self, text: str):
        if text and text.strip():
            await self.tts_text_queue.put(text)


class SkillLayer:
    """技能层：只做意图识别与技能执行，不涉及设备。"""

    def __init__(self, intent_router_service, skill_orchestrator):
        self.intent_router_service = intent_router_service
        self.skill_orchestrator = skill_orchestrator

    async def decide(self, user_text: str):
        loop = asyncio.get_event_loop()
        intent = await loop.run_in_executor(None, self.intent_router_service.route, user_text)
        return await self.skill_orchestrator.dispatch(intent)


class LLMLayer:
    """大脑层：只负责对话上下文构建和语言生成。"""

    def __init__(self, brain):
        self.brain = brain

    async def stream_chat(self, user_text: str, user_id: str, search_ctx: str = ""):
        related_memories = self.brain.recall_memories(user_text, user_id)
        memory_str = f"【关于 {user_id} 的记忆】: {';'.join(related_memories)}" if related_memories else ""

        if search_ctx:
            system_msg = "你叫小千，是一个活泼可爱的语音助手。请结合上下文极简回答用户，不超过30字。"
            temperature = 0.1
        else:
            system_msg = f"你叫小千，是一个活泼可爱的语音助手，现在的对话者是 {user_id}, 回答请简短口语化。"
            temperature = 0.7

        msgs, self.brain.running_summary = await self.brain._build_context_api(
            system_prompt=system_msg,
            user_input=user_text,
            memory_str=memory_str,
            search_ctx=search_ctx,
        )

        async for chunk in self.brain.stream_chat_llm(msgs, temperature=temperature):
            yield chunk

    def post_process(self, user_text: str, full_response: str, user_id: str):
        self.brain._post_process(user_text, full_response, user_id)


class ThreeLayerPipeline:
    """三层总编排：接口层 -> 技能层 -> 大脑层。"""

    def __init__(
        self,
        *,
        interface_layer: InterfaceLayer,
        skill_layer: SkillLayer,
        llm_layer: LLMLayer,
        state: StateBridge,
    ):
        self.interface = interface_layer
        self.skill = skill_layer
        self.llm = llm_layer
        self.state = state
        self.split_punc = ["。", "！", "？", "；", "...", ".", "!", "?"]

    async def _stream_and_speak(self, user_text: str, user_id: str, t0_start: float, search_ctx: str = "") -> str:
        full_response = ""
        sentence_buffer = ""
        first_sentence = True

        async for chunk in self.llm.stream_chat(user_text=user_text, user_id=user_id, search_ctx=search_ctx):
            sentence_buffer += chunk
            if any(p in chunk for p in self.split_punc):
                sentence = sentence_buffer.strip()
                if sentence:
                    await self.interface.speak(sentence, t0_ref=t0_start if first_sentence else None)
                    first_sentence = False
                    full_response += sentence_buffer
                sentence_buffer = ""

        if sentence_buffer.strip():
            await self.interface.speak(sentence_buffer.strip(), t0_ref=t0_start if first_sentence else None)
            full_response += sentence_buffer

        return full_response

    async def run(self):
        while True:
            audio_path = await self.interface.next_audio()
            t0_start = time.time()
            self.state.set_processing(True)
            self.state.clear_segments()

            try:
                # 首次注册门禁
                if self.interface.config.sv_enabled and not self.interface.has_enrolled_users() and not self.state.get_flag_sv_enroll():
                    await self.interface.speak_plain("欢迎使用，请说一句话注册声纹。")
                    self.state.set_temp_register_name("主人")
                    self.state.set_flag_sv_enroll(1)
                    self.state.set_processing(False)
                    continue

                raw_text = await self.interface.transcribe(audio_path)
                if not raw_text:
                    self.state.set_processing(False)
                    continue

                wakeup_consumed = await self.interface.maybe_handle_wakeup(raw_text)
                if wakeup_consumed:
                    self.state.set_processing(False)
                    continue

                user_id = await self.interface.verify_speaker(audio_path)
                if not user_id:
                    self.state.set_processing(False)
                    continue

                # 技能层
                skill_result = await self.skill.decide(raw_text)

                if skill_result.handled and skill_result.action == "register_voice":
                    # 兼容旧控制指令
                    command = skill_result.reply or ""
                    if command.startswith("ACTION_REGISTER:"):
                        target_name = command.split(":", 1)[1]
                    else:
                        target_name = "Unknown_User"

                    if target_name == "Unknown_User":
                        await self.interface.speak_plain("请问怎么称呼您？")
                    else:
                        self.state.set_temp_register_name(target_name)
                        self.state.set_flag_sv_enroll(1)
                        await self.interface.speak_plain(f"准备录入{target_name}的声纹，请听到滴声后说话。")
                    self.state.set_processing(False)
                    continue

                if skill_result.handled and skill_result.action == "order_food":
                    await self.interface.speak(skill_result.reply or "抱歉，我暂时无法处理点餐请求。", t0_ref=t0_start)
                    self.llm.post_process(raw_text, skill_result.reply or "", user_id)
                    self.state.set_processing(False)
                    continue

                if skill_result.handled and skill_result.action == "web_search":
                    full_response = await self._stream_and_speak(
                        user_text=raw_text,
                        user_id=user_id,
                        t0_start=t0_start,
                        search_ctx=skill_result.search_context,
                    )
                    self.llm.post_process(raw_text, full_response, user_id)
                    self.state.set_processing(False)
                    continue

                if skill_result.handled:
                    reply = skill_result.reply or "已执行。"
                    await self.interface.speak(reply, t0_ref=t0_start)
                    self.llm.post_process(raw_text, reply, user_id)
                    self.state.set_processing(False)
                    continue

                # 纯对话走大脑层
                full_response = await self._stream_and_speak(
                    user_text=raw_text,
                    user_id=user_id,
                    t0_start=t0_start,
                    search_ctx="",
                )
                self.llm.post_process(raw_text, full_response, user_id)

            except Exception as e:
                print(f"[ThreeLayerPipeline] 处理失败: {e}")
            finally:
                self.state.set_processing(False)
                try:
                    os.remove(audio_path)
                except Exception:
                    pass
