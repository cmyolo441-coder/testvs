"""Original multilingual motion-production method; pure prepared catalog data.

Conceptual source: revfactory/showreel at
64521220a2f8e91cb9894400c48c15b7f45205d6 (MIT, Copyright 2026
Minho Hwang (revfactory)). This method prose is original; the craft references
are attributed translations, and the separately packaged toolkit contains the
upstream engine and explicitly attributed modified MIT companion code.
This module grants no execution, network, installation, or publication authority.
"""
from __future__ import annotations

from .catalog_types import ExpertQuestion, ProcedureCheck, ProcedureStep, SkillDefinition, SkillExample

SHOW_REEL_ARTIFACTS = (
    "show_reel_brief/v1",
    "show_reel_source_asset_register/v1",
    "show_reel_concept_selection/v1",
    "show_reel_storyboard/v1",
    "show_reel_timing_map/v1",
    "show_reel_renderer_handoff/v1",
    "show_reel_preview_review/v1",
    "show_reel_export_manifest/v1",
    "show_reel_qa_report/v1",
    "show_reel_delivery_manifest/v1",
)
_BRIEF, _SOURCES, _CONCEPT, _STORY, _TIMING, _ADAPTER, _PREVIEW, _EXPORT, _QA, _DELIVERY = SHOW_REEL_ARTIFACTS
_INPUTS = ("objective audience and channel", "duration format and budget", "source brand and asset permissions", "available production tools")

DEFINITION = SkillDefinition(
    "show-reel",
    "Motion showcase to make: produce a source-grounded product film with audience-specific concepts, segment direction, authored motion, measured locale/export QA and reproducible source delivery.",
    ("show-reel", "show reel", "showreel", "product showcase video", "animated product film", "motion showcase", "animated portfolio", "launch film", "product teaser video"),
    "Use when a user needs an authored time-based product or work showcase, from evidence and audience framing to rendered motion, localized variants, export inspection, and source delivery.",
    category="design", phase="motion-production", hermes_role="operator",
    delegation_boundary="retained-catalog-intent",
    progressive_disclosure=True,
    required_inputs=_INPUTS,
    expert_questions=(
        ExpertQuestion(_INPUTS[0], "Who should watch this, on which channel, in which content language, and what should they understand or do?", "누가 어떤 채널에서 어떤 언어로 보고 무엇을 이해하거나 행동해야 하나요?"),
        ExpertQuestion(_INPUTS[1], "Which duration, dimensions/aspect ratio, fps, size cap, deadline and revision budget are explicit?", "명시된 길이, 크기와 비율, fps, 용량 상한, 기한과 수정 예산은 무엇인가요?"),
        ExpertQuestion(_INPUTS[2], "Which source revisions, brand assets and usage/voice permissions are approved, and what may be shared?", "승인된 출처 버전, 브랜드 자료와 사용·음성 권한은 무엇이며 어디까지 공유할 수 있나요?"),
        ExpertQuestion(_INPUTS[3], "Which local production tools and licensed fonts are already available in the approved workspace?", "승인된 작업 폴더에서 이미 사용할 수 있는 제작 도구와 라이선스가 확인된 폰트는 무엇인가요?"),
    ),
    expected_outputs=SHOW_REEL_ARTIFACTS,
    why_this_exists="`show-reel` owns the complete motion-production method so factual claims, visual decisions, locale timing, revision budget, measured exports, and delivery receipts cannot drift apart. It does not render inside OMH core.",
    opening_steps=(
        "Read `references/brief-and-locales.md` and `references/storyboard-and-evidence.md` before proposing visuals. Recover objective, audience, channel, duration, aspect ratio, fps, size cap, source/brand permissions, and tool availability from supplied context; honor explicit format and duration, ask only for irretrievable hard facts, and label reversible creative assumptions.",
        "Resolve content language from current explicit user/audience/channel instructions, then applicable stable preferences, then user language. Keep reply_language under host persona and separate content_locale, onscreen_text, subtitle_text, pronunciation_text, voice_policy, font_policy, and direction_policy; never replace literal IDs, commands, URLs, or trademarks.",
        "Compare at least 3 genuinely distinct concept candidates before selection; record the rubric, winner, rejected alternatives, and per-segment art direction. Read `references/craft-workflow.md` (the full upstream production method), `toolkit/README.md` under the loaded skill directory with a file reader, then `references/renderer-adapter.md` before choosing tools; a missing renderer means HOLD on a requested video, not a completed delivery.",
    ),
    handoff_policy="Keep brief, source verification, concept selection, storyboard, critique, and evidence synthesis with the current host. Use only explicitly selected available production tools in an approved workspace. Load `references/procedure.md` for the ordered method and the production references for working templates; the optional local toolkit is declared data, never automatically executed. A prepared handoff is not a render, translation review, listening check, attachment, or publication; never add a core render command.",
    procedure_checks=(
        ProcedureCheck("reel_brief_check", ("objective", "audience_channel", "format_duration_budget", "locale_precedence", "language_tracks", "literal_tokens", "unknowns"), "Require the brief and separate language tracks with decision basis; HOLD on conflicting explicit requirements or missing irretrievable facts, never on an aesthetic choice the author can label as an assumption."),
        ProcedureCheck("reel_source_check", ("claim_source_version", "evidence_class", "brand_verification", "asset_rights", "consent", "conflicts"), "Bind claims to dated versioned sources and assets to rights and consent; HOLD on unresolved brand/source conflicts or rights. Label concept visuals as concept on screen and in the manifest, never as captured product evidence."),
        ProcedureCheck("reel_concept_check", ("candidates", "shared_rubric", "selection", "rejected_alternatives", "revision_budget"), "Require at least 3 distinct narrative and visual approaches, not palette swaps; compare audience fit, evidence clarity, native script, motion intent, feasibility and legibility before selecting one within the revision budget."),
        ProcedureCheck("reel_story_check", ("segments", "source_asset_ids", "art_direction", "motion_intent", "locale_runs", "distinct_compositions"), "Every segment must bind time, message, source/asset IDs, medium, palette, typography, texture, composition, movement and transition, and adjacent segments must differ in at least two of medium, palette, typeface, motion and texture even when one semantic object carries the film; HOLD invented product UI or unsupported claims unless visibly labeled concept."),
        ProcedureCheck("reel_timing_check", ("audio_mode", "measured_cues", "reading_time", "pronunciation_approval", "voice_rights", "sync_status"), "Use measured native-script reading and actual utterance timing, not a fixed character-per-minute formula. When narration is needed require pronunciation, supported locale, voice consent and measured word cues before scene locking; music-only uses beat cues, muted channels need no audio or STT."),
        ProcedureCheck("reel_adapter_check", ("selected_executor", "capabilities", "workspace_scope", "deterministic_time", "commands_timeout_cancel", "partial_resume", "font_readiness", "toolkit_provenance", "source_bound_run"), "A selected renderer needs observed capability and a bounded reproducible recipe with time-based seek, timeout/cancel and partial/resume policy; HOLD missing capability, pageerror or font readiness, and never install tools, download fonts or call paid APIs implicitly."),
        ProcedureCheck("reel_preview_check", ("style_frames", "draft", "transition_strips", "keep_fix_rebuild", "weakest_scene", "revision_spend", "delivery_size_review", "strongest_scene", "anti_slide_review", "composition_variety", "before_after_and_boundaries"), "Inspect actual style frames then a moving draft and transition strips at delivery-size; assign keep/fix/rebuild and revise the weakest scene within a declared iteration/time budget, then re-render and compare. A draft contact sheet where one composition fills most of the runtime is REVISE: rebuild segments by medium and motion. Stills alone cannot pass motion QA."),
        ProcedureCheck("reel_export_check", ("files_hashes", "codec_dimensions_fps_duration", "color_metadata", "filesize_cap", "audio_metrics", "av_sync"), "Record measured hashes, size, duration, fps, codec, color primaries/range and channel constraints; HOLD unsupported codec, oversized output or missing measurements. Audio requires loudness/true peak and A/V sync, with STT only when speech exists; numerical results are not listening evidence."),
        ProcedureCheck("reel_qa_check", ("content_brand", "locale_font_bidi", "cut_motion", "delivery_size_legibility", "temporal_audio", "critical_defects", "disposition", "evidence_refs", "technical_verdict", "artistic_verdict", "playback_verdict", "listening_verdict"), "Return PASS/REVISE/HOLD per check and overall HOLD on any critical or unobserved required check; inspect actual delivery-size playback, cuts, transitions, frozen frames and flashes, native line breaks, shaping and mixed RTL/LTR runs. No linguistic completeness, accessibility safety or human listening claim follows from automated metrics."),
        ProcedureCheck("reel_delivery_check", ("generated_files", "attachment_receipt", "recipient_state", "publication_approval", "source_package", "missing_evidence"), "Separate generated, attached, received and published; a local path proves no transfer. Package reproduction sources, pinned tool versions, locale variants, licenses and QA receipts without secrets/private originals. External upload, spend and publication require exact-target approval and readback; missing requested video or QA keeps delivery HOLD."),
    ),
    procedure_steps=(
        ProcedureStep("reel_brief", "analysis", _INPUTS, (_BRIEF,), ("reel_brief_check",), "Create the audience brief with explicit duration/format, language decision basis, separate reply/content/subtitle/pronunciation/voice/font/direction tracks, literal tokens, budget and missing facts; retrieve what is available before asking."),
        ProcedureStep("reel_verify_sources", "validation", (_INPUTS[0], _INPUTS[2]), (_SOURCES,), ("reel_source_check",), "Verify claim provenance, revision, brand identity, screenshots, font and asset rights; distinguish supplied, observed, vendor-asserted, inferred and concept. Resolve conflicts or HOLD; never fabricate metrics, customers or real demos."),
        ProcedureStep("reel_select_concept", "analysis", (_INPUTS[0], _INPUTS[1], _INPUTS[2]), (_CONCEPT,), ("reel_concept_check",), "Author at least 3 distinct concept candidates, compare one shared rubric, select with reasons and rejected alternatives, and fix the iteration/time budget including weakest-scene revision; do not impose a default aesthetic."),
        ProcedureStep("reel_direct_segments", "production", (_INPUTS[0], _INPUTS[1], _INPUTS[2]), (_STORY,), ("reel_story_check",), "Author each timed segment with message, source/asset IDs, medium, palette, typography, texture, layout, camera/object movement and transition, listing a distinct composition per beat so that adjacent segments differ in at least two of medium, palette, typeface, motion and texture; bind native-script text runs and visibly distinguish concept from captured product evidence."),
        ProcedureStep("reel_measure_timing", "production", (_INPUTS[0], _INPUTS[1], _INPUTS[3]), (_TIMING,), ("reel_timing_check",), "Select muted, music-only or narration mode only as needed; measure reading, approved pronunciation and word/beat cues before locking scene times, keep literal onscreen text separate, and HOLD unsupported voice locales or consent."),
        ProcedureStep("reel_author_motion", "production", _INPUTS, (_ADAPTER,), ("reel_adapter_check",), "Read the installed `toolkit/README.md`, `references/motion-techniques.md`, `references/craft-engine.md` and `references/craft-techniques.md` with ordinary file tools; write scenes fresh from the segment art direction with the engine or another observed host-tool recipe. Bind toolkit provenance and source-bound run receipts, safe inputs/cwd, versions, deterministic seek, commands, timeout/cancel and font readiness; serial capture has no resume/chunk claim. Author motion without hidden installs or core subprocesses."),
        ProcedureStep("reel_review_revise", "validation", (_INPUTS[0], _INPUTS[1], _INPUTS[3]), (_PREVIEW,), ("reel_preview_check",), "Render style frames, then draft playback and transition strips at delivery-size; record keep/fix/rebuild by scene, identify strongest and weakest scenes, reject repeated headline/card/rail grammar, spend the declared revision budget rebuilding the weakest, compare same-time before/after plus both cut boundaries, re-render, and stop at the budget with unresolved defects visible."),
        ProcedureStep("reel_export", "production", (_INPUTS[1], _INPUTS[3]), (_EXPORT,), ("reel_export_check",), "Export actual master/share variants as requested; measure codec, color, dimensions, fps, duration, filesize and hashes, audio loudness/true peak and sync when audio exists. Keep failed and partial outputs distinct from final files."),
        ProcedureStep("reel_validate", "validation", _INPUTS, (_QA,), ("reel_qa_check", "reel_export_check"), "Inspect actual delivery-size playback, content/brand, localized native line breaks, glyph fallback, Arabic shaping/bidi isolation, cuts, motion and audio; return evidence-linked PASS/REVISE/HOLD, keep technical, artistic, full-speed playback and listening verdicts independent; never whole-video PASS from a screenshot, score, metric or preset."),
        ProcedureStep("reel_deliver", "validation", _INPUTS, (_DELIVERY,), ("reel_delivery_check", "reel_qa_check"), "Deliver requested files and source package with selected sources, pinned versions, licenses, locale variants, regeneration recipe and QA receipts; distinguish generated/attached/received/published with exact-target readback and honest missing evidence. HOLD a video request if only a storyboard exists."),
    ),
    artifact_expectations=("Keep media, source files and authored text in the approved project folder; durable OMH records contain only bounded safe references, digests, states and gaps, never raw media, private source text or credentials.",),
    safety_rules=(
        "No automatic network/model/API calls, subprocess execution, dependency installation, font downloads, account discovery, voice cloning or paid defaults inside core; every chosen external action needs scope and authority.",
        "A renderer unavailable, timeout, pageerror, missing font or partial export is HOLD or failed, never successful delivery. Never invent a tool receipt, translation, listening review or product capture.",
        "Use user/official/rights-cleared assets first; generated visuals carry generation provenance and concept labels. Publish/spend/voice cloning requires separate explicit consent and exact-target approval.",
    ),
    quality_tier="evidence-gated",
    quality_bar=(
        "Require at least 3 distinct concept candidates, per-segment art direction and source-grounded motion; never force a fixed palette, visual style, duration or audio track.",
        "Use keep/fix/rebuild critique and a weakest-scene revision budget; reject repeated headline/card/rail grammar, require delivery-size moving evidence and both rebuilt cut boundaries. Technical metrics never fill independent artistic, playback or listening verdicts.",
        "Read `references/assets-and-rights.md`, `references/render-qa-delivery.md` and `references/source-package.md` before export and delivery; missing required evidence keeps HOLD, never fabricated success.",
    ),
    do_not_use_when=(
        "Only a static infographic or summary card is needed; use `img-summary`.",
        "The request is a website, landing page or interactive UI build; use `frontend`.",
        "The output is a deck, PDF, spreadsheet or office document; use `materials-package`.",
        "Only a premium layout quality assessment is wanted; use `design-quality-gate`.",
        "The user already has rendered frames and needs a visual verdict, not new production; use `visual-qa`.",
        "Routine trimming, file conversion, live broadcasting, fishing equipment, tape storage, film criticism and social posting alone do not need this method.",
    ),
    good_example=SkillExample("Create a 20-second muted product showcase video for a Japanese audience using these approved screenshots and English brand names.", "Verify sources and locale, compare three concepts, direct segments, author motion with available tools, inspect the measured export, and deliver sources with honest gaps.", "An authored time-based showcase needs one coherent production method, not only an image prompt or critique."),
    bad_example=SkillExample("Convert this existing clip to a smaller MP4.", "Use a scoped media conversion tool; do not begin concept development.", "An existing file conversion is not an authored showcase."),
    final_checklist=(
        "Brief, source rights, selected concept and per-segment art direction exist; current explicit audience language beats stable preferences and user language, with reply and each content track kept separate.",
        "Actual style frames, draft playback, keep/fix/rebuild, weakest-scene revision and delivery-size checks are observed or HOLD; codec/color/filesize, cuts, motion and locale QA are not inferred from file existence.",
        "Requested video and source package are generated and transferred only when receipts exist; source package is reproducible and rights-cleared, and every missing evidence item remains named.",
    ),
    recovery_notes=(
        "If sources, rights or irretrievable audience facts conflict, HOLD and ask one decision-changing question; label creative assumptions rather than interviewing for every aesthetic choice.",
        "If font, renderer or voice capability is unavailable, return the useful storyboard and source handoff but keep the requested video HOLD; offer a subtitle-only or available-tool alternative without silently changing scope.",
        "If revision budget is exhausted or export/transfer fails, keep the last observed partial state, exact defect and next repair; do not claim completion or retry an external mutation blindly.",
    ),
    situations=("our work needs a compelling introduction", "turn approved product captures into a timed narrative", "portfolio needs movement instead of static slides", "audiences need different scripts and reading rhythms", "short product teaser with reproducible sources"),
)
