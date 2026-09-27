# ExamGuard Greenfield Rebuild Pack

Copy these files into the repository root.

Read order:
1. AGENTS.md
2. PROJECT_CONTEXT.md
3. FUNCTIONAL_SPEC.md
4. DATABASE_SCHEMA.md
5. SYSTEM_ARCHITECTURE.md
6. UI_UX_SPEC.md
7. API_CONTRACT.md
8. CONFIG_SPEC.md
9. IMPLEMENTATION_PLAN.md
10. CODEX_REBUILD_PROMPT.md

Then give Codex:
```text
Read AGENTS.md and every document referenced by CODEX_REBUILD_PROMPT.md.
Follow CODEX_REBUILD_PROMPT.md exactly.
Treat the existing application/database as legacy.
Preserve only working infrastructure where useful.
Do not code until you first return the requested repository assessment.
```

Core business model:
```text
Room → Seat → ExamSession → Candidate Assignment → Monitoring → Event → Review → Evidence → Appeal
```

Current CV milestone:
```text
MP4 → native browser playback → YOLO11n → ByteTrack → WebSocket → Canvas
```

TSM/action recognition is a later integration phase.
