# Review remediation checklist

The user authorized implementation of the previous review findings. Preserve FFmpeg 9,
existing media/model artifacts, the separate pinned Irodori environment, and all
AGENTS.md audio/language/cloning contracts. Do not commit, push, or merge.

## Design and acceptance

Use stable native Apple Silicon packages: PyTorch 2.15.0, TorchAudio 2.11.1,
TorchCodec 0.17.0, and Demucs 4.1.0, subject to the actual resolver and native
smoke checks. TorchCodec 0.17 supports PyTorch >=2.11 and FFmpeg 4–9;
TorchAudio 2.11 supports PyTorch >=2.11. Never downgrade FFmpeg.
Sources: https://github.com/meta-pytorch/torchcodec#installing-torchcodec,
https://pypi.org/project/torchaudio/2.11.1/,
https://pypi.org/project/demucs/4.1.0/.

Isolate source/analysis caches by resolved input identity and active ASR/separation
settings, beneath the existing audio-mode/ASR/translation structure. Isolate all
TTS render caches and final names by engine, effective synthesis/mix settings,
translations and references as applicable. Preserve old outputs; do not migrate
unverifiable caches. Complete files are published atomically after validation.
TTS failures remain retryable and prevent publication of a completed video.

Overlapping utterances are sequenced in source-start order, preserving every
utterance. Partition the source video at overlap boundaries and extend a boundary
frame when an utterance has no remaining source interval; never replay an entire
overlapping interval. Use cumulative frame/sample boundaries for retiming and
verify the joined track lengths. The resulting dubbed turns are sequential.

## Progress Status

Current: Complete
Next: None

| Done | ID | Outcome, files and acceptance evidence | Depends on |
| :--- | :--- | :--- | :--- |
| [x] | I1 | PyTorch 2.15.0 / TorchAudio 2.11.1 / TorchCodec 0.17.0 / Demucs 4.1.0 installed; FFmpeg 9 retained; native decode, Demucs CLI and pyannote/OmniVoice/VoxCPM imports passed. MLX Audio stays 0.6.5 because >=0.6.6 conflicts with VoxCPM2's datasets/fsspec requirements. | None |
| [x] | I2 | Source/content/active-ASR namespaces isolate replaced/same-name inputs. Source-change extraction and folder-output exclusion regressions passed in the final 124-test suite. | None |
| [x] | I3 | All renders/final names distinguish source, engine, effective settings, references and translations; reference/settings/completion regressions passed. | I2 |
| [x] | I4 | Failed/None/corrupt required TTS stays retryable and prevents completed publication; validated siblings survive. Retry and interrupted-FLAC tests passed. | I3 |
| [x] | I5 | Extracted audio, separated stems, TTS and video outputs stage/validate before replace. Completion/concat manifests and retime-part identities prevent stale/corrupt reuse; interruption and real-media repair regressions passed. | I3 |
| [x] | I6 | Preserved contained/partial/simultaneous overlapping turns; all five retime coverage/monotonic-source tests passed (first execution). | None |
| [x] | I7 | FFmpeg 9 ten-chunk render produced exactly 200 video frames/320000 audio samples; production pipeline tests passed both raw/separated mixing and duration-changing cache repair. | I6 |
| [x] | I8 | Clipped OmniVoice references use ASR of the exact audio/language; eight reference regression tests passed (second execution, path normalization repair). | None |
| [x] | I9 | VoxCPM2 uses its own limits/prefixed references without reference ASR; covered by the eight reference tests. | I8 |
| [x] | I10 | Loopback default, authenticated external bind, consistent key forwarding and shell-safe mode0700 launcher; 13 Irodori tests passed (second execution). | None |
| [x] | I11 | Irodori wrapper finally releases only owned servers, including startup/generation failures; covered by 13 Irodori tests. | I4 |
| [x] | I12 | Balanced interval index/nearest endpoints preserve existing overlap/tie/fallback semantics; randomized nested/unsorted comparison and fallback tests passed. | None |
| [x] | I13 | Atomic input-bound translation records and validated per-utterance TTS records/files replace per-segment full snapshots; interruption and changed-source resume tests passed. | I4 |
| [x] | I14 | Irodori fixtures explicitly select their step count; scoped ty rule updated. All 124 tests, Ruff and ty passed without forcing the user's sampling setting to 4. | None |
| [x] | I15 | Native decoder/import/CLI smoke, real FFmpeg pipeline integration, 18-case dry-run, final 124-test suite and static checks passed; full-model inference limitation recorded below. | I1–I14 |
| [x] | I16 | Synchronized both READMEs with identity-based names/cache paths, per-utterance records, completion/retry behavior, Irodori authentication/launcher regeneration, and dependency pins. Reviewed CHANGELOG.md/NOTES.md/AGENTS.md/.env.example; source cross-check, 44 local file/heading references, and documentation whitespace checks passed. | I15 |

## Handoff checkpoint — 2026-11-21

The user requested a checkpoint before context compaction, then explicitly
resumed I16 in the next chat. Implementation and its required checks are complete
through I15. Do not restart dependency research, replay successful model/media
checks, or create a separate checklist. Continue I16 only unless a concrete
defect requires an implementation change.

I16 documentation scope (completed):

1. Update the existing "Resume and Outputs" sections in README.md and README_ja.md.
   Every final name now has a 16-character source/analysis/synthesis/mix identity
   between `<translation-namespace>` and the existing optional Irodori/design
   marker. Work paths add `input_<20-character-hash>/` after translation and before
   optional voice design. All TTS engines use `render/<fingerprint>/`; retiming
   adds `retime/<parts-and-cadence-fingerprint>/`. Completion requires the matching
   `.complete.json` manifest. Translation uses source-bound `.records/` entries;
   TTS writes per-utterance `tts_records/` and final aggregate metadata. Old outputs
   stay intact. Completed outputs are excluded from future folder scans.
2. Document localhost-only Irodori by default, authenticated external binds,
   owner-only generated launchers and regenerating a manual launcher with
   `uv run xlanguage-dubbing --generate-script` after configuration changes.
   Existing generated launchers and externally managed servers were not modified.
3. Document the FFmpeg 9 compatible pins and the deliberate MLX Audio 0.6.5 hold.
   Python runtime was 3.13.11; FFmpeg was 9.0.2. PyTorch 2.15.0, TorchAudio 2.11.1,
   TorchCodec 0.17.0, Demucs 4.1.0, NumPy 2.5.5, pyannote.audio 4.1.1,
   Accelerate 1.16.0 and Ruff 0.17.6 are installed. Irodori's separate pinned
   environment remains unchanged. The active ignored `.env` changed only its
   unauthenticated wildcard host to `127.0.0.1`; its sampling settings were retained.
4. Review documentation diffs for consistency and unresolved references, run
   `git diff --check`, mark I16 complete and give the final concise handoff.
   Documentation-only edits do not require rerunning implementation tests.

I16 completion evidence — 2026-11-21: Both READMEs were synchronized and
independently checked against the implementation. Local file/heading references
passed (44); `git diff --check` and the untracked plan whitespace review passed.
NOTES.md no longer describes README synchronization as pending. CHANGELOG.md,
AGENTS.md, and .env.example already covered the verified behavior and needed no
additional changes. Only documentation changed during I16; prior implementation
evidence remains valid. No required work remains.

Current verification evidence (do not reset bounded attempts):

- Dependency sync succeeded on execution 3 after resolving the known MLX Audio
  conflict and limiting uv resolution to the documented macOS arm64 platform.
  Log: `/private/tmp/xlanguage-review-sync.log`.
- Native TorchCodec decoded a fresh synthetic 16000-sample WAV. Demucs CLI help
  and Demucs/pyannote/OmniVoice/VoxCPM imports passed. No full-model inference or
  model downloads were required for this smoke check.
- Final full regression suite passed **124 tests in 9.305s** on execution 2.
  Execution 1 exposed only old NeMo fixtures expecting the pre-identity directory
  and empty fake WAVs; fixtures were updated to the new path and valid WAVs.
  Log: `/private/tmp/xlanguage-review-full-tests.log`.
- Final Ruff and ty checks both passed on execution 3. Earlier diagnostics were
  repaired imports, the deliberate adjacent zip's `strict=False`, reference dict
  type narrowing, and a test baseline's callable type. No broad suppression.
- The 18-case matrix dry-run passed on execution 1. Synthetic FFmpeg/retime tests
  and raw/separated pipeline tests passed, including repair with changed individual
  TTS durations but unchanged total length. ASR/translation/TTS/separation outputs
  were deterministic fixtures in integration tests; real model quality/listening
  and full-video inference have not been revalidated.
- `git diff --check` passed before this tracker update. No commits, pushes,
  merges, existing-media deletion, or external-server shutdown were performed.

## Verification limits

Use terminal/API/deterministic tests only; do not use CUA or interactive browser
testing. Each verification unit permits one initial run and two focused repair
reruns (three total); retain failures/attempt counts if interrupted. Stop testing
when the required units pass. Do not run full-length model inference repeatedly.

- Dependency resolve/sync: `UV_CACHE_DIR=/private/tmp/xlanguage-review-uv uv sync
  --inexact`; 15-minute limit; project environment only, no Irodori sync.
- Native smoke: project Python, synthetic WAV, TorchCodec `get_all_samples()`,
  package imports and installed Demucs CLI help; 120-second limit. No model download.
- Focused regression units: `uv run --no-sync python -m unittest discover -s tests
  -p '<affected-pattern>'`; 120 seconds per unit. Synthetic media only.
- Final static checks: `uv run --no-sync ruff check src tests scripts` and
  `uv run --no-sync ty check`; 120 seconds each.
- Final regression suite: `uv run --no-sync python -m unittest discover -s tests`;
  180-second limit; retain user sampling settings, no forced default step count.
- FFmpeg timeline smoke: synthetic 30fps source and ten speech/gap chunks;
  120-second limit; video error <= one frame, joined audio exact target samples,
  final MP4 fully decodes. Verify both raw/separated mixing arguments in tests.
- Matrix: `uv run --no-sync python scripts/run_config_matrix.py --dry-run`;
  60-second limit; all supported 18 combinations represented. Confirm exact CLI
  arguments from the installed script before execution.
- Documentation: `git diff --check` and focused diff/path review; 60-second limit.

Unavailable native full-model generation is reported as a limitation, not a
passed smoke test. After automated completion recommend one short user video
with overlapping speakers and a resumed run as optional listening/operation
checks; these are not implementation completion gates.
