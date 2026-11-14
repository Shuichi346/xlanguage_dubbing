from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path

SETUP = Path(__file__).resolve().parents[1] / "scripts" / "setup_irodori.sh"
FORK = "https://github.com/Shuichi346/Irodori-TTS-Server-nonWatermark.git"


class IrodoriSetupTests(unittest.TestCase):
    def _run_setup(self, origin: str, *, outdated: bool = False, dirty: bool = False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            server = root / "server with spaces"
            server.mkdir()
            manifest = server / "pyproject.toml"
            manifest.write_text("preserve local configuration\n")
            tools = root / "bin"
            tools.mkdir()
            git = tools / "git"
            git.write_text('''#!/bin/sh
printf '%s\\n' "$*" >> "$GIT_LOG"
case "$*" in
    *"remote get-url origin") printf '%s\\n' "$TEST_ORIGIN" ;;
    *"rev-parse HEAD") printf '%s\\n' "$TEST_HEAD" ;;
    *"status --porcelain --untracked-files=no")
        if [ "$TEST_DIRTY" = 1 ]; then printf ' M pyproject.toml\\n'; fi ;;
    *) exit 97 ;;
esac
''')
            uv = tools / "uv"
            uv.write_text('''#!/bin/sh
if [ "${VIRTUAL_ENV+x}" = x ]; then exit 96; fi
printf '%s\\0' "$@" > "$UV_LOG"
''')
            git.chmod(0o755)
            uv.chmod(0o755)
            revision_match = re.search(r'^server_revision="([0-9a-f]+)"', SETUP.read_text(), re.M)
            self.assertIsNotNone(revision_match)
            assert revision_match is not None
            environment = {
                **os.environ,
                "PATH": f"{tools}:/usr/bin:/bin",
                "VIRTUAL_ENV": "/unrelated/project/.venv",
                "GIT_LOG": str(root / "git.log"),
                "UV_LOG": str(root / "uv.log"),
                "TEST_ORIGIN": origin,
                "TEST_HEAD": "old-revision" if outdated else revision_match[1],
                "TEST_DIRTY": "1" if dirty else "0",
            }
            result = subprocess.run(
                ["/bin/bash", str(SETUP), str(server)],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            uv_log = root / "uv.log"
            arguments = uv_log.read_bytes().decode().split("\0")[:-1] if uv_log.exists() else []
            self.assertEqual(manifest.read_text(), "preserve local configuration\n")
            return result, arguments, (root / "git.log").read_text(), str(server)

    def test_matching_fork_uses_locked_sync_in_its_own_environment(self):
        result, arguments, git_log, server = self._run_setup(FORK)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(arguments, ["sync", "--project", server, "--locked", "--extra", "cpu", "--inexact"])
        self.assertNotIn("checkout", git_log)

    def test_original_checkout_is_preserved_and_rejected(self):
        result, arguments, git_log, _ = self._run_setup("https://github.com/Aratako/Irodori-TTS-Server.git")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Choose a new server directory", result.stderr)
        self.assertEqual(arguments, [])
        self.assertNotIn("checkout", git_log)

    def test_outdated_dirty_fork_is_preserved_before_fetch_or_checkout(self):
        result, arguments, git_log, _ = self._run_setup(FORK, outdated=True, dirty=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("local changes", result.stderr)
        self.assertEqual(arguments, [])
        self.assertNotIn("fetch", git_log)
        self.assertNotIn("checkout", git_log)


if __name__ == "__main__":
    unittest.main()
