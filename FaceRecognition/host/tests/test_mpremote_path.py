import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


TOOL = Path(__file__).resolve().parents[2] / "tools/mpremote_path.py"
SPEC = importlib.util.spec_from_file_location("tested_mpremote_path", TOOL)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


class MpremotePathTests(unittest.TestCase):
    def test_windows_dashboard_environment_is_discovered_without_path_activation(self):
        with tempfile.TemporaryDirectory(prefix="Nicla clone ") as temporary:
            app = Path(temporary) / "FaceRecognition"
            executable = app / ".venv-nicla-windows/Scripts/mpremote.exe"
            executable.parent.mkdir(parents=True)
            executable.write_bytes(b"fixture")
            with patch.object(resolver.shutil, "which", return_value=None):
                self.assertEqual(resolver.resolve_mpremote(app), executable)

    def test_explicit_path_install_keeps_its_precedence(self):
        with patch.object(resolver.shutil, "which", return_value="/tools/mpremote"):
            self.assertEqual(resolver.resolve_mpremote(Path("app")), Path("/tools/mpremote"))


if __name__ == "__main__":
    unittest.main()
