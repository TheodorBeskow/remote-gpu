import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

from click.testing import CliRunner

from remote_gpu.cli import main
from remote_gpu.config import Config, KaggleConfig, Paths, load_config
from remote_gpu.kaggle_manager import KaggleRunner
from remote_gpu.kaggleify import build_preamble


class RemoteGpuTests(unittest.TestCase):
    def test_cli_help_describes_config_and_status(self):
        runner = CliRunner()
        run_help = runner.invoke(main, ["run", "--help"])
        self.assertEqual(run_help.exit_code, 0)
        help_text = " ".join(run_help.output.split())
        self.assertIn("remote-gpu-settings.yaml", help_text)
        self.assertIn("script's directory", help_text)
        self.assertIn("parent directories", help_text)
        self.assertIn("local_input", help_text)
        self.assertIn("local_output", help_text)
        self.assertIn("Push and return immediately - use", help_text)
        self.assertNotIn("\ufffd", help_text)

        status_help = runner.invoke(main, ["status", "--help"])
        self.assertEqual(status_help.exit_code, 0)
        self.assertIn("authentication status", status_help.output)
        self.assertNotIn("quota info", status_help.output)

    def test_internet_setting_and_kernel_without_input(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            entry = root / "solve.py"
            entry.write_text("print('hello')", encoding="utf-8")
            config_path = root / "remote-gpu-settings.yaml"
            for enabled in (False, True):
                with self.subTest(enabled=enabled):
                    config_path.write_text(
                        f"kaggle:\n  user: tester\n  internet_enabled: {str(enabled).lower()}\n"
                        "datasets:\n  iris: uciml/iris\n",
                        encoding="utf-8",
                    )
                    config = load_config(root)
                    runner = KaggleRunner(config)
                    with mock.patch.object(runner, "_dataset_exists", side_effect=AssertionError):
                        runner._sync_dataset(entry)
                    self.assertFalse(runner._managed_exists)

                    def push(*args):
                        self.assertEqual(args[:2], ("kernels", "push"))
                        pushed = Path(args[-1])
                        metadata = json.loads((pushed / "kernel-metadata.json").read_text())
                        self.assertEqual(metadata["enable_internet"], enabled)
                        self.assertEqual(metadata["dataset_sources"], ["uciml/iris"])
                        code = (pushed / "solve.py").read_text()
                        self.assertNotIn("remote-gpu-data", code)
                        self.assertIn("'iris': 'iris'", code)
                        return mock.Mock(returncode=0)

                    upload = root / "upload"
                    upload.mkdir(exist_ok=True)
                    with mock.patch("remote_gpu.kaggle_manager.tempfile.TemporaryDirectory", return_value=nullcontext(upload)), mock.patch(
                        "remote_gpu.kaggle_manager._kaggle", side_effect=push
                    ):
                        runner._push_kernel(entry)

    def test_custom_paths_and_managed_mount(self):
        config = Config(
            project_dir=Path.cwd(),
            name="run",
            paths=Paths(
                local_input="./assets/input",
                local_output="./results/output",
                kaggle_input="/kaggle/input/custom",
                kaggle_output="/kaggle/working/custom",
            ),
            datasets={"iris": "uciml/iris"},
        )
        preamble = build_preamble(config, managed_input=True)
        self.assertIn("'assets/input': 'remote-gpu-data'", preamble)
        self.assertIn("_input = '/kaggle/input/custom'", preamble)
        self.assertIn("_output = '/kaggle/working/custom/results/output'", preamble)
        self.assertIn("_local_output = 'results/output'", preamble)
        self.assertIn("os.symlink(_output, _local_output)", preamble)
        self.assertNotIn("'assets/input': 'remote-gpu-data'", build_preamble(config))
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", "/kaggle/input/custom")), mock.patch(
            "glob.glob", side_effect=lambda pattern, **kwargs: [f"/kaggle/input/custom/{pattern.rsplit('/', 1)[-1]}"]
        ) as glob, mock.patch("os.makedirs"), mock.patch("os.symlink") as symlink, mock.patch(
            "os.listdir", return_value=[]
        ), mock.patch("os.path.lexists", return_value=False), mock.patch.dict(
            "sys.modules", {"torch": None}
        ):
            exec(preamble, {})
        self.assertTrue(all(call.args[0].startswith("/kaggle/input/custom/") for call in glob.call_args_list))
        symlink.assert_any_call("/kaggle/working/custom/results/output", "results/output")

    def test_fetch_custom_output_root(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            config = Config(
                project_dir=root,
                name="run",
                kaggle=KaggleConfig(user="tester"),
                paths=Paths(kaggle_output="/kaggle/working/custom"),
            )
            runner = KaggleRunner(config)
            download = root / "download"
            (download / "custom" / "output").mkdir(parents=True)
            (download / "custom" / "output" / "result.txt").write_text("done")
            (download / "log.txt").write_text("log")

            with mock.patch("remote_gpu.kaggle_manager.tempfile.TemporaryDirectory", return_value=nullcontext(download)), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(returncode=0)
            ):
                runner._fetch_output(root)
            self.assertEqual((root / "output" / "result.txt").read_text(), "done")
            self.assertEqual((root / "output" / "log.txt").read_text(), "log")
            self.assertFalse((root / "output" / "custom").exists())


if __name__ == "__main__":
    unittest.main()
