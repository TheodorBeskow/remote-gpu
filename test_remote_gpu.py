import io
import json
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import click
from click.testing import CliRunner

from remote_gpu.cli import main
from remote_gpu.config import Config, KaggleConfig, Paths, find_config, load_config
from remote_gpu.kaggle_manager import KaggleRunner, _terminal_status
from remote_gpu.kaggleify import build_preamble


class RemoteGpuTests(unittest.TestCase):
    def test_cli_help_describes_config_and_status(self):
        runner = CliRunner()
        run_help = runner.invoke(main, ["run", "--help"])
        self.assertEqual(run_help.exit_code, 0)
        help_text = " ".join(run_help.output.split())
        self.assertIn("remote-gpu-settings.yaml", help_text)
        self.assertIn("or settings.yaml", help_text)
        self.assertIn("script's directory", help_text)
        self.assertIn("parent directories", help_text)
        self.assertIn("local_input", help_text)
        self.assertIn("local_output", help_text)
        self.assertIn("Push and return immediately - use", help_text)
        self.assertIn("--internet / --no-internet", help_text)
        self.assertIn("--dry-run", help_text)
        self.assertNotIn("\ufffd", help_text)

        status_help = runner.invoke(main, ["status", "--help"])
        self.assertEqual(status_help.exit_code, 0)
        self.assertIn("authentication status", status_help.output)
        self.assertNotIn("quota info", status_help.output)

        logs_help = runner.invoke(main, ["logs", "--help"])
        self.assertEqual(logs_help.exit_code, 0)
        self.assertIn("may not expose live cell output", logs_help.output)

    def test_settings_yaml_alias_and_precedence(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            alias = root / "settings.yaml"
            alias.write_text("name: alias\nkaggle:\n  internet_enabled: true\n", encoding="utf-8")
            self.assertEqual(find_config(nested), alias)
            config = load_config(nested)
            self.assertEqual(config.project_dir, root)
            self.assertEqual(config.name, "alias")
            self.assertTrue(config.kaggle.internet_enabled)

            canonical = root / "remote-gpu-settings.yaml"
            canonical.write_text("name: canonical\n", encoding="utf-8")
            self.assertEqual(find_config(nested), canonical)
            self.assertEqual(load_config(nested).name, "canonical")

            nearer = nested / "settings.yaml"
            nearer.write_text("name: nearer\n", encoding="utf-8")
            self.assertEqual(find_config(nested), nearer)
            self.assertEqual(load_config(nested).name, "nearer")

    def test_deprecated_quota_warning_is_not_treated_as_active(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            (root / "settings.yaml").write_text(
                "runtime:\n  quota_warning_hours: 5\n", encoding="utf-8"
            )
            warnings = io.StringIO()
            with redirect_stderr(warnings):
                config = load_config(root)
            self.assertFalse(hasattr(config.runtime, "quota_warning_hours"))
            self.assertIn("quota_warning_hours", warnings.getvalue())
            self.assertIn("no effect", warnings.getvalue())

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
                        self.assertIn("'iris': 'uciml/iris'", code)
                        return mock.Mock(returncode=0)

                    upload = root / "upload"
                    upload.mkdir(exist_ok=True)
                    with mock.patch("remote_gpu.kaggle_manager.tempfile.TemporaryDirectory", return_value=nullcontext(upload)), mock.patch(
                        "remote_gpu.kaggle_manager._kaggle", side_effect=push
                    ):
                        runner._push_kernel(entry)

    def test_internet_cli_override_does_not_change_yaml(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            entry = root / "solve.py"
            entry.write_text("print('hello')", encoding="utf-8")
            settings = root / "settings.yaml"
            for configured, flag, expected in (
                (False, "--internet", True),
                (True, "--no-internet", False),
                (True, None, True),
            ):
                with self.subTest(flag=flag):
                    settings.write_text(
                        f"kaggle:\n  user: tester\n  internet_enabled: {str(configured).lower()}\n",
                        encoding="utf-8",
                    )
                    args = ["run", str(entry), "--detach"]
                    if flag:
                        args.append(flag)
                    with mock.patch.object(KaggleRunner, "launch", autospec=True) as launch:
                        result = CliRunner().invoke(main, args)
                        self.assertEqual(result.exit_code, 0, result.output)
                        self.assertEqual(launch.call_args.args[0].config.kaggle.internet_enabled, expected)
                    self.assertEqual(load_config(root).kaggle.internet_enabled, configured)

    def test_dry_run_previews_metadata_without_side_effects(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            entry = root / "solve.py"
            entry.write_text("print('hello')", encoding="utf-8")
            (root / "settings.yaml").write_text(
                "kaggle:\n  user: tester\n  internet_enabled: true\n"
                "datasets:\n  iris: uciml/iris\n",
                encoding="utf-8",
            )
            with mock.patch("remote_gpu.kaggle_manager._username", return_value="tester"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=AssertionError
            ), mock.patch.object(KaggleRunner, "_save_state", side_effect=AssertionError):
                preview = CliRunner().invoke(main, ["run", str(entry), "--dry-run", "--cpu", "--no-internet"])
                self.assertEqual(preview.exit_code, 0, preview.output)
                metadata = json.loads(preview.output)
                self.assertEqual(metadata["id"], f"tester/remote-gpu-{root.name}")
                self.assertFalse(metadata["enable_gpu"])
                self.assertFalse(metadata["enable_internet"])
                self.assertEqual(metadata["code_file"], "solve.py")
                self.assertEqual(metadata["dataset_sources"], ["uciml/iris"])
                self.assertFalse((root / ".remote-gpu").exists())

                (root / "input").mkdir()
                preview = CliRunner().invoke(main, ["run", str(entry), "--dry-run", "--internet", "--detach"])
                self.assertEqual(preview.exit_code, 0, preview.output)
                metadata = json.loads(preview.output)
                self.assertTrue(metadata["enable_internet"])
                self.assertEqual(metadata["dataset_sources"], ["uciml/iris", f"tester/remote-gpu-{root.name}-data"])
                self.assertFalse((root / ".remote-gpu").exists())

    def test_failed_run_attempts_recovery_but_stays_failed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            entry = root / "solve.py"
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            def kaggle(*args):
                if args[1] == "logs":
                    return mock.Mock(returncode=0, stdout="[]")
                return mock.Mock(returncode=0, stdout="status: error")

            with mock.patch.object(runner, "launch"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=kaggle
            ), mock.patch.object(runner, "_fetch_output") as fetch:
                with self.assertRaisesRegex(Exception, "kernel.*error"):
                    runner.run(entry)
                fetch.assert_called_once_with(root)

            with mock.patch.object(runner, "launch"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=kaggle
            ), mock.patch.object(runner, "_fetch_output", side_effect=click.ClickException("Kaggle output unavailable")) as fetch:
                with self.assertRaisesRegex(Exception, "Kaggle output unavailable"):
                    runner.run(entry)
                fetch.assert_called_once_with(root)

            with mock.patch.object(runner, "launch"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=kaggle
            ), mock.patch.object(runner, "_fetch_output", return_value=False):
                with self.assertRaisesRegex(click.ClickException, "kernel error; Kaggle returned no output files"):
                    runner.run(entry)

            with mock.patch.object(runner, "launch"), mock.patch.object(
                runner, "_wait", return_value="complete"
            ), mock.patch.object(runner, "_fetch_output", return_value=False):
                runner.run(entry)

    def test_failed_run_recovers_checkpoint_file(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            download = root / "download"
            (download / "output").mkdir(parents=True)
            (download / "output" / "checkpoint.pt").write_bytes(b"checkpoint")
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))

            def kaggle(*args):
                if args[1] == "logs":
                    return mock.Mock(returncode=0, stdout="[]")
                return mock.Mock(returncode=0, stdout="status: timeout")

            with mock.patch.object(runner, "launch"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=kaggle
            ), mock.patch("remote_gpu.kaggle_manager.tempfile.TemporaryDirectory", return_value=nullcontext(download)):
                with self.assertRaisesRegex(click.ClickException, "kernel timeout"):
                    runner.run(root / "solve.py")
            self.assertEqual((root / "output" / "checkpoint.pt").read_bytes(), b"checkpoint")

    def test_terminal_status_does_not_match_incomplete(self):
        self.assertIsNone(_terminal_status("status: incomplete"))
        self.assertEqual(_terminal_status("status: timed out"), "timed out")

    def test_pull_attempts_failed_output_and_reports_unavailability(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            with mock.patch.object(runner, "_load_state", return_value={"entry_dir": str(root)}), mock.patch(
                "remote_gpu.kaggle_manager._kaggle",
                return_value=mock.Mock(returncode=0, stdout="status: cancelled"),
            ), mock.patch.object(runner, "_fetch_output") as fetch:
                runner.pull()
                fetch.assert_called_once_with(root)
            with mock.patch.object(runner, "_load_state", return_value={"entry_dir": str(root)}), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(returncode=0, stdout="status: cancelled")
            ), mock.patch.object(runner, "_fetch_output", return_value=False):
                with self.assertRaisesRegex(click.ClickException, "Kaggle returned no output files"):
                    runner.pull()

            with mock.patch("remote_gpu.kaggle_manager.tempfile.TemporaryDirectory", return_value=nullcontext(root)), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
                    returncode=1, stderr="No output files available"
                )
            ):
                with self.assertRaisesRegex(Exception, "No output files available"):
                    runner._fetch_output(root)
            self.assertFalse((root / "output").exists())
            empty = root / "empty"
            empty.mkdir()
            with mock.patch("remote_gpu.kaggle_manager.tempfile.TemporaryDirectory", return_value=nullcontext(empty)), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(returncode=0)
            ):
                self.assertFalse(runner._fetch_output(root))

    def test_logs_explain_empty_running_output_once(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            statuses = iter(("status: running", "status: complete"))

            def kaggle(*args):
                if args[1] == "logs":
                    return mock.Mock(returncode=0, stdout="[]")
                return mock.Mock(returncode=0, stdout=next(statuses))

            notice = io.StringIO()
            with mock.patch("remote_gpu.kaggle_manager._kaggle", side_effect=kaggle) as calls, mock.patch(
                "remote_gpu.kaggle_manager.time.sleep"
            ), redirect_stderr(notice):
                runner.stream_logs(show_all=True, follow=True, save=str(root / "run.log"))
            self.assertEqual(notice.getvalue().count("No logs available"), 1)
            self.assertIn("live cell output", notice.getvalue())
            self.assertIn("Kaggle UI", notice.getvalue())
            self.assertEqual((root / "run.log").read_text(), "")
            self.assertEqual(sum(call.args[1] == "logs" for call in calls.call_args_list), 2)

    def test_logs_distinguish_missing_output_from_api_error(self):
        runner = KaggleRunner(Config(project_dir=Path.cwd(), name="run", kaggle=KaggleConfig(user="tester")))
        notice = io.StringIO()
        with mock.patch("remote_gpu.kaggle_manager._kaggle", side_effect=(
            mock.Mock(returncode=0, stdout="[]"),
            mock.Mock(returncode=0, stdout="status: complete"),
        )), redirect_stderr(notice):
            runner.stream_logs(show_all=True, follow=False, save=None)
        self.assertIn("No logs available", notice.getvalue())

        with mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
            returncode=1, stderr="Not authorized", stdout=""
        )):
            with self.assertRaisesRegex(click.ClickException, "Not authorized"):
                runner.stream_logs(show_all=True, follow=False, save=None)
            self.assertEqual(runner._fetch_log_entries(), [])

        notice = io.StringIO()
        printed = io.StringIO()
        with mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
            returncode=0, stdout='[{"data": "progress\\n"}]'
        )), redirect_stderr(notice), redirect_stdout(printed):
            runner.stream_logs(show_all=True, follow=False, save=None)
        self.assertEqual(printed.getvalue(), "progress\n")
        self.assertEqual(notice.getvalue(), "")

    def test_runtime_messages_use_ascii(self):
        runner = KaggleRunner(Config(project_dir=Path.cwd(), name="run", kaggle=KaggleConfig(user="tester")))
        with mock.patch.object(runner, "_print_new_logs", side_effect=KeyboardInterrupt), mock.patch(
            "remote_gpu.kaggle_manager.click.echo"
        ) as echo:
            self.assertFalse(runner._wait())
            self.assertTrue(all(str(call.args[0]).isascii() for call in echo.call_args_list))
        with mock.patch("remote_gpu.kaggle_manager.click.echo") as echo:
            runner._sync_dataset(Path.cwd() / "missing" / "solve.py")
            self.assertTrue(all(str(call.args[0]).isascii() for call in echo.call_args_list))

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
        self.assertIn(
            "'assets/input': 'tester/remote-gpu-data'",
            build_preamble(config, managed_input=True, managed_dataset="tester/remote-gpu-data"),
        )
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", "/kaggle/input/custom")), mock.patch(
            "glob.glob", side_effect=lambda pattern, **kwargs: [f"/kaggle/input/custom/{pattern.rsplit('/', 1)[-1]}"]
        ) as glob, mock.patch("os.path.isdir", return_value=True), mock.patch("os.makedirs"), mock.patch(
            "os.symlink"
        ) as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
            "os.path.lexists", return_value=False
        ), mock.patch.dict(
            "sys.modules", {"torch": None}
        ):
            exec(preamble, {})
        self.assertTrue(all(call.args[0].startswith("/kaggle/input/custom/") for call in glob.call_args_list))
        symlink.assert_any_call("/kaggle/working/custom/results/output", "results/output")

    def test_dataset_mount_prefers_full_owner_and_rejects_ambiguity(self):
        input_root = "/kaggle/input"
        hits = [f"{input_root}/datasets/{owner}/shared" for owner in ("alice", "bob")]
        config = Config(
            project_dir=Path.cwd(), name="run",
            datasets={"alice_data": "alice/shared", "bob_data": "bob/shared"},
        )
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", input_root)), mock.patch(
            "glob.glob", return_value=hits
        ), mock.patch("os.path.isdir", return_value=True), mock.patch("os.makedirs"), mock.patch(
            "os.symlink"
        ) as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
            "os.path.lexists", return_value=False
        ), mock.patch.dict("sys.modules", {"torch": None}):
            exec(build_preamble(config), {})
        symlink.assert_any_call(hits[0], "alice_data")
        symlink.assert_any_call(hits[1], "bob_data")

        config.datasets = {"missing": "charlie/shared"}
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", input_root)), mock.patch(
            "glob.glob", return_value=hits
        ), mock.patch("os.path.isdir", return_value=True), mock.patch.dict(
            "sys.modules", {"torch": None}
        ):
            with self.assertRaisesRegex(RuntimeError, "ambiguous.*charlie/shared"):
                exec(build_preamble(config), {})

        config.datasets = {"flat": "alice/shared"}
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", input_root)), mock.patch(
            "glob.glob", return_value=[f"{input_root}/shared"]
        ), mock.patch("os.path.isdir", return_value=True), mock.patch("os.makedirs"), mock.patch(
            "os.symlink"
        ) as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
            "os.path.lexists", return_value=False
        ), mock.patch.dict("sys.modules", {"torch": None}):
            exec(build_preamble(config), {})
        symlink.assert_any_call(f"{input_root}/shared", "flat")

        config.datasets = {"alice_data": "alice/shared", "bob_data": "bob/shared"}
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", input_root)), mock.patch(
            "glob.glob", return_value=[f"{input_root}/shared"]
        ), mock.patch("os.path.isdir", return_value=True), mock.patch.dict(
            "sys.modules", {"torch": None}
        ):
            with self.assertRaisesRegex(RuntimeError, "ambiguous.*shared"):
                exec(build_preamble(config), {})

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
