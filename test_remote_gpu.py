import importlib.util
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import click
from click.testing import CliRunner

from remote_gpu.cli import main
from remote_gpu.config import Config, KaggleConfig, Paths, find_config, load_config
from remote_gpu.kaggle_manager import KaggleRunner, _kaggle, _terminal_status
from remote_gpu.kaggleify import build_preamble, kaggleify_notebook


class RemoteGpuTests(unittest.TestCase):
    def test_cli_help_describes_config_and_status(self):
        runner = CliRunner()
        run_help = runner.invoke(main, ["run", "--help"])
        self.assertEqual(run_help.exit_code, 0)
        help_text = " ".join(run_help.output.split())
        self.assertIn("remote-gpu-settings.yaml", help_text)
        self.assertIn("or settings.yaml", help_text)
        self.assertIn("both exist in the same directory", help_text)
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
        self.assertIn("credentials and API access", status_help.output)
        self.assertNotIn("quota info", status_help.output)

        logs_help = runner.invoke(main, ["logs", "--help"])
        self.assertEqual(logs_help.exit_code, 0)
        self.assertIn("may not expose live cell output", logs_help.output)

    def test_status_verifies_kaggle_api_access(self):
        runner = CliRunner()
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            with mock.patch.dict(os.environ, {"KAGGLE_USERNAME": "tester"}, clear=True), mock.patch(
                "remote_gpu.cli.Path.home", return_value=root
            ), mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(returncode=0, stdout="ref\n")) as kaggle:
                result = runner.invoke(main, ["status"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(kaggle.call_args.args, ("datasets", "list", "--mine", "-p", "1", "--format", "csv"))
            self.assertIn("authenticated: tester", result.output)
            self.assertIn("API access: ok", result.output)

            with mock.patch.dict(os.environ, {}, clear=True), mock.patch(
                "remote_gpu.cli.Path.home", return_value=root
            ), mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
                returncode=1, stderr="403 authentication required", stdout=""
            )):
                result = runner.invoke(main, ["status"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("403 authentication required", result.output)
            self.assertNotIn("authenticated:", result.output)

    def test_clean_removes_only_retained_downloads(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            root = Path.cwd()
            (root / "settings.yaml").write_text("name: clean-test\n", encoding="utf-8")
            downloads = root / ".remote-gpu" / "downloads"
            (downloads / "old").mkdir(parents=True)
            (downloads / "old" / "file.pt").write_bytes(b"old")
            state = root / ".remote-gpu" / "state.json"
            state.write_text("{}", encoding="utf-8")

            declined = runner.invoke(main, ["clean"], input="n\n")
            self.assertEqual(declined.exit_code, 1)
            self.assertTrue((downloads / "old" / "file.pt").exists())

            result = runner.invoke(main, ["clean", "--yes"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertFalse(downloads.exists())
            self.assertTrue(state.exists())

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
            with self.assertRaisesRegex(click.UsageError, "both.*settings.yaml"):
                find_config(nested)
            entry = root / "solve.py"
            entry.write_text("print('hello')", encoding="utf-8")
            result = CliRunner().invoke(main, ["run", str(entry), "--dry-run"])
            self.assertEqual(result.exit_code, 2)
            self.assertIn("keep only one", result.output)

            nearer = nested / "settings.yaml"
            nearer.write_text("name: nearer\n", encoding="utf-8")
            self.assertEqual(find_config(nested), nearer)
            self.assertEqual(load_config(nested).name, "nearer")

            alias.unlink()
            self.assertEqual(find_config(root), canonical)
            self.assertEqual(load_config(root).name, "canonical")

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
                        self.assertIn("https://github.com/TheodorBeskow/remote-gpu", code)
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

    def test_kaggle_output_subprocess_handles_unicode_on_windows(self):
        real_run = subprocess.run

        def simulate_kaggle(command, **kwargs):
            self.assertEqual(command[:3], ["kaggle", "kernels", "output"])
            return real_run([sys.executable, "-B", "-c", "print('\\u4f60')"], **kwargs)

        with mock.patch.dict(os.environ, {"PYTHONIOENCODING": "cp1252"}), mock.patch(
            "remote_gpu.kaggle_manager.subprocess.run", side_effect=simulate_kaggle
        ):
            result = _kaggle("kernels", "output", "tester/run", "-p", "output")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "\u4f60\n")

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

            process = mock.Mock(stdout=io.BytesIO())
            process.wait.return_value = 0
            with mock.patch.object(runner, "launch"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=kaggle
            ), mock.patch.object(runner, "_output_stage", return_value=download), mock.patch(
                "remote_gpu.kaggle_manager.subprocess.Popen", return_value=process
            ):
                with self.assertRaisesRegex(click.ClickException, "kernel timeout"):
                    runner.run(root / "solve.py")
            self.assertEqual((root / "output" / "checkpoint.pt").read_bytes(), b"checkpoint")

    def test_terminal_status_does_not_match_incomplete(self):
        self.assertIsNone(_terminal_status("status: incomplete"))
        self.assertIsNone(_terminal_status("KernelWorkerStatus.CANCEL_REQUESTED"))
        self.assertEqual(_terminal_status("KernelWorkerStatus.CANCEL_ACKNOWLEDGED"), "cancel_acknowledged")
        self.assertEqual(_terminal_status("status: timed out"), "timed out")

    def test_acknowledged_cancellation_stops_polling_and_attempts_recovery(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            statuses = iter(("KernelWorkerStatus.CANCEL_REQUESTED", "KernelWorkerStatus.CANCEL_ACKNOWLEDGED"))

            def kaggle(*args):
                if args[1] == "logs":
                    return mock.Mock(returncode=0, stdout="[]")
                return mock.Mock(returncode=0, stdout=next(statuses))

            with mock.patch.object(runner, "launch"), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=kaggle
            ) as calls, mock.patch("remote_gpu.kaggle_manager.time.sleep"), mock.patch.object(
                runner, "_fetch_output", return_value=False
            ) as fetch:
                with self.assertRaisesRegex(click.ClickException, "kernel cancel_acknowledged; Kaggle returned no output files"):
                    runner.run(root / "solve.py")
            fetch.assert_called_once_with(root)
            self.assertEqual(sum(call.args[1] == "status" for call in calls.call_args_list), 2)
            with mock.patch.object(runner, "launch"), mock.patch.object(
                runner, "_wait", return_value="cancel_acknowledged"
            ), mock.patch.object(runner, "_fetch_output", return_value=True):
                with self.assertRaisesRegex(click.ClickException, "verify downloaded output"):
                    runner.run(root / "solve.py")

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

            with mock.patch.object(runner, "_load_state", return_value={"entry_dir": str(root)}), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
                    returncode=0, stdout="KernelWorkerStatus.CANCEL_ACKNOWLEDGED"
                )
            ), mock.patch.object(runner, "_fetch_output", return_value=True) as fetch:
                notice = io.StringIO()
                with redirect_stderr(notice):
                    runner.pull()
                fetch.assert_called_once_with(root)
                self.assertIn("verify downloaded output", notice.getvalue())
            with mock.patch.object(runner, "_load_state", return_value={"entry_dir": str(root)}), mock.patch(
                "remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
                    returncode=0, stdout="KernelWorkerStatus.CANCEL_ACKNOWLEDGED"
                )
            ), mock.patch.object(runner, "_fetch_output", return_value=False):
                with self.assertRaisesRegex(click.ClickException, "Kaggle returned no output files"):
                    runner.pull()

            failed = mock.Mock(stdout=io.BytesIO(b"No output files available\n"))
            failed.wait.return_value = 1
            with mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", return_value=failed):
                with self.assertRaisesRegex(Exception, "No output files available"):
                    runner._fetch_output(root)
            self.assertFalse((root / "output").exists())
            empty = root / "empty"
            empty.mkdir()
            successful = mock.Mock(stdout=io.BytesIO())
            successful.wait.return_value = 0
            with mock.patch.object(runner, "_output_stage", return_value=empty), mock.patch(
                "remote_gpu.kaggle_manager.subprocess.Popen", return_value=successful
            ):
                self.assertFalse(runner._fetch_output(root))

    def test_run_explains_missing_live_logs_once(self):
        runner = KaggleRunner(Config(project_dir=Path.cwd(), name="run", kaggle=KaggleConfig(user="tester")))
        statuses = iter(("status: running", "status: running", "status: complete"))

        def kaggle(*args):
            if args[1] == "logs":
                return mock.Mock(returncode=0, stdout="[]")
            return mock.Mock(returncode=0, stdout=next(statuses))

        with mock.patch("remote_gpu.kaggle_manager._kaggle", side_effect=kaggle), mock.patch(
            "remote_gpu.kaggle_manager.time.sleep"
        ), mock.patch("remote_gpu.kaggle_manager.click.echo") as echo:
            self.assertEqual(runner._wait(), "complete")
        notices = [call.args[0] for call in echo.call_args_list if "No logs available" in call.args[0]]
        self.assertEqual(len(notices), 1)
        self.assertIn("Kaggle UI", notices[0])

    def test_run_retries_transient_status_failures(self):
        runner = KaggleRunner(Config(project_dir=Path.cwd(), name="run", kaggle=KaggleConfig(user="tester")))
        statuses = iter(("status: running", "status: complete"))

        def kaggle(*args):
            if args[1] == "logs":
                return mock.Mock(returncode=0, stdout="[]")
            if args[0:3] == ("kernels", "status", runner.kernel_slug) and not getattr(kaggle, "failed", False):
                kaggle.failed = True
                return mock.Mock(returncode=1, stderr="DNS lookup failed", stdout="")
            return mock.Mock(returncode=0, stdout=next(statuses))

        errors = io.StringIO()
        with mock.patch("remote_gpu.kaggle_manager._kaggle", side_effect=kaggle), mock.patch(
            "remote_gpu.kaggle_manager.time.sleep"
        ) as sleep, redirect_stderr(errors):
            self.assertEqual(runner._wait(), "complete")
        sleep.assert_any_call(5)
        self.assertIn("DNS lookup failed", errors.getvalue())

    def test_logs_follow_streams_native_kaggle_output(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            process = mock.Mock(stdout=io.BytesIO("step 1\rstep 2\n".encode("utf-8")))
            process.wait.return_value = 0
            output = io.StringIO()
            with mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
                returncode=0, stdout="-f, --follow  Stream live execution logs"
            )) as kaggle, mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", return_value=process) as popen, redirect_stdout(output):
                runner.stream_logs(show_all=True, follow=True, save=str(root / "run.log"))
            kaggle.assert_called_once_with("kernels", "logs", "--help")
            self.assertEqual(popen.call_args.args[0], ["kaggle", "kernels", "logs", "-f", runner.kernel_slug])
            self.assertEqual(popen.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")
            self.assertEqual(output.getvalue(), "step 1\rstep 2\n")
            self.assertEqual((root / "run.log").read_bytes().decode("utf-8"), output.getvalue())

    def test_logs_follow_does_not_stop_kernel_on_interrupt(self):
        runner = KaggleRunner(Config(project_dir=Path.cwd(), name="run", kaggle=KaggleConfig(user="tester")))
        process = mock.Mock(stdout=io.BytesIO())
        process.poll.return_value = None
        stream = mock.Mock()
        stream.read.side_effect = KeyboardInterrupt
        with mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
            returncode=0, stdout="-f, --follow"
        )), mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", return_value=process), mock.patch(
            "remote_gpu.kaggle_manager.io.TextIOWrapper", return_value=stream
        ):
            runner.stream_logs(show_all=False, follow=True, save=None)
        process.terminate.assert_called_once_with()
        stream.close.assert_called_once_with()

    def test_logs_follow_reports_cli_failure(self):
        runner = KaggleRunner(Config(project_dir=Path.cwd(), name="run", kaggle=KaggleConfig(user="tester")))
        process = mock.Mock(stdout=io.BytesIO())
        process.wait.return_value = 1
        process.poll.return_value = 1
        with mock.patch("remote_gpu.kaggle_manager._kaggle", return_value=mock.Mock(
            returncode=0, stdout="-f, --follow"
        )), mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", return_value=process):
            with self.assertRaisesRegex(click.ClickException, "Kaggle live logs failed"):
                runner.stream_logs(show_all=False, follow=True, save=None)

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
            self.assertEqual(sum(call.args[:2] == ("kernels", "logs") and "--help" not in call.args for call in calls.call_args_list), 2)

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

    def test_notebook_cell_ids_are_valid_unique_and_stable(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            src, dst, second = (root / name for name in ("source.ipynb", "upload.ipynb", "again.ipynb"))
            cells = [
                {"cell_type": "markdown", "metadata": {}, "source": ["no id"]},
                {"cell_type": "code", "id": "remote-gpu-setup", "metadata": {}, "source": ["pass"], "execution_count": None, "outputs": []},
                {"cell_type": "code", "id": "existing", "metadata": {}, "source": ["pass"], "execution_count": None, "outputs": []},
                {"cell_type": "code", "id": "existing", "metadata": {}, "source": ["pass"], "execution_count": None, "outputs": []},
            ]
            original = {"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 4}
            src.write_text(json.dumps(original), encoding="utf-8")
            config = Config(project_dir=root, name="run")
            kaggleify_notebook(src, dst, config)
            notebook = json.loads(dst.read_text(encoding="ascii"))
            ids = [cell["id"] for cell in notebook["cells"]]
            self.assertEqual(notebook["nbformat_minor"], 5)
            self.assertEqual(ids[0], "remote-gpu-setup-1")
            self.assertIn("https://github.com/TheodorBeskow/remote-gpu", "".join(notebook["cells"][0]["source"]))
            self.assertEqual(ids[2], "remote-gpu-setup")
            self.assertEqual(ids[3], "existing")
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(all(re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", cell_id) for cell_id in ids))
            self.assertEqual(json.loads(src.read_text(encoding="utf-8")), original)

            kaggleify_notebook(src, second, config)
            self.assertEqual(json.loads(second.read_text(encoding="ascii")), notebook)
            kaggleify_notebook(dst, second, config)
            self.assertEqual(json.loads(second.read_text(encoding="ascii")), notebook)

    def test_example_notebook_has_ids_after_conversion(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            source = Path(__file__).parent / "example" / "solve.ipynb"
            output = root / "solve.ipynb"
            kaggleify_notebook(source, output, Config(project_dir=root, name="run"))
            notebook = json.loads(output.read_text(encoding="ascii"))
            cells = notebook["cells"]
            self.assertTrue(all("id" in cell for cell in cells))
            self.assertEqual(len({cell["id"] for cell in cells}), len(cells))
            if importlib.util.find_spec("nbformat") is not None:
                import nbformat

                nbformat.validate(nbformat.from_dict(notebook))

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
        managed = "/kaggle/input/custom/remote-gpu-data"
        iris = "/kaggle/input/custom/datasets/uciml/iris"
        with mock.patch("os.path.exists", side_effect=lambda path: path in ("/kaggle", "/kaggle/input/custom")), mock.patch(
            "os.path.isdir", side_effect=lambda path: path in (managed, iris)
        ), mock.patch("os.scandir", side_effect=AssertionError("direct mounts need no scan")), mock.patch(
            "os.makedirs"
        ), mock.patch("os.symlink") as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
            "os.path.lexists", return_value=False
        ), mock.patch.dict("sys.modules", {"torch": None}):
            exec(preamble, {})
        symlink.assert_any_call(managed, "assets/input")
        symlink.assert_any_call(iris, "iris")
        symlink.assert_any_call("/kaggle/working/custom/results/output", "results/output")

    def test_dataset_mount_avoids_scanning_dataset_contents(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            alice = root / "datasets" / "alice" / "shared"
            bob = root / "datasets" / "bob" / "shared"
            alice.mkdir(parents=True)
            bob.mkdir(parents=True)
            (alice / "panoramas").mkdir()
            config = Config(
                project_dir=root, name="run", paths=Paths(kaggle_input=root.as_posix()),
                datasets={"alias_alice": "alice/shared", "alias_bob": "bob/shared"},
            )
            real_exists = os.path.exists
            with mock.patch("os.path.exists", side_effect=lambda path: path == "/kaggle" or real_exists(path)), mock.patch(
                "glob.glob", side_effect=AssertionError("recursive glob must not run")
            ), mock.patch("os.scandir", side_effect=AssertionError("direct mounts need no directory scan")), mock.patch(
                "os.makedirs"
            ), mock.patch("os.symlink") as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
                "os.path.lexists", return_value=False
            ), mock.patch.dict("sys.modules", {"torch": None}):
                exec(build_preamble(config), {})
            symlink.assert_any_call(alice.as_posix(), "alias_alice")
            symlink.assert_any_call(bob.as_posix(), "alias_bob")

    def test_dataset_mount_fallback_is_bounded(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            for index in range(257):
                (root / f"mount-{index}").mkdir()
            config = Config(
                project_dir=root, name="run", paths=Paths(kaggle_input=root.as_posix()),
                datasets={"missing_alias": "owner/not-mounted"},
            )
            real_exists = os.path.exists
            with mock.patch("os.path.exists", side_effect=lambda path: path == "/kaggle" or real_exists(path)), mock.patch(
                "glob.glob", side_effect=AssertionError("recursive glob must not run")
            ), mock.patch.dict("sys.modules", {"torch": None}):
                with self.assertRaisesRegex(RuntimeError, "mount search limit reached"):
                    exec(build_preamble(config), {})

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            flat = root / "datasets"
            flat.mkdir()
            for index in range(257):
                (flat / f"panorama-{index}").mkdir()
            config = Config(
                project_dir=root, name="run", paths=Paths(kaggle_input=root.as_posix()),
                datasets={"flat_alias": "owner/datasets"},
            )
            real_exists, real_scandir = os.path.exists, os.scandir

            def shallow_scandir(path):
                self.assertEqual(str(path).replace("\\", "/"), root.as_posix())
                return real_scandir(path)

            with mock.patch("os.path.exists", side_effect=lambda path: path == "/kaggle" or real_exists(path)), mock.patch(
                "os.scandir", side_effect=shallow_scandir
            ), mock.patch("glob.glob", side_effect=AssertionError("recursive glob must not run")), mock.patch(
                "os.makedirs"
            ), mock.patch("os.symlink") as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
                "os.path.lexists", return_value=False
            ), mock.patch.dict("sys.modules", {"torch": None}):
                exec(build_preamble(config), {})
            symlink.assert_any_call(flat.as_posix(), "flat_alias")

    def test_dataset_mount_prefers_full_owner_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            alice = root / "datasets" / "alice" / "shared"
            bob = root / "datasets" / "bob" / "shared"
            alice.mkdir(parents=True)
            bob.mkdir(parents=True)
            config = Config(
                project_dir=root, name="run", paths=Paths(kaggle_input=root.as_posix()),
                datasets={"alice_data": "alice/shared", "bob_data": "bob/shared"},
            )
            real_exists, real_scandir = os.path.exists, os.scandir
            scanned = []

            def shallow_scandir(path):
                scanned.append(str(path).replace("\\", "/"))
                self.assertIn(scanned[-1], {root.as_posix(), (root / "datasets").as_posix(), (root / "flat").as_posix()})
                return real_scandir(path)

            def resolve():
                with mock.patch("os.path.exists", side_effect=lambda path: path == "/kaggle" or real_exists(path)), mock.patch(
                    "glob.glob", side_effect=AssertionError("recursive glob must not run")
                ), mock.patch("os.scandir", side_effect=shallow_scandir), mock.patch("os.makedirs"), mock.patch(
                    "os.symlink"
                ) as symlink, mock.patch("os.listdir", return_value=[]), mock.patch(
                    "os.path.lexists", return_value=False
                ), mock.patch.dict("sys.modules", {"torch": None}):
                    exec(build_preamble(config), {})
                return symlink

            symlink = resolve()
            symlink.assert_any_call(alice.as_posix(), "alice_data")
            symlink.assert_any_call(bob.as_posix(), "bob_data")
            self.assertEqual(scanned, [])

            config.datasets = {"missing": "charlie/shared"}
            with self.assertRaisesRegex(RuntimeError, "ambiguous.*charlie/shared"):
                resolve()
            self.assertEqual(set(scanned), {root.as_posix(), (root / "datasets").as_posix()})

            flat_root = root / "flat"
            (flat_root / "shared").mkdir(parents=True)
            config.paths.kaggle_input = flat_root.as_posix()
            config.datasets = {"flat_alias": "alice/shared"}
            scanned.clear()
            symlink = resolve()
            symlink.assert_any_call((flat_root / "shared").as_posix(), "flat_alias")
            self.assertEqual(set(scanned), {flat_root.as_posix()})

            config.datasets = {"alice_data": "alice/shared", "bob_data": "bob/shared"}
            with self.assertRaisesRegex(RuntimeError, "ambiguous.*shared"):
                resolve()

    def test_output_download_retries_from_persistent_stage(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            stages = []

            def download(command, **kwargs):
                self.assertEqual(command[command.index("--page-size") + 1], "200")
                stage = Path(command[command.index("-p") + 1])
                stages.append(stage)
                self.assertEqual(stage.parent.parent, root / ".remote-gpu")
                output = stage / "output"
                output.mkdir(parents=True, exist_ok=True)
                first = output / "first.pt"
                first.write_bytes(b"first")
                if len(stages) == 1:
                    return mock.Mock(stdout=io.BytesIO(b"RemoteDisconnected: remote end closed\n"), wait=lambda: 1)
                self.assertEqual(first.read_bytes(), b"first")
                second = output / "second.pt"
                second.write_bytes(b"second")
                return mock.Mock(stdout=io.BytesIO(f"Output file downloaded to {second}\n".encode()), wait=lambda: 0)

            printed = io.StringIO()
            errors = io.StringIO()
            with mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", side_effect=download) as popen, mock.patch(
                "remote_gpu.kaggle_manager._kaggle", side_effect=AssertionError("output must stream")
            ), mock.patch("remote_gpu.kaggle_manager.time.sleep") as sleep, redirect_stdout(printed), redirect_stderr(errors):
                self.assertTrue(runner._fetch_output(root))
            self.assertEqual(popen.call_count, 2)
            sleep.assert_called_once_with(5)
            self.assertEqual(stages[0], stages[1])
            self.assertEqual((root / "output" / "first.pt").read_bytes(), b"first")
            self.assertEqual((root / "output" / "second.pt").read_bytes(), b"second")
            self.assertFalse(stages[0].exists())
            self.assertIn("downloaded 1", printed.getvalue())
            self.assertIn("transfer complete", printed.getvalue())
            self.assertIn("retrying", errors.getvalue())

    def test_output_download_stops_after_retry_limit(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))

            def download(command, **kwargs):
                stage = Path(command[command.index("-p") + 1])
                (stage / "output").mkdir(parents=True, exist_ok=True)
                (stage / "output" / "partial.pt").write_bytes(b"partial")
                return mock.Mock(stdout=io.BytesIO(b"HTTP 503 service unavailable\n"), wait=lambda: 1)

            with mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", side_effect=download) as popen, mock.patch(
                "remote_gpu.kaggle_manager.time.sleep"
            ) as sleep, redirect_stderr(io.StringIO()):
                with self.assertRaisesRegex(click.ClickException, "after 3 attempts"):
                    runner._fetch_output(root)
            self.assertEqual(popen.call_count, 3)
            self.assertEqual(sleep.call_args_list, [mock.call(5), mock.call(10)])
            stage = root / ".remote-gpu" / "downloads" / runner._load_state()["download_run_id"]
            self.assertEqual((stage / "output" / "partial.pt").read_bytes(), b"partial")
            self.assertFalse((root / "output").exists())

    def test_output_copy_retries_without_recopying_finished_files(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            stage = runner._output_stage()
            (stage / "output").mkdir()
            (stage / "output" / "a.pt").write_bytes(b"a")
            (stage / "output" / "b.pt").write_bytes(b"b")
            def finished_download(*args, **kwargs):
                process = mock.Mock(stdout=io.BytesIO())
                process.wait.return_value = 0
                return process

            real_copy = shutil.copy2
            attempts = 0

            def interrupted_copy(source, destination):
                nonlocal attempts
                attempts += 1
                if attempts == 2:
                    raise OSError("disk interrupted")
                return real_copy(source, destination)

            with mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", side_effect=finished_download), mock.patch(
                "remote_gpu.kaggle_manager.shutil.copy2", side_effect=interrupted_copy
            ):
                with self.assertRaisesRegex(click.ClickException, "staged files retained"):
                    runner._fetch_output(root)
            self.assertEqual(len(list((root / "output").iterdir())), 1)
            with mock.patch("remote_gpu.kaggle_manager.subprocess.Popen", side_effect=finished_download), mock.patch(
                "remote_gpu.kaggle_manager.shutil.copy2", wraps=real_copy
            ) as copy, redirect_stdout(io.StringIO()) as printed:
                self.assertTrue(runner._fetch_output(root))
            self.assertEqual(copy.call_count, 1)
            self.assertRegex(printed.getvalue(), r"1 copied \(1 B\), 1 unchanged")
            self.assertEqual((root / "output" / "a.pt").read_bytes(), b"a")
            self.assertEqual((root / "output" / "b.pt").read_bytes(), b"b")
            self.assertFalse(stage.exists())

    def test_invalid_download_state_cannot_escape_ignored_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            runner._save_state({"download_kernel": runner.kernel_slug, "download_run_id": "../../outside"})
            with self.assertRaisesRegex(click.ClickException, "invalid download run ID"):
                runner._output_stage()

    def test_temporary_upload_files_stay_in_ignored_directory(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            with runner._temp_dir() as path:
                self.assertEqual(Path(path).parent, root / ".remote-gpu")
                self.assertTrue(Path(path).is_dir())
            self.assertFalse(Path(path).exists())

    def test_new_launch_uses_new_download_stage(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as directory:
            root = Path(directory)
            runner = KaggleRunner(Config(project_dir=root, name="run", kaggle=KaggleConfig(user="tester")))
            with mock.patch.object(runner, "_sync_dataset"), mock.patch.object(runner, "_push_kernel"):
                runner.launch(root / "solve.py")
                stage = runner._output_stage()
                (stage / "previous.pt").write_bytes(b"previous")
                runner.launch(root / "solve.py")
                self.assertNotEqual(runner._output_stage(), stage)
                self.assertTrue((stage / "previous.pt").exists())

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

            process = mock.Mock(stdout=io.BytesIO())
            process.wait.return_value = 0
            with mock.patch.object(runner, "_output_stage", return_value=download), mock.patch(
                "remote_gpu.kaggle_manager.subprocess.Popen", return_value=process
            ):
                runner._fetch_output(root)
            self.assertEqual((root / "output" / "result.txt").read_text(), "done")
            self.assertEqual((root / "output" / "log.txt").read_text(), "log")
            self.assertFalse((root / "output" / "custom").exists())


if __name__ == "__main__":
    unittest.main()
