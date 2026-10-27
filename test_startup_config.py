"""Tests for startup prompt order, validation, and non-interactive overrides."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import startup_config


class StartupConfigTests(unittest.TestCase):
    def test_startup_asks_method_then_rpm_and_retries_invalid_values(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=True), \
                patch("builtins.input", side_effect=["x", "2", "zero", "7"]) as prompt:
            mode, rpm, pending = startup_config.choose_startup_config("deepseek")

        self.assertEqual((mode, rpm, pending), ("user_only", 7, 100))
        self.assertEqual(prompt.call_count, 4)
        self.assertIn("method 1 or 2", prompt.call_args_list[1].args[0])
        self.assertIn("requests per minute", prompt.call_args_list[3].args[0])

    def test_environment_values_skip_interactive_prompts(self):
        with patch.dict(os.environ, {
            "PROMPT_MODE": "2",
            "DEEPSEEK_RPM": "12",
            "DEEPSEEK_MAX_QUEUE": "30",
            "BRIDGE_STARTUP_CONFIGURED": "1",
        }, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=True), \
                patch("builtins.input") as prompt:
            config = startup_config.choose_startup_config("deepseek")

        self.assertEqual(config, ("user_only", 12, 30))
        prompt.assert_not_called()

    def test_interactive_launch_still_asks_when_environment_defaults_exist(self):
        with patch.dict(os.environ, {
            "PROMPT_MODE": "user_only",
            "DEEPSEEK_RPM": "12",
        }, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=True), \
                patch("builtins.input", side_effect=["", ""]) as prompt:
            config = startup_config.choose_startup_config("deepseek")

        self.assertEqual(config, ("user_only", 12, 100))
        self.assertEqual(prompt.call_count, 2)
        self.assertIn("default 2", prompt.call_args_list[0].args[0])
        self.assertIn("[12]", prompt.call_args_list[1].args[0])

    def test_noninteractive_launch_uses_safe_defaults(self):
        with patch.dict(os.environ, {}, clear=True), \
                patch.object(startup_config, "_is_interactive", return_value=False):
            self.assertEqual(startup_config.choose_startup_config("qwen"), ("full", 5, 100))

    def test_environment_values_are_explicit_for_run_both_children(self):
        base = {"EXISTING": "kept", "BRIDGE_RPM": "99"}
        env = startup_config.environment_for_child("qwen", "full", 8, 25, base)
        self.assertEqual(env["EXISTING"], "kept")
        self.assertEqual(env["PROMPT_MODE"], "full")
        self.assertEqual(env["QWEN_RPM"], "8")
        self.assertEqual(env["QWEN_MAX_QUEUE"], "25")
        self.assertEqual(env["BRIDGE_STARTUP_CONFIGURED"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
