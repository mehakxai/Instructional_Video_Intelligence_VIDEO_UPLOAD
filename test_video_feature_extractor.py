import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import video_feature_extractor as vfe


class FindFFmpegTests(unittest.TestCase):
    def test_add_ffmpeg_to_path_exposes_binary_directory(self):
        with patch.dict("video_feature_extractor.os.environ", {"PATH": "existing"}):
            vfe._add_ffmpeg_to_path("/tmp/ffmpeg-bin/ffmpeg")
            self.assertEqual(
                vfe.os.environ["PATH"],
                str(vfe.Path("/tmp/ffmpeg-bin")) + os.pathsep + "existing",
            )

    @patch("video_feature_extractor.shutil.which", return_value="C:/ffmpeg/bin/ffmpeg.exe")
    def test_find_ffmpeg_executable_uses_path_when_available(self, mock_which):
        self.assertEqual(vfe.find_ffmpeg_executable(), "C:/ffmpeg/bin/ffmpeg.exe")

    @patch("video_feature_extractor.shutil.which", return_value=None)
    @patch("video_feature_extractor.os.name", "nt")
    @patch("video_feature_extractor.Path.exists", return_value=True)
    def test_find_ffmpeg_executable_falls_back_to_common_windows_locations(self, mock_exists, mock_which):
        with patch("video_feature_extractor.os.environ", {"LOCALAPPDATA": "C:/Users/Test/AppData/Local"}):
            result = vfe.find_ffmpeg_executable()
        self.assertIsNotNone(result)

    @patch("video_feature_extractor.shutil.which", return_value=None)
    def test_find_ffmpeg_executable_uses_imageio_ffmpeg_fallback(self, mock_which):
        class FakeImageIOFFmpeg:
            @staticmethod
            def get_ffmpeg_exe():
                return "/tmp/ffmpeg-bin/ffmpeg"

        with patch.dict("sys.modules", {"imageio_ffmpeg": FakeImageIOFFmpeg}):
            with patch("video_feature_extractor.Path.exists", return_value=True):
                self.assertEqual(vfe.find_ffmpeg_executable(), "/tmp/ffmpeg-bin/ffmpeg")

    @patch("video_feature_extractor.find_ffmpeg_executable", return_value=None)
    def test_audio_to_wav_raises_clear_error_when_ffmpeg_missing(self, mock_finder):
        with tempfile.TemporaryDirectory() as td:
            video = os.path.join(td, "video.mp4")
            with open(video, "wb") as f:
                f.write(b"dummy")
            with self.assertRaises(FileNotFoundError):
                vfe.audio_to_wav(video, os.path.join(td, "audio.wav"))

    @patch("video_feature_extractor.find_ffmpeg_executable", return_value="C:/ffmpeg/bin/ffmpeg.exe")
    @patch("video_feature_extractor.subprocess.run")
    def test_audio_to_wav_generates_silence_when_video_has_no_audio(self, mock_run, mock_finder):
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["ffmpeg", "-i", "video.mp4", "-vn", "-ac", "1", "-ar", "16000", "audio.wav"],
                returncode=1,
                stderr=b"Output file does not contain any stream",
                stdout=b"",
            ),
            subprocess.CompletedProcess(
                args=["ffmpeg", "-y", "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono", "-t", "1", "-ar", "16000", "-ac", "1", "audio.wav"],
                returncode=0,
                stderr=b"",
                stdout=b"",
            ),
        ]
        with tempfile.TemporaryDirectory() as td:
            video = os.path.join(td, "video.mp4")
            audio = os.path.join(td, "audio.wav")
            vfe.audio_to_wav(video, audio)
            self.assertEqual(mock_run.call_count, 2)
            self.assertIn("anullsrc", " ".join(mock_run.call_args_list[1].args[0]))


if __name__ == "__main__":
    unittest.main()
