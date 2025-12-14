"""
Logger Utility
==============

Provides logging functionality to capture terminal output to a file.
"""
import sys

from pathlib import Path
from typing  import Optional


class TeeStream:
  """
  A stream that writes to both stdout and a file.
  """
  def __init__(self, file_path: Path):
    self.terminal = sys.stdout
    self.log_file = open(file_path, 'w')
  
  def write(self, message: str):
    self.terminal.write(message)
    self.log_file.write(message)
    self.log_file.flush()
  
  def flush(self):
    self.terminal.flush()
    self.log_file.flush()
  
  def close(self):
    self.log_file.close()


class LoggerContext:
  """
  Context to hold logger state for cleanup.
  """
  def __init__(
    self,
    tee_stdout : TeeStream,
    tee_stderr : TeeStream,
    original_stdout,
    original_stderr,
  ):
    self.tee_stdout      = tee_stdout
    self.tee_stderr      = tee_stderr
    self.original_stdout = original_stdout
    self.original_stderr = original_stderr


def start_logging(
  log_filepath: Path,
) -> LoggerContext:
  """
  Start logging terminal output (stdout and stderr) to a file.
  
  Input:
  ------
    log_filepath : Path
      Path to the log file.
      
  Output:
  -------
    context : LoggerContext
      Context object for cleanup.
  """
  # Store original streams
  original_stdout = sys.stdout
  original_stderr = sys.stderr
  
  # Create tee streams
  tee_stdout = TeeStream(log_filepath)
  tee_stderr = TeeStream(log_filepath)
  
  # Redirect stdout and stderr
  sys.stdout = tee_stdout
  sys.stderr = tee_stderr
  
  return LoggerContext(
    tee_stdout,
    tee_stderr,
    original_stdout,
    original_stderr,
  )


def stop_logging(
  context: Optional[LoggerContext],
) -> None:
  """
  Stop logging and restore original stdout/stderr.
  
  Input:
  ------
    context : LoggerContext | None
      Context object from start_logging.
      
  Output:
  -------
    None
  """
  if context is None:
    return
  
  # Restore original streams
  sys.stdout = context.original_stdout
  sys.stderr = context.original_stderr
  
  # Close log files
  context.tee_stdout.close()
  context.tee_stderr.close()
