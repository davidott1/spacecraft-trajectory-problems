import sys
from pathlib import Path
from typing  import Optional


class DualOutputLogger:
  """
  A class that duplicates stdout to both the terminal and a log file.
  
  Usage:
  ------
    log_filepath = Path(<log_filepath>)
    logger = DualOutputLogger(log_filepath)
    logger.start()
    # ... all print statements go to both terminal and file ...
    logger.stop()
  """
  
  def __init__(
    self,
    log_filepath : Path,
  ) -> None:
    """
    Initialize the DualOutputLogger.
    
    Input:
    ------
      log_filepath : Path
        Path to the log file.
    """
    self.log_filepath    = log_filepath
    self.log_file        = None
    self.original_stdout = None
  
  def start(
    self,
  ) -> None:
    """
    Start logging to the file while preserving terminal output.
    """
    self.original_stdout = sys.stdout
    self.log_file        = open(self.log_filepath, 'w')
    sys.stdout           = self
  
  def stop(
    self,
  ) -> None:
    """
    Stop logging and restore original stdout.
    """
    if self.original_stdout is not None:
      sys.stdout = self.original_stdout
    if self.log_file is not None:
      self.log_file.close()
      self.log_file = None
  
  def write(
    self,
    message : str,
  ) -> None:
    """
    Write message to both terminal and log file.
    
    Input:
    ------
      message : str
        The message to write.
    """
    if self.original_stdout is not None:
      self.original_stdout.write(message)
    if self.log_file is not None:
      self.log_file.write(message)
  
  def flush(
    self,
  ) -> None:
    """
    Flush both terminal and log file buffers.
    
    This forces any buffered output to be immediately written to the terminal
    and log file, rather than waiting in memory. Required for sys.stdout
    compatibility and ensures real-time output display.
    """
    if self.original_stdout is not None:
      self.original_stdout.flush()
    if self.log_file is not None:
      self.log_file.flush()


def start_logging(
  log_filepath: Path,
) -> DualOutputLogger:
  """
  Create and start a DualOutputLogger for the given filepath.
  
  Input:
  ------
    log_filepath : Path
      Path to the log file.
      
  Output:
  -------
    DualOutputLogger
      The started logger instance (call .stop() when done).
  """
  logger = DualOutputLogger(log_filepath)
  logger.start()
  return logger


def stop_logging(
  logger: DualOutputLogger,
) -> None:
  """
  Stop the DualOutputLogger and restore original stdout.
  
  Input:
  ------
    logger : DualOutputLogger
      The logger instance to stop.
  """
  logger.stop()
