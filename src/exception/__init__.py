import sys
import logging
import traceback

def error_message_detail(error: Exception, error_detail: sys) -> str:
    """
    Extract detailed error information including file name, line number, and the error message.

    :param error: The exception object.
    :param error_detail: The sys module to access traceback details.
    :return: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    # Walk to the last traceback frame (deepest cause)
    while exc_tb and exc_tb.tb_next:
        exc_tb = exc_tb.tb_next

    # Log the full traceback for better debugging

    logging.error(error)
    traceback.print_exc()

    return error


class MyException(Exception):
    """
    Custom exception class for handling errors in your project.
    """
    def __init__(self, error: Exception, error_detail: sys):
        """
        Initializes MyException with a detailed error message.

        :param error: The exception object to wrap.
        :param error_detail: The sys module to access traceback details.
        """
        super().__init__(str(error))
        self.error_message = error_message_detail(error, error_detail)

    def __str__(self) -> str:
        """
        Returns the string representation of the error message.
        """
        return self.error_message
