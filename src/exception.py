import sys

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    
    return f"Error occurred in the python script [{file_name}] at line number [{exc_tb.tb_lineno}] : {str(error)}"

class CustomException(Exception):
    def __init__(self, error, error_detail):
        self.error_message = error_message_detail(error, error_detail)
        super().__init__(self.error_message)