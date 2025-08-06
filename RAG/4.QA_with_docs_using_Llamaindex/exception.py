import sys  

class CustomError(Exception):
    def __init__(self, error_message, error_details):

        self.error_message = error_message 
        
        _, _, error_traceback = error_details.exc_info()

        self.line_number = error_traceback.tb_lineno 
        
        self.file_name = error_traceback.tb_frame.f_code.co_filename 
    
    def __str__(self):
        return f"Error in script '{self.file_name}' at line {self.line_number}: {self.error_message}"  # 

if __name__ == "__main__":
    try:
        pass
    except Exception as error:
        raise CustomError(error, sys)
