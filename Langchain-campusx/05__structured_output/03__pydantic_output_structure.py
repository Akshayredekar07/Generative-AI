
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

class Student(BaseModel):
    name : str = "Akshay"
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0, lt=10, default=7.9, description='decimal values is the student cgpa')


new_student = {
    'name': 'Karan',
    'age' : 34,
    'email': 'abc@gmail.com'
}

student = Student(**new_student)

student_dict = dict(student)

print(student_dict['age'])


student_json = student.model_dump_json()