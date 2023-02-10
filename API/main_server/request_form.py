from pydantic import BaseModel


class LabeledData(BaseModel):
    check1: bool
    check2: bool
    check3: bool
    id: int
    text: str
    labeledText: str
    nickName: str

class CheckedData(BaseModel):
    label: int
    nick_name: str
    target: str
    text: str
    id: int

class RewardData(BaseModel):
    reward1: int
    reward2: int
    reward3: int
    reward4: int
    id: int
    
class PostData(BaseModel):
    id: int
    
class CommentData(BaseModel):
    id: int
    comment: str
    fixed: str