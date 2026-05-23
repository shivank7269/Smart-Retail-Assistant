from pydantic import BaseModel
from datetime import datetime

class SalesRecord(BaseModel):
    order_id : str
    order_date : datetime
    region : str
    category : str
    product_id : str
    quantity : int
    sales : float
    discount : float
    profit : float

