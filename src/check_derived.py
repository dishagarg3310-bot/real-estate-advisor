import pandas as pd

d = pd.read_csv("data/model_data.csv")
flags = [c for c in d.columns if c.startswith("Amenity_") and c != "Amenity_Count"]
print("Amenity flags:", flags)
print("Price/sqft max diff:", (d.Price_in_Lakhs * 100000 / d.Size_in_SqFt - d.Price_per_SqFt_Rupees).abs().max())
print("Floor_Inconsistent match:", ((d.Floor_No > d.Total_Floors).astype(int) == d.Floor_Inconsistent).mean())
print("Amenity_Count match:", (d[flags].sum(axis=1) == d.Amenity_Count).mean())
print(d[["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Age_of_Property", "Floor_No", "Total_Floors"]].agg(["min", "max"]))