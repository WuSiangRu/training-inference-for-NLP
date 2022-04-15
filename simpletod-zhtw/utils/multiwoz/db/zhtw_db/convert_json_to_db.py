#%%
import pandas as pd
import sqlite3

# %%
df = pd.read_json("zhtw_attraction_db.json")
df[["location"]] = df[["location"]].applymap(str)

conn = sqlite3.connect("zhtw_attraction_dbase.db")
c = conn.cursor()

c.execute(
    "CREATE TABLE ATTRACTIONS (地址 text, 區域 text, 費用 text, id text, location text, 名稱 text, openhours text, 電話 text, 郵編 text, 價格範圍 text, 型別 text)"
)
conn.commit()

df.to_sql("ATTRACTIONS", conn, if_exists="replace", index=True)
# %%
df = pd.read_json("zhtw_restaurant_db.json")
df[["location"]] = df[["location"]].applymap(str)

conn = sqlite3.connect("zhtw_restaurant_dbase.db")
c = conn.cursor()

c.execute(
    "CREATE TABLE RESTAURANTS (地址 text, 區域 text, 食物 text, id text, introduction text, location text, 名稱 text, 電話 text, 郵編 text, 價格範圍 text, 型別 text)"
)
conn.commit()

df.to_sql("RESTAURANTS", conn, if_exists="replace", index=True)

#%%
df = pd.read_json("zhtw_hotel_db.json")
df[["location"]] = df[["location"]].applymap(str)
df[["價格"]] = df[["價格"]].applymap(str)

conn = sqlite3.connect("zhtw_hotel_dbase.db")
c = conn.cursor()

c.execute(
    "CREATE TABLE HOTELS (地址 text, 區域 text, 網際網路 text, 停車處 text, id text, location text, 名稱 text, 電話 text, 郵編 text, 價格 text, 價格範圍 text, 星級 text, takesbookings text, 型別 text)"
)
conn.commit()

df.to_sql("HOTELS", conn, if_exists="replace", index=True)
# %%
df = pd.read_json("zhtw_train_db.json")

conn = sqlite3.connect("zhtw_train_dbase.db")
c = conn.cursor()

c.execute(
    "CREATE TABLE TRAINS (到達時間 text, 日期 text, 出發地 text, 目的地 text, 時間 text, 出發時間 text, 價格 text, 列車號 text)"
)
conn.commit()

df.to_sql("TRAINS", conn, if_exists="replace", index=True)
# %%
df = pd.read_json("zhtw_train_db.json")

conn = sqlite3.connect("zhtw_train_dbase.db")
c = conn.cursor()

c.execute(
    "CREATE TABLE TRAINS (到達時間 text, 日期 text, 出發地 text, 目的地 text, 時間 text, 出發時間 text, 價格 text, 列車號 text)"
)
conn.commit()

df.to_sql("TRAINS", conn, if_exists="replace", index=True)

#%%

# the original hospital and taxi db files seem to be empty.