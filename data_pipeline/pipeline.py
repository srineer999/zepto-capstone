import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re

BASE = "https://books.toscrape.com/"
headers = {"User-Agent": "Mozilla/5.0"}

# Get category links
html = requests.get(BASE, headers=headers).text
soup = BeautifulSoup(html, "html.parser")

category_links = soup.select("ul.nav-list ul li a")[:4]

books = []

# Scrape 3 categories
for link in category_links:
    category = link.text.strip()
    url = BASE + link["href"]

    html = requests.get(url, headers=headers).text
    soup = BeautifulSoup(html, "html.parser")

    for book in soup.select("article.product_pod"):
        title = book.h3.a["title"]

        price = book.select_one(".price_color").text
        price_gbp = float(re.sub(r"[^0-9.]", "", price))

        rating_word = book.select_one(".star-rating")["class"][1]

        rating_map = {
            "One": 1,
            "Two": 2,
            "Three": 3,
            "Four": 4,
            "Five": 5
        }

        rating = rating_map[rating_word]

        availability = book.select_one(".availability").text.strip()
        in_stock = "In stock" in availability

        books.append([
            title,
            price_gbp,
            rating,
            in_stock,
            category
        ])

# Create DataFrame
df = pd.DataFrame(
    books,
    columns=[
        "title",
        "price_gbp",
        "rating",
        "in_stock",
        "category"
    ]
)

# GBP to INR
df["price_inr"] = df["price_gbp"] * 105.50

print("TOTAL BOOKS:", len(df))
print(df.head())

# Create SQLite database
conn = sqlite3.connect("books.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS categories (
    category_id INTEGER PRIMARY KEY,
    category_name TEXT UNIQUE
)
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS books (
    book_id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    price_gbp REAL,
    rating INTEGER,
    in_stock INTEGER,
    price_inr REAL,
    category_id INTEGER,
    FOREIGN KEY(category_id) REFERENCES categories(category_id)
)
""")

# Insert categories
categories = df["category"].unique()

for i, category in enumerate(categories, 1):
    cur.execute(
        "INSERT OR IGNORE INTO categories VALUES (?, ?)",
        (i, category)
    )

# Insert books
for _, row in df.iterrows():

    category_id = list(categories).index(row["category"]) + 1

    cur.execute("""
    INSERT INTO books
    (title, price_gbp, rating, in_stock, price_inr, category_id)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (
        row["title"],
        row["price_gbp"],
        row["rating"],
        row["in_stock"],
        row["price_inr"],
        category_id
    ))

conn.commit()

# Required SQL queries

queries = [

    "SELECT * FROM books LIMIT 10",

    "SELECT * FROM books WHERE rating >= 4",

    """SELECT category_id, COUNT(*) AS total_books
       FROM books
       GROUP BY category_id""",

    """SELECT * FROM books
       ORDER BY price_inr DESC
       LIMIT 10""",

    """SELECT b.title, b.rating, c.category_name
       FROM books b
       JOIN categories c
       ON b.category_id = c.category_id
       ORDER BY b.rating DESC"""
]

for i, query in enumerate(queries, 1):

    print("\nQUERY", i)

    result = pd.read_sql(query, conn)

    print(result)

# Read tables using pandas
books_sql = pd.read_sql(
    "SELECT * FROM books",
    conn
)

categories_sql = pd.read_sql(
    "SELECT * FROM categories",
    conn
)

# Required merge
merged = pd.merge(
    books_sql,
    categories_sql,
    on="category_id"
)

print("\nMERGED RESULT")
print(merged.head())

conn.close()

print("\nPIPELINE COMPLETED SUCCESSFULLY!")