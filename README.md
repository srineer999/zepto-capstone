# Zepto Data & AI Platform

## Data Pipeline

This module scrapes book data from Books to Scrape, cleans the data,
converts GBP prices to INR, stores the data in SQLite, and performs SQL analysis.

## Setup

Install the required libraries:

pip install requests beautifulsoup4 pandas

## Run

py data_pipeline/pipeline.py

## Data Collected

The pipeline collects:

- Book title
- Price in GBP
- Rating
- Availability
- Category

The pipeline collected 71 books from multiple categories.

## Data Cleaning

- Price is converted to float.
- Rating is converted to an integer from 1 to 5.
- Availability is converted to an in_stock boolean.
- GBP price is converted to INR.

The fixed conversion rate is:

1 GBP = 105.50 INR

## Database

SQLite is used as the relational database.

The database contains two tables:

- categories
- books

The books table is connected to categories using category_id.

## SQL Analysis

Five SQL queries are executed covering:

1. Book listing
2. Rating filtering
3. Category grouping
4. Price ordering
5. JOIN between books and categories

SQL results are read using pandas read_sql().
The tables are also combined using pandas merge().

## Design Decisions

Requests and BeautifulSoup are used for web scraping.
Pandas is used for data processing and SQL result analysis.
SQLite is used because it is lightweight and does not require
an external database server.