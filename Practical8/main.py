import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.sql.window import Window

DATA_PATH = "/Volumes/workspace/default/trends/stock_market.csv"

# ---------- 1. SPARK SESSION ----------
print("=" * 60); print("1. SPARK SESSION"); print("=" * 60)
print("Spark version:", spark.version)

# ---------- 2. LOAD DATASET ----------
print("=" * 60); print("2. LOAD STOCK DATASET"); print("=" * 60)
raw = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
print("Total raw records:", raw.count())

# Keep only the three fields we need: Timestamp, Stock, Price
# (Close has some non-numeric text, so try_cast turns bad values into null)
# (Date has two formats: M/d/yyyy and MM-dd-yy, so parse each one separately)
d_slash = F.to_date(F.col("Date"), "M/d/yyyy")
d_dash = F.to_date(F.col("Date"), "MM-dd-yy")
parsed = F.when(F.col("Date").contains("/"), d_slash).otherwise(d_dash)
# 2-digit years like 97 are read as 2097, so move them back 100 years
parsed = F.when(F.year(parsed) > 2021, F.add_months(parsed, -1200)).otherwise(parsed)

stock = (raw.select(parsed.alias("Timestamp"),
                    F.col("Index").alias("Stock"),
                    F.expr("try_cast(Close AS DOUBLE)").alias("Price"))
            .filter(F.col("Price").isNotNull() & (F.col("Price") > 0))
            .cache())
print("Clean records:", stock.count())

# ---------- 3. DISPLAY SCHEMA AND RECORDS ----------
print("=" * 60); print("3. SCHEMA AND STOCK RECORDS"); print("=" * 60)
stock.printSchema()
print("Columns:", stock.columns)
print("Null prices:", stock.filter(F.col("Price").isNull()).count())
print("Distinct stocks:", stock.select("Stock").distinct().count())
display(stock.orderBy("Stock", "Timestamp").limit(10))

# ---------- 4. GROUP BY STOCK SYMBOL ----------
print("=" * 60); print("4. GROUP BY STOCK SYMBOL"); print("=" * 60)
grouped = stock.groupBy("Stock")
print("Grouped by Stock:", grouped)
display(grouped.count().withColumnRenamed("count", "Records").orderBy("Stock"))

# ---------- 5. AVERAGE STOCK PRICE ----------
print("=" * 60); print("5. AVERAGE STOCK PRICE"); print("=" * 60)
avg_df = grouped.agg(F.round(F.avg("Price"), 2).alias("Avg_Price"))
display(avg_df.orderBy(F.desc("Avg_Price")))

# ---------- 6. MAXIMUM STOCK PRICE ----------
print("=" * 60); print("6. MAXIMUM STOCK PRICE"); print("=" * 60)
max_df = grouped.agg(F.round(F.max("Price"), 2).alias("Max_Price"))
display(max_df.orderBy(F.desc("Max_Price")))

# ---------- 7. MINIMUM STOCK PRICE ----------
print("=" * 60); print("7. MINIMUM STOCK PRICE"); print("=" * 60)
min_df = grouped.agg(F.round(F.min("Price"), 2).alias("Min_Price"))
display(min_df.orderBy("Min_Price"))

# ---------- 8. STOCK SUMMARY REPORT ----------
print("=" * 60); print("8. STOCK SUMMARY REPORT"); print("=" * 60)
summary = grouped.agg(
    F.count("Price").alias("Records"),
    F.round(F.avg("Price"), 2).alias("Avg_Price"),
    F.round(F.max("Price"), 2).alias("Max_Price"),
    F.round(F.min("Price"), 2).alias("Min_Price"),
    F.min("Timestamp").alias("From"),
    F.max("Timestamp").alias("To"))
summary = summary.withColumn("Price_Range", F.round(F.col("Max_Price") - F.col("Min_Price"), 2))
display(summary.orderBy("Stock"))

# ---------- 9. BEST-PERFORMING STOCKS ----------
print("=" * 60); print("9. BEST-PERFORMING STOCKS (LAST 5 YEARS)"); print("=" * 60)
START = "2016-06-01"   # common period: every stock has data from this date
perf = (stock.filter(F.col("Timestamp") >= START)
        .groupBy("Stock")
        .agg(F.min_by("Price", "Timestamp").alias("Start_Price"),
             F.max_by("Price", "Timestamp").alias("End_Price")))
perf = (perf.withColumn("Return_%", F.round((F.col("End_Price") - F.col("Start_Price")) * 100
                                            / F.col("Start_Price"), 2))
            .orderBy(F.desc("Return_%")))
display(perf)
best = perf.first()
worst = perf.orderBy("Return_%").first()
print(f"Best-performing stock: {best['Stock']} with {best['Return_%']}% return")

# ---------- 10. MARKET TRENDS ----------
print("=" * 60); print("10. MARKET TRENDS (LAST 1 YEAR)"); print("=" * 60)
YEAR_START = "2020-06-01"
trend = (stock.filter(F.col("Timestamp") >= YEAR_START)
         .groupBy("Stock")
         .agg(F.min_by("Price", "Timestamp").alias("Start_Price"),
              F.max_by("Price", "Timestamp").alias("End_Price")))
trend = (trend.withColumn("Change_%", F.round((F.col("End_Price") - F.col("Start_Price")) * 100
                                              / F.col("Start_Price"), 2))
              .withColumn("Trend", F.when(F.col("Change_%") > 0, "Uptrend").otherwise("Downtrend"))
              .orderBy(F.desc("Change_%")))
display(trend)
up = trend.filter(F.col("Trend") == "Uptrend").count()
print(f"{up} of {trend.count()} markets are in an uptrend.")

# ---------- 11. INVESTMENT RECOMMENDATIONS ----------
print("=" * 60); print("11. INVESTMENT RECOMMENDATIONS"); print("=" * 60)
print(f"1. Prefer {best['Stock']}, the best performer with {best['Return_%']}% return in 5 years.")
print(f"2. Be careful with {worst['Stock']}, the weakest performer with {worst['Return_%']}% return.")
print("3. Buy stocks in an uptrend and avoid those in a downtrend.")
print("4. Spread money across different markets to reduce risk.")
print("5. Watch moving averages and volatility regularly before buying or selling.")

# ================= SUPPLEMENTARY =================
# S1. Daily returns
print("=" * 60); print("S1. DAILY STOCK RETURNS"); print("=" * 60)
w = Window.partitionBy("Stock").orderBy("Timestamp")
returns = (stock.withColumn("Prev_Price", F.lag("Price").over(w))
           .filter(F.col("Prev_Price").isNotNull())
           .withColumn("Daily_Return_%",
                       F.round((F.col("Price") - F.col("Prev_Price")) * 100 / F.col("Prev_Price"), 3)))
display(returns.filter(F.col("Stock") == "NSEI").orderBy(F.desc("Timestamp")).limit(10))

# S2. Top-performing stock over a period (calendar year 2020)
print("=" * 60); print("S2. TOP-PERFORMING STOCK IN 2020"); print("=" * 60)
p2020 = (stock.filter(F.col("Timestamp").between("2020-01-01", "2020-12-31"))
         .groupBy("Stock")
         .agg(F.min_by("Price", "Timestamp").alias("Start_Price"),
              F.max_by("Price", "Timestamp").alias("End_Price"))
         .withColumn("Return_%", F.round((F.col("End_Price") - F.col("Start_Price")) * 100
                                         / F.col("Start_Price"), 2))
         .orderBy(F.desc("Return_%")))
display(p2020)
top2020 = p2020.first()
print(f"Top performer in 2020: {top2020['Stock']} with {top2020['Return_%']}% return")

# S3. Moving average (7-day and 30-day)
print("=" * 60); print("S3. MOVING AVERAGE"); print("=" * 60)
w7 = Window.partitionBy("Stock").orderBy("Timestamp").rowsBetween(-6, 0)
w30 = Window.partitionBy("Stock").orderBy("Timestamp").rowsBetween(-29, 0)
ma = (stock.withColumn("MA_7", F.round(F.avg("Price").over(w7), 2))
      .withColumn("MA_30", F.round(F.avg("Price").over(w30), 2)))
display(ma.filter(F.col("Stock") == "NSEI").orderBy(F.desc("Timestamp")).limit(10))

# S4. Volatility comparison (std deviation of daily returns, last 5 years)
print("=" * 60); print("S4. VOLATILITY COMPARISON"); print("=" * 60)
vol = (returns.filter(F.col("Timestamp") >= START)
       .groupBy("Stock")
       .agg(F.round(F.stddev("Daily_Return_%"), 3).alias("Volatility_%"))
       .orderBy(F.desc("Volatility_%")))
display(vol)
print(f"Most volatile: {vol.first()['Stock']}")

# S5. Visualization
print("=" * 60); print("S5. PRICE TREND VISUALIZATION"); print("=" * 60)
pdf = (ma.filter((F.col("Stock") == "NSEI") & (F.col("Timestamp") >= "2020-01-01"))
       .orderBy("Timestamp").toPandas())
plt.figure(figsize=(9, 5))
plt.plot(pdf["Timestamp"], pdf["Price"], label="Close price")
plt.plot(pdf["Timestamp"], pdf["MA_30"], label="30-day moving average")
plt.xlabel("Date"); plt.ylabel("Price")
plt.title("NSEI Price Trend (2020 onwards)")
plt.legend(); plt.show()

rp = perf.toPandas()
plt.figure(figsize=(8, 5))
plt.barh(rp["Stock"][::-1], rp["Return_%"][::-1])
plt.xlabel("Return in last 5 years (%)"); plt.ylabel("Stock")
plt.title("Stock Performance Comparison")
plt.show()

print("=" * 60); print("EXPERIMENT COMPLETED"); print("=" * 60)