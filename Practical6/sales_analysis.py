import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


def main():

    spark = (
        SparkSession.builder
        .appName("ECommerceSalesAnalysis")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    print("========================================")
    print("1. LOAD SALES DATA")
    print("========================================")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "sales_data.csv")
    df = spark.read.csv(data_path, header=True, inferSchema=True)

    print("Sample Records from Dataset:")
    df.show(5, truncate=False)

    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"Total Rows: {total_rows}")
    print(f"Total Columns: {total_cols}")
    print(f"Column Names: {df.columns}\n")

    print("========================================")
    print("2. SCHEMA")
    print("========================================")
    df.printSchema()
    print()

    print("========================================")
    print("3. DATA TYPE VALIDATION")
    print("========================================")
    print("Actual Column Data Types:")
    for col_name, dtype in df.dtypes:
        print(f" - {col_name:25}: {dtype}")
    print()

    print("Checking for Null / Missing Values per Column:")
    null_exprs = [
        F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(c)
        for c in df.columns
    ]
    df.select(null_exprs).show()

    distinct_rows = df.dropDuplicates().count()
    duplicate_rows = total_rows - distinct_rows
    print(f"Total Records     : {total_rows}")
    print(f"Distinct Records  : {distinct_rows}")
    print(f"Duplicate Records : {duplicate_rows}")

    invalid_qty_price = df.filter((F.col("Quantity_Sold") <= 0) | (F.col("Unit_Price") <= 0)).count()
    print(f"Records with non-positive Quantity or Price: {invalid_qty_price}\n")

    print("========================================")
    print("4. REVENUE CALCULATION")
    print("========================================")
    df = df.withColumn(
        "Revenue",
        F.round(F.col("Quantity_Sold") * F.col("Unit_Price"), 2)
    )

    print("Sample Records with Calculated Revenue:")
    df.select("Product_ID", "Quantity_Sold", "Unit_Price", "Revenue").show(5)

    print("========================================")
    print("5. CATEGORY-WISE REVENUE")
    print("========================================")
    category_rev = (
        df.groupBy("Product_Category")
        .agg(F.round(F.sum("Revenue"), 2).cast("decimal(18,2)").alias("Total_Revenue"))
    )
    category_rev.show()

    print("========================================")
    print("6. TOP-SELLING PRODUCTS")
    print("========================================")
    top_products = (
        df.groupBy("Product_ID")
        .agg(F.sum("Quantity_Sold").alias("Total_Quantity_Sold"))
        .orderBy(F.desc("Total_Quantity_Sold"))
    )
    print("Top 10 Selling Products by Quantity Sold:")
    top_products.show(10)

    print("========================================")
    print("7. CITY-WISE REVENUE")
    print("========================================")
    city_rev = (
        df.groupBy("Region")
        .agg(F.round(F.sum("Revenue"), 2).cast("decimal(18,2)").alias("Total_Revenue"))
        .orderBy(F.desc("Total_Revenue"))
    )
    print("Total Revenue by City / Region (Descending):")
    city_rev.show()

    print("========================================")
    print("8. SORTED CATEGORY REVENUE")
    print("========================================")
    sorted_category_rev = category_rev.orderBy(F.desc("Total_Revenue"))
    print("Product Categories from Highest to Lowest Revenue:")
    sorted_category_rev.show()

    print("========================================")
    print("9. ANALYTICAL SUMMARY")
    print("========================================")
    agg_summary = df.agg(
        F.count("*").alias("Total_Transactions"),
        F.sum("Quantity_Sold").alias("Total_Quantity_Sold"),
        F.sum("Revenue").alias("Total_Revenue"),
        F.avg("Revenue").alias("Avg_Revenue")
    ).first()

    total_transactions = agg_summary["Total_Transactions"]
    total_quantity_sold = agg_summary["Total_Quantity_Sold"]
    total_revenue = float(agg_summary["Total_Revenue"])
    avg_revenue = float(agg_summary["Avg_Revenue"])

    num_categories = df.select("Product_Category").distinct().count()
    num_cities = df.select("Region").distinct().count()

    highest_cat_row = sorted_category_rev.first()
    highest_rev_category = highest_cat_row["Product_Category"]
    highest_rev_cat_amount = float(highest_cat_row["Total_Revenue"])

    lowest_cat_row = category_rev.orderBy(F.asc("Total_Revenue")).first()
    lowest_rev_category = lowest_cat_row["Product_Category"]
    lowest_rev_cat_amount = float(lowest_cat_row["Total_Revenue"])

    top_product_row = top_products.first()
    top_product_id = top_product_row["Product_ID"]
    top_product_qty = top_product_row["Total_Quantity_Sold"]

    highest_city_row = city_rev.first()
    highest_rev_city = highest_city_row["Region"]
    highest_rev_city_amount = float(highest_city_row["Total_Revenue"])

    print(f"Total Transactions        : {total_transactions:,}")
    print(f"Total Quantity Sold       : {total_quantity_sold:,}")
    print(f"Total Revenue             : ${total_revenue:,.2f}")
    print(f"Average Revenue / Record  : ${avg_revenue:,.2f}")
    print(f"Number of Categories      : {num_categories}")
    print(f"Number of Cities / Regions: {num_cities}")
    print(f"Highest-Revenue Category  : {highest_rev_category} (${highest_rev_cat_amount:,.2f})")
    print(f"Lowest-Revenue Category   : {lowest_rev_category} (${lowest_rev_cat_amount:,.2f})")
    print(f"Top-Selling Product ID    : {top_product_id} ({top_product_qty:,} units sold)")
    print(f"Highest-Revenue City      : {highest_rev_city} (${highest_rev_city_amount:,.2f})\n")

    print("========================================")
    print("10. BUSINESS INSIGHTS & RECOMMENDATIONS")
    print("========================================")
    highest_cat_share = (highest_rev_cat_amount / total_revenue) * 100
    lowest_cat_share = (lowest_rev_cat_amount / total_revenue) * 100
    highest_city_share = (highest_rev_city_amount / total_revenue) * 100

    print("BUSINESS INSIGHTS:")
    print(f" - Highest Revenue Category : {highest_rev_category} generating ${highest_rev_cat_amount:,.2f} ({highest_cat_share:.2f}% of total sales).")
    print(f" - Top-Selling Product      : Product #{top_product_id} with {top_product_qty:,} units sold, indicating high customer demand.")
    print(f" - Highest Revenue City     : {highest_rev_city} region leading sales with ${highest_rev_city_amount:,.2f} ({highest_city_share:.2f}% share).")
    print(f" - Lowest-Performing Cat.   : {lowest_rev_category} generating ${lowest_rev_cat_amount:,.2f} ({lowest_cat_share:.2f}% share).")
    print()
    print("STRATEGIC RECOMMENDATIONS:")
    print(f" 1. Inventory Planning   : Ensure robust safety stock and high replenishment priority for '{highest_rev_category}' and top product #{top_product_id} to prevent stockouts.")
    print(f" 2. Targeted Marketing   : Allocate higher ad spend and localized digital campaigns to '{highest_rev_city}' to maximize ROI from proven high-converting markets.")
    print(f" 3. Bundle Promotions    : Create cross-category bundle offers pairing high-demand '{highest_rev_category}' goods with slower-moving '{lowest_rev_category}' items.")
    print(f" 4. Revitalize Low Tier  : Conduct pricing sensitivity analysis and promotional flash sales for '{lowest_rev_category}' to expand market share and customer trial.\n")

    print("========================================")
    print("SUPPLEMENTARY ANALYSIS")
    print("========================================")

    print("--- S.1 Top 3 Revenue-Generating Cities ---")
    city_rev.show(3)

    print("--- S.2 Average Revenue Per Order ---")
    print("Note: Each dataset row represents an individual transaction record.")
    avg_order_df = df.agg(F.round(F.avg("Revenue"), 2).alias("Avg_Revenue_Per_Order"))
    avg_order_df.show()

    print("--- S.3 Discount, Discount Amount, and Final Revenue ---")
    df = df.withColumn("DiscountAmount", F.round(F.col("Revenue") * F.col("Discount"), 2))
    df = df.withColumn("FinalRevenue", F.round(F.col("Revenue") - F.col("DiscountAmount"), 2))

    print("Sample Records with Discount and Final Net Revenue:")
    df.select("Product_ID", "Revenue", "Discount", "DiscountAmount", "FinalRevenue").show(5)

    print("Category-wise Final Net Revenue after Discount:")
    cat_final_rev = (
        df.groupBy("Product_Category")
        .agg(
            F.round(F.sum("Revenue"), 2).cast("decimal(18,2)").alias("Gross_Revenue"),
            F.round(F.sum("DiscountAmount"), 2).cast("decimal(18,2)").alias("Total_Discounts"),
            F.round(F.sum("FinalRevenue"), 2).cast("decimal(18,2)").alias("Final_Net_Revenue")
        )
        .orderBy(F.desc("Final_Net_Revenue"))
    )
    cat_final_rev.show()

    print("--- S.4 Least-Performing Product Category ---")
    least_perf_category = category_rev.orderBy(F.asc("Total_Revenue")).limit(1)
    least_perf_category.show()

    print("--- S.5 Monthly Revenue Summary ---")
    if "Sale_Date" in df.columns:
        print("Date column 'Sale_Date' detected. Extracting Year/Month for chronological analysis:")
        monthly_df = (
            df.withColumn("Parsed_Date", F.to_date(F.col("Sale_Date"), "yyyy-MM-dd"))
              .withColumn("Year", F.year(F.col("Parsed_Date")))
              .withColumn("Month", F.month(F.col("Parsed_Date")))
              .withColumn("YearMonth", F.date_format(F.col("Parsed_Date"), "yyyy-MM"))
              .groupBy("YearMonth")
              .agg(
                  F.count("*").alias("Total_Orders"),
                  F.round(F.sum("Revenue"), 2).cast("decimal(18,2)").alias("Monthly_Gross_Revenue"),
                  F.round(F.sum("FinalRevenue"), 2).cast("decimal(18,2)").alias("Monthly_Net_Revenue")
              )
              .orderBy("YearMonth")
        )
        monthly_df.show(20)
    else:
        print("Existing dataset does not contain a date column. Skipping monthly breakdown.")

    print("========================================")
    print("SPARK SQL")
    print("========================================")
    df.createOrReplaceTempView("sales")
    print("Temporary SQL View 'sales' registered.")

    print("\nRunning SQL Query: Category-wise Revenue")
    spark_sql_category = spark.sql("""
        SELECT 
            Product_Category,
            CAST(ROUND(SUM(Revenue), 2) AS DECIMAL(18,2)) AS TotalRevenue
        FROM sales
        GROUP BY Product_Category
        ORDER BY TotalRevenue DESC
    """)
    spark_sql_category.show()

    print("Running SQL Query: Top 5 High-Value Transactions")
    spark.sql("""
        SELECT 
            Product_ID,
            Product_Category,
            Region,
            Quantity_Sold,
            Unit_Price,
            Revenue,
            FinalRevenue
        FROM sales
        ORDER BY Revenue DESC
        LIMIT 5
    """).show()

    print("========================================")
    print("TRANSFORMATIONS & ACTIONS DEMONSTRATION")
    print("========================================")
    print("1. Transformations (e.g., select, filter, withColumn, groupBy, orderBy):")
    print("   - Lazily evaluated: Spark creates a Logical Plan / DAG without reading all data immediately.")

    lazy_df = df.filter(F.col("Revenue") > 5000).select("Product_ID", "Product_Category", "Revenue")
    print("   - lazy_df created with .filter() and .select(). Execution Plan:")
    lazy_df.explain()

    print("\n2. Actions (e.g., show, count, collect, first):")
    print("   - Triggers the actual DAG execution across worker nodes and returns results.")
    result_count = lazy_df.count()
    print(f"   - Action executed: lazy_df.count() returned {result_count} records with Revenue > $5,000.\n")

    spark.stop()
    print("SparkSession stopped successfully.")


if __name__ == "__main__":
    main()