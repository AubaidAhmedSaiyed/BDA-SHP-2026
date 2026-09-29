import os
import time
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark import StorageLevel


def timed(label, fn):
    """Run fn(), print and return the time taken in seconds."""
    start = time.time()
    result = fn()
    secs = time.time() - start
    print(f"{label:45}: {secs:.4f} sec")
    return secs


def avg_ratings(d):
    # collect() is the action that forces the full calculation
    return d.groupBy("movieId").agg(F.avg("rating").alias("avg_rating")).collect()


def main():
    print("========================================")
    print("1. CREATE SPARK SESSION")
    print("========================================")
    spark = (
        SparkSession.builder
        .appName("CachingAndPartitioning")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark version:", spark.version)
    print("Available cores:", spark.sparkContext.defaultParallelism, "\n")

    print("========================================")
    print("2. LOAD RATINGS DATASET")
    print("========================================")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, "movie_ratings.csv")
    df = spark.read.csv(data_path, header=True, inferSchema=True)
    df.show(5)
    print("Total Rows   :", df.count())
    print("Total Columns:", len(df.columns), "\n")

    print("========================================")
    print("3. SCHEMA")
    print("========================================")
    df.printSchema()

    print("========================================")
    print("4. CURRENT NUMBER OF PARTITIONS")
    print("========================================")
    print("Partitions:", df.rdd.getNumPartitions(), "\n")

    print("========================================")
    print("5. AVERAGE MOVIE RATINGS")
    print("========================================")
    avg_df = df.groupBy("movieId").agg(F.round(F.avg("rating"), 2).alias("Avg_Rating"))
    avg_df.orderBy(F.desc("Avg_Rating")).show(5)

    print("========================================")
    print("6. EXECUTION TIME BEFORE OPTIMIZATION")
    print("========================================")
    avg_ratings(df)  # warm-up run (JVM start-up), not counted
    t_before = timed("Average ratings (no cache)", lambda: avg_ratings(df))

    print("\n========================================")
    print("7. APPLY cache()")
    print("========================================")
    df.cache()
    print("Is cached:", df.is_cached)

    print("\n========================================")
    print("8. TRIGGER CACHING USING count()")
    print("========================================")
    t_cache_build = timed("count() to build cache", lambda: df.count())

    print("\n========================================")
    print("9. RECALCULATE AVERAGE RATINGS")
    print("========================================")
    avg_df.orderBy(F.desc("Avg_Rating")).show(5)

    print("========================================")
    print("10. EXECUTION TIME AFTER CACHING")
    print("========================================")
    t_after = timed("Average ratings (cached)", lambda: avg_ratings(df))

    print("\n========================================")
    print("11. REPARTITION THE DATASET")
    print("========================================")
    df_rep = df.repartition(8)
    df_rep.cache()
    df_rep.count()  # materialise so the timing below is only the calculation

    print("========================================")
    print("12. UPDATED PARTITION COUNT")
    print("========================================")
    print("Before repartition:", df.rdd.getNumPartitions())
    print("After repartition :", df_rep.rdd.getNumPartitions(), "\n")

    print("========================================")
    print("13. COMPARE PERFORMANCE")
    print("========================================")
    t_rep = timed("Average ratings (cached + repartitioned)", lambda: avg_ratings(df_rep))

    print("\n========================================")
    print("14. PERFORMANCE REPORT")
    print("========================================")
    print(f"{'Stage':40}{'Partitions':>12}{'Time (sec)':>14}")
    print("-" * 66)
    print(f"{'Before optimization (no cache)':40}{'-':>12}{t_before:>14.4f}")
    print(f"{'After cache()':40}{df.rdd.getNumPartitions():>12}{t_after:>14.4f}")
    print(f"{'After cache + repartition(8)':40}{df_rep.rdd.getNumPartitions():>12}{t_rep:>14.4f}")
    if t_after > 0:
        print(f"\nSpeed-up from caching: {t_before / t_after:.2f}x")
    print(f"Time saved by caching : {t_before - t_after:.4f} sec\n")

    df_rep.unpersist()
    df.unpersist()

    print("========================================")
    print("SUPPLEMENTARY 1: cache() vs persist(DISK_ONLY)")
    print("========================================")
    d1 = spark.read.csv(data_path, header=True, inferSchema=True)
    d1.cache()
    d1.count()
    t_mem = timed("cache() (memory)", lambda: avg_ratings(d1))
    d1.unpersist()

    d2 = spark.read.csv(data_path, header=True, inferSchema=True)
    d2.persist(StorageLevel.DISK_ONLY)
    d2.count()
    t_disk = timed("persist(DISK_ONLY)", lambda: avg_ratings(d2))
    d2.unpersist()
    print()

    print("========================================")
    print("SUPPLEMENTARY 2: repartition into 2, 4, 8")
    print("========================================")
    base = spark.read.csv(data_path, header=True, inferSchema=True)
    results = {}
    for n in [2, 4, 8]:
        dn = base.repartition(n)
        dn.cache()
        dn.count()
        print(f"Partitions = {dn.rdd.getNumPartitions()}")
        results[n] = timed(f"  Average ratings with {n} partitions", lambda: avg_ratings(dn))
        dn.unpersist()
    print()

    print("========================================")
    print("SUPPLEMENTARY 3: multiple actions, cached vs not cached")
    print("========================================")
    actions = {
        "count()": lambda d: d.count(),
        "average rating per movie": lambda d: avg_ratings(d),
        "top 10 most rated movies": lambda d: d.groupBy("movieId").count()
                                                 .orderBy(F.desc("count")).limit(10).collect(),
        "distinct users": lambda d: d.select("userId").distinct().count(),
    }
    plain = spark.read.csv(data_path, header=True, inferSchema=True)
    print("Without cache:")
    for name, fn in actions.items():
        timed(f"  {name}", lambda fn=fn: fn(plain))

    cached = spark.read.csv(data_path, header=True, inferSchema=True).cache()
    cached.count()
    print("With cache:")
    for name, fn in actions.items():
        timed(f"  {name}", lambda fn=fn: fn(cached))
    cached.unpersist()
    print()

    print("========================================")
    print("SUPPLEMENTARY 4: coalesce() vs repartition()")
    print("========================================")
    start_parts = base.rdd.getNumPartitions()
    print("Original partitions:", start_parts, "(coalesce cannot increase, so we start from 8)")

    # coalesce can only reduce partitions, so first start from 8
    start8 = base.repartition(8)
    co = start8.coalesce(2)
    co.cache()
    co.count()
    print("coalesce(2) partitions    :", co.rdd.getNumPartitions())
    timed("  Average ratings after coalesce(2)", lambda: avg_ratings(co))
    print("  Rows per partition:", co.rdd.glom().map(len).collect())
    co.unpersist()

    rp = base.repartition(2)
    rp.cache()
    rp.count()
    print("repartition(2) partitions :", rp.rdd.getNumPartitions())
    timed("  Average ratings after repartition(2)", lambda: avg_ratings(rp))
    print("  Rows per partition:", rp.rdd.glom().map(len).collect())
    rp.unpersist()

    print("\nPlan check (look for 'Exchange' = shuffle):")
    print("coalesce plan:")
    start8.coalesce(2).explain()
    print("repartition plan:")
    base.repartition(2).explain()
    print()

    print("========================================")
    print("SUPPLEMENTARY 5: larger dataset and scalability")
    print("========================================")
    big = base
    for _ in range(4):        # 2^4 = 16 copies, about 1.7 million rows
        big = big.union(big)
    print("Large dataset rows:", big.count())

    t_big_plain = timed("Large: no cache", lambda: avg_ratings(big))
    big.cache()
    timed("Large: count() to build cache", lambda: big.count())
    t_big_cached = timed("Large: cached", lambda: avg_ratings(big))
    big_rep = big.repartition(8)
    big_rep.cache()
    big_rep.count()
    t_big_rep = timed("Large: cached + repartition(8)", lambda: avg_ratings(big_rep))

    print(f"\nCaching speed-up on large data: {t_big_plain / t_big_cached:.2f}x")
    big_rep.unpersist()
    big.unpersist()

    spark.stop()
    print("\nSparkSession stopped successfully.")


if __name__ == "__main__":
    main()