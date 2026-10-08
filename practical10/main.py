import time
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer, OneHotEncoder
from pyspark.ml.clustering import KMeans, BisectingKMeans
from pyspark.ml.functions import vector_to_array
from sklearn.cluster import KMeans as SkKMeans
from sklearn.preprocessing import StandardScaler as SkScaler

N = 1_000_000          # number of customers (use 100_000 if the notebook is slow)
DATA_PATH = "/Volumes/workspace/default/trends/retail_customers_csv"
FEATURES = ["total_spend", "purchase_count", "avg_basket_size", "days_since_last_purchase"]

# ---------- one-time: generate the simulated customer dataset ----------
# segment: (share, purchase_count, avg_basket, days_since_last, category probabilities)
segments = {
    0: (0.15, 40, 150, 10, [0.6, 0.2, 0.1, 0.1]),     # loyal high-value
    1: (0.25, 35, 40, 15, [0.1, 0.2, 0.5, 0.2]),      # frequent small-basket
    2: (0.15, 6, 300, 45, [0.3, 0.5, 0.1, 0.1]),      # big-basket occasional
    3: (0.30, 3, 50, 25, [0.1, 0.1, 0.2, 0.6]),       # new / low activity
    4: (0.15, 15, 90, 200, [0.25, 0.25, 0.25, 0.25]), # lapsed / at risk
}
cats = ["Electronics", "Home", "Fashion", "Grocery"]

def pick(r, values_by_seg):
    """Build a CASE WHEN on the hidden segment id."""
    e = None
    for s, v in values_by_seg.items():
        e = F.when(F.col("seg") == s, v) if e is None else e.when(F.col("seg") == s, v)
    return e

def category_expr():
    e = None
    for s, (_, _, _, _, p) in segments.items():
        c1, c2, c3 = p[0], p[0] + p[1], p[0] + p[1] + p[2]
        inner = (F.when(F.col("r") < c1, cats[0]).when(F.col("r") < c2, cats[1])
                 .when(F.col("r") < c3, cats[2]).otherwise(cats[3]))
        e = F.when(F.col("seg") == s, inner) if e is None else e.when(F.col("seg") == s, inner)
    return e

cum, edges = 0, []
for s, v in segments.items():
    cum += v[0]; edges.append((s, cum))
seg_expr = None
for s, edge in edges:
    seg_expr = (F.when(F.col("u") < edge, s) if seg_expr is None
                else seg_expr.when(F.col("u") < edge, s))

gen = (spark.range(1, N + 1).withColumnRenamed("id", "customer_id")
       .withColumn("u", F.rand(42)).withColumn("seg", seg_expr)
       .withColumn("r", F.rand(7))
       .withColumn("purchase_count",
                   F.greatest(F.lit(1), F.round(pick("seg", {s: F.lit(v[1]) for s, v in segments.items()})
                                                * (1 + 0.25 * F.randn(1))).cast("int")))
       .withColumn("avg_basket_size",
                   F.round(F.greatest(F.lit(5.0), pick("seg", {s: F.lit(float(v[2])) for s, v in segments.items()})
                                      * (1 + 0.2 * F.randn(2))), 2))
       .withColumn("total_spend", F.round(F.col("purchase_count") * F.col("avg_basket_size")
                                          * (0.9 + 0.2 * F.rand(3)), 2))
       .withColumn("days_since_last_purchase",
                   F.greatest(F.lit(1), F.round(pick("seg", {s: F.lit(float(v[3])) for s, v in segments.items()})
                                                * (1 + 0.3 * F.randn(4))).cast("int")))
       .withColumn("category_preference", category_expr())
       .select("customer_id", "total_spend", "purchase_count", "avg_basket_size",
               "days_since_last_purchase", "category_preference"))
gen.write.mode("overwrite").option("header", True).csv(DATA_PATH)

# ---------- 1. LOAD AND PREPROCESS ----------
print("=" * 60); print("1. LOAD AND PREPROCESS DATASET"); print("=" * 60)
print("Spark version:", spark.version)
raw = spark.read.option("header", True).option("inferSchema", True).csv(DATA_PATH)
print("Total raw records:", raw.count())
raw.printSchema()
print("Null values per column:")
display(raw.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in raw.columns]))
df = (raw.dropna().dropDuplicates(["customer_id"])
      .filter((F.col("total_spend") > 0) & (F.col("purchase_count") > 0)))
print("Clean records:", df.count())
display(df.limit(10))
display(df.select(FEATURES).describe())

# ---------- 2. FEATURE EXTRACTION ----------
print("=" * 60); print("2. FEATURE EXTRACTION"); print("=" * 60)
assembler = VectorAssembler(inputCols=FEATURES, outputCol="features")
assembled = assembler.transform(df)
display(assembled.select("customer_id", "features").limit(5))

# ---------- 3. FEATURE SCALING ----------
print("=" * 60); print("3. FEATURE SCALING (StandardScaler)"); print("=" * 60)
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withMean=True, withStd=True)
scaled = scaler.fit(assembled).transform(assembled)
display(scaled.select("features", "scaledFeatures").limit(5))

# ---------- 4. SPARK MLLIB K-MEANS (k = 5) ----------
print("=" * 60); print("4. K-MEANS CLUSTERING, k = 5"); print("=" * 60)
def wssse(model, data):
    """Within Set Sum of Squared Errors: sum of squared distance of each point to its centre."""
    centers = F.array(*[F.array(*[F.lit(float(x)) for x in c]) for c in model.clusterCenters()])
    d = (model.transform(data)
         .withColumn("v", vector_to_array("scaledFeatures"))
         .withColumn("c", centers[F.col("prediction")]))
    sq = sum((F.col("v")[i] - F.col("c")[i]) ** 2 for i in range(len(FEATURES)))
    return d.agg(F.sum(sq)).first()[0]

kmeans = KMeans(k=5, seed=42, maxIter=20, featuresCol="scaledFeatures", predictionCol="prediction")
t0 = time.time()
model = kmeans.fit(scaled)
print(f"Training time: {time.time() - t0:.2f} sec")
pred = model.transform(scaled)
display(pred.groupBy("prediction").count().orderBy("prediction"))

# ---------- 5. EVALUATION: WSSSE ----------
print("=" * 60); print("5. WSSSE"); print("=" * 60)
cost5 = wssse(model, scaled)
print(f"WSSSE for k=5: {cost5:,.2f}")

# ---------- 6. VISUALIZE CLUSTER CENTERS AND INTERPRET ----------
print("=" * 60); print("6. CLUSTER CENTERS AND SEGMENTS"); print("=" * 60)
stats = df.agg(*[F.avg(c).alias(c + "_m") for c in FEATURES],
               *[F.stddev(c).alias(c + "_s") for c in FEATURES]).first()
rows = []
for i, c in enumerate(model.clusterCenters()):
    row = {"Cluster": i}
    for j, f in enumerate(FEATURES):
        row[f] = round(c[j] * stats[f + "_s"] + stats[f + "_m"], 2)   # back to original units
    rows.append(row)
centers_pdf = pd.DataFrame(rows).set_index("Cluster")
sizes = pred.groupBy("prediction").count().toPandas().set_index("prediction")["count"]
centers_pdf["Customers"] = sizes
centers_pdf["Share_%"] = (centers_pdf["Customers"] * 100 / centers_pdf["Customers"].sum()).round(1)

# label each cluster from its centre values
labels, left = {}, set(centers_pdf.index)
def take(col, name, largest=True):
    pick_i = (centers_pdf.loc[list(left), col].idxmax() if largest
              else centers_pdf.loc[list(left), col].idxmin())
    labels[pick_i] = name; left.discard(pick_i)
take("days_since_last_purchase", "Lapsed / at-risk customers")
take("total_spend", "High-value loyal customers")
take("purchase_count", "Frequent small-basket buyers")
take("avg_basket_size", "Big-basket occasional buyers")
for i in left: labels[i] = "New / low-activity customers"
centers_pdf["Segment"] = pd.Series(labels)
display(spark.createDataFrame(centers_pdf.reset_index()))

scaled_centers = pd.DataFrame(model.clusterCenters(), columns=FEATURES)
scaled_centers.index = [f"C{i}: {labels[i]}" for i in scaled_centers.index]
scaled_centers.plot(kind="bar", figsize=(10, 5))
plt.ylabel("Scaled centre value"); plt.title("Cluster Centers (standardized features)")
plt.xticks(rotation=20, ha="right"); plt.tight_layout(); plt.show()

sample = pred.select("purchase_count", "avg_basket_size", "prediction").sample(0.01, seed=1).toPandas()
plt.figure(figsize=(7, 5))
plt.scatter(sample["purchase_count"], sample["avg_basket_size"], c=sample["prediction"], s=6, cmap="tab10")
plt.xlabel("Purchase count"); plt.ylabel("Average basket size"); plt.title("Customer Segments")
plt.show()

# ---------- 7. SINGLE-NODE SKLEARN vs SPARK PARALLEL K-MEANS ----------
print("=" * 60); print("7. PERFORMANCE: SKLEARN vs SPARK"); print("=" * 60)
full_pdf = df.select(FEATURES).toPandas()
results = []
for n in [N // 10, N // 2, N]:
    part = full_pdf.iloc[:n]
    t0 = time.time()
    x = SkScaler().fit_transform(part)
    sk = SkKMeans(n_clusters=5, n_init=1, max_iter=20, random_state=42).fit(x)
    t_sk = time.time() - t0

    sub = scaled.limit(n)
    t0 = time.time()
    sp = KMeans(k=5, seed=42, maxIter=20, featuresCol="scaledFeatures").fit(sub)
    t_sp = time.time() - t0
    results.append((n, round(t_sk, 2), round(t_sp, 2), round(sk.inertia_, 1)))
perf = pd.DataFrame(results, columns=["Records", "sklearn_sec", "spark_sec", "sklearn_WSSSE"])
display(spark.createDataFrame(perf))
perf.set_index("Records")[["sklearn_sec", "spark_sec"]].plot(kind="bar", figsize=(7, 4))
plt.ylabel("Time (sec)"); plt.title("Single-node sklearn vs Spark K-Means"); plt.show()

# ================= SUPPLEMENTARY =================
# S1. k = 3 and k = 7, elbow curve
print("=" * 60); print("S1. WSSSE FOR DIFFERENT k AND ELBOW CURVE"); print("=" * 60)
ks, costs = list(range(2, 9)), []
for k in ks:
    m = KMeans(k=k, seed=42, maxIter=20, featuresCol="scaledFeatures").fit(scaled)
    costs.append(wssse(m, scaled))
elbow = pd.DataFrame({"k": ks, "WSSSE": [round(c, 2) for c in costs]})
display(spark.createDataFrame(elbow))
print(f"k=3: {costs[1]:,.2f}   k=5: {costs[3]:,.2f}   k=7: {costs[5]:,.2f}")
plt.figure(figsize=(7, 4))
plt.plot(ks, costs, marker="o")
plt.xlabel("Number of clusters (k)"); plt.ylabel("WSSSE"); plt.title("Elbow Curve")
plt.xticks(ks); plt.grid(True); plt.show()

# S2. Bisecting K-Means vs K-Means
print("=" * 60); print("S2. BISECTING K-MEANS"); print("=" * 60)
bkm = BisectingKMeans(k=5, seed=42, maxIter=20, featuresCol="scaledFeatures",
                      predictionCol="bk_pred").fit(scaled)
bk_pred = bkm.transform(pred)
bcenters = F.array(*[F.array(*[F.lit(float(x)) for x in c]) for c in bkm.clusterCenters()])
bd = (bk_pred.withColumn("v", vector_to_array("scaledFeatures"))
      .withColumn("c", bcenters[F.col("bk_pred")]))
bsq = sum((F.col("v")[i] - F.col("c")[i]) ** 2 for i in range(len(FEATURES)))
print(f"K-Means WSSSE: {cost5:,.2f}")
print(f"Bisecting K-Means WSSSE: {bd.agg(F.sum(bsq)).first()[0]:,.2f}")
display(bk_pred.groupBy("bk_pred").count().orderBy("bk_pred"))
print("Overlap between the two methods (rows = K-Means, columns = Bisecting):")
display(bk_pred.crosstab("prediction", "bk_pred"))

# S3. Add product category preference
print("=" * 60); print("S3. ADD PRODUCT CATEGORY PREFERENCE"); print("=" * 60)
idx = StringIndexer(inputCol="category_preference", outputCol="cat_idx").fit(df)
enc = OneHotEncoder(inputCols=["cat_idx"], outputCols=["cat_vec"]).fit(idx.transform(df))
d3 = enc.transform(idx.transform(df))
a3 = VectorAssembler(inputCols=FEATURES, outputCol="num_features").transform(d3)
s3 = StandardScaler(inputCol="num_features", outputCol="num_scaled", withMean=True, withStd=True)
a3 = s3.fit(a3).transform(a3)
a3 = VectorAssembler(inputCols=["num_scaled", "cat_vec"], outputCol="scaledFeatures").transform(a3)
m3 = KMeans(k=5, seed=42, maxIter=20, featuresCol="scaledFeatures").fit(a3)
p3 = m3.transform(a3)
print("Cluster composition by category preference (rows = cluster):")
display(p3.crosstab("prediction", "category_preference").orderBy("prediction_category_preference"))

print("=" * 60); print("EXPERIMENT COMPLETED"); print("=" * 60)