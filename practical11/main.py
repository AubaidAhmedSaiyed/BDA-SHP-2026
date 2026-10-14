import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.sql import functions as F
from pyspark.ml.feature import Tokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array

N = 500_000  # number of SMS messages (use 100_000 if the notebook is slow)
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.sms_spam")  # creates the volume once
BASE = "/Volumes/workspace/default/sms_spam"
DATA_PATH = BASE + "/sms_spam_large_csv"
TRAIN_PATH = BASE + "/sms_train_features"
VAL_PATH = BASE + "/sms_val_features"
TEST_PATH = BASE + "/sms_test_features"

# ---------- one-time: generate the simulated SMS dataset ----------
spam_words = ["free", "win", "winner", "prize", "claim", "urgent", "cash", "offer", "txt", "call",
              "now", "reply", "stop", "congratulations", "bonus", "credit", "loan", "click", "link",
              "limited", "guaranteed", "mobile", "ringtone", "voucher", "won", "award", "subscribe",
              "dating", "text", "selected", "pounds", "customer", "service", "latest", "colour"]
ham_words = ["ok", "see", "later", "home", "dinner", "tomorrow", "meeting", "class", "thanks", "sorry",
             "love", "come", "going", "lunch", "today", "friend", "night", "good", "morning", "work",
             "bus", "time", "phone", "wait", "tell", "know", "need", "want", "mom", "office",
             "weekend", "movie", "coming", "back", "sure"]
common_words = ["the", "a", "to", "is", "and", "you", "i", "your", "for", "on", "in", "it", "me",
                "my", "are", "be", "we", "will", "at", "this"]

def word_from(words):
    arr = F.array(*[F.lit(w) for w in words])
    return F.element_at(arr, (F.rand() * len(words)).cast("int") + 1)

def random_word(is_spam):
    """Own-class word 40%, common/filler word 40%, other-class word 20%."""
    r = F.rand()
    filler = F.concat(F.lit("w"), (F.rand() * 300).cast("int").cast("string"))
    common = F.when(F.rand() < 0.5, word_from(common_words)).otherwise(filler)
    own = F.when(is_spam, word_from(spam_words)).otherwise(word_from(ham_words))
    other = F.when(is_spam, word_from(ham_words)).otherwise(word_from(spam_words))
    return F.when(r < 0.4, own).when(r < 0.8, common).otherwise(other)

gen = spark.range(N).withColumn("is_spam", F.rand(42) < 0.13)  # ~13% spam, like the UCI data
for i in range(12):
    gen = gen.withColumn(f"w{i}", random_word(F.col("is_spam")))
gen = (gen.withColumn("label", F.when(F.col("is_spam"), "spam").otherwise("ham"))
          .withColumn("message_text", F.concat_ws(" ", *[F.col(f"w{i}") for i in range(12)]))
          .select("label", "message_text"))
gen.write.mode("overwrite").option("header", True).csv(DATA_PATH)

# ---------- 1. LOAD THE DATASET ----------
print("=" * 60); print("1. LOAD THE SMS DATASET"); print("=" * 60)
print("Spark version:", spark.version)
raw = spark.read.option("header", True).csv(DATA_PATH)
print("Total raw records:", raw.count())
raw.printSchema()
print("Null values per column:")
display(raw.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in raw.columns]))
df = (raw.dropna()
         .filter(F.length("message_text") > 0)
         .withColumn("label_idx", F.when(F.col("label") == "spam", 1.0).otherwise(0.0)))
print("Clean records:", df.count())
display(df.limit(10))
print("Class distribution:")
display(df.groupBy("label").count())

# ---------- 2. TEXT PREPROCESSING: TOKENIZE, STOP WORDS, TF-IDF ----------
print("=" * 60); print("2. TEXT PREPROCESSING AND TF-IDF"); print("=" * 60)
train_raw, val_raw, test_raw = df.randomSplit([0.7, 0.15, 0.15], seed=42)

tokenizer = Tokenizer(inputCol="message_text", outputCol="tokens")
remover = StopWordsRemover(inputCol="tokens", outputCol="clean_tokens")
cv = CountVectorizer(inputCol="clean_tokens", outputCol="tf", vocabSize=1000, minDF=5)
idf = IDF(inputCol="tf", outputCol="tfidf")

def text_steps(d):
    return remover.transform(tokenizer.transform(d))

train_tok = text_steps(train_raw)
cv_model = cv.fit(train_tok)                                   # vocabulary learned on train only
idf_model = idf.fit(cv_model.transform(train_tok))             # IDF learned on train only

def featurize(d):
    return idf_model.transform(cv_model.transform(text_steps(d))).select("label_idx", "tfidf")

display(text_steps(train_raw).select("message_text", "tokens", "clean_tokens").limit(5))
display(featurize(train_raw).limit(5))

# save the features once so every later step reads them quickly (works as a cache)
featurize(train_raw).write.mode("overwrite").parquet(TRAIN_PATH)
featurize(val_raw).write.mode("overwrite").parquet(VAL_PATH)
featurize(test_raw).write.mode("overwrite").parquet(TEST_PATH)
train = spark.read.parquet(TRAIN_PATH)
val = spark.read.parquet(VAL_PATH)
test = spark.read.parquet(TEST_PATH)
print("Train / validation / test rows:", train.count(), val.count(), test.count())
print("Feature vector size (vocabulary):", train.first()["tfidf"].size)

# ---------- helper: classification metrics for the spam class ----------
def metrics(pred_df, pred_col="prediction"):
    rows = pred_df.groupBy("label_idx", pred_col).count().collect()
    d = {(r["label_idx"], r[pred_col]): r["count"] for r in rows}
    tp, fp = d.get((1.0, 1.0), 0), d.get((0.0, 1.0), 0)
    fn, tn = d.get((1.0, 0.0), 0), d.get((0.0, 0.0), 0)
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return dict(TP=tp, FP=fp, FN=fn, TN=tn, Accuracy=acc, Precision=prec, Recall=rec, F1=f1)

auc_eval = BinaryClassificationEvaluator(labelCol="label_idx", rawPredictionCol="rawPrediction",
                                         metricName="areaUnderROC")

# ---------- 3. LOGISTIC REGRESSION WITH SPARK MLLIB ----------
print("=" * 60); print("3. LOGISTIC REGRESSION (Spark MLlib)"); print("=" * 60)
lr = LogisticRegression(featuresCol="tfidf", labelCol="label_idx", maxIter=20,
                        regParam=0.01, elasticNetParam=0.0)
t0 = time.time()
lr_model = lr.fit(train)
print(f"Training time: {time.time() - t0:.2f} sec")
print("Number of weights:", len(lr_model.coefficients), " Intercept:", round(lr_model.intercept, 4))

# ---------- 4. HYPERPARAMETER TUNING (regParam, maxIter, elasticNetParam) ----------
print("=" * 60); print("4. HYPERPARAMETER TUNING"); print("=" * 60)
grid_rows = []
for reg in [0.001, 0.01, 0.1]:
    for en in [0.0, 0.5, 1.0]:
        for it in [10, 30]:
            m = LogisticRegression(featuresCol="tfidf", labelCol="label_idx", maxIter=it,
                                   regParam=reg, elasticNetParam=en).fit(train)
            p = m.transform(val)
            mt = metrics(p)
            grid_rows.append((reg, en, it, round(auc_eval.evaluate(p), 4),
                              round(mt["Accuracy"], 4), round(mt["Recall"], 4)))
grid = pd.DataFrame(grid_rows, columns=["regParam", "elasticNetParam", "maxIter",
                                        "val_AUC", "val_Accuracy", "val_Recall"])
display(spark.createDataFrame(grid))
best = grid.sort_values("val_AUC", ascending=False).iloc[0]
print("Best on validation set:\n", best)

# ---------- 5. EVALUATION ON THE TEST SET ----------
print("=" * 60); print("5. EVALUATION"); print("=" * 60)
best_lr = LogisticRegression(featuresCol="tfidf", labelCol="label_idx",
                             maxIter=int(best["maxIter"]), regParam=float(best["regParam"]),
                             elasticNetParam=float(best["elasticNetParam"]))
final_model = best_lr.fit(train)
test_pred = final_model.transform(test)
mt = metrics(test_pred)
mt["ROC_AUC"] = auc_eval.evaluate(test_pred)
print("Confusion matrix (spam = positive class):")
print(f"            Pred ham   Pred spam")
print(f"Actual ham  {mt['TN']:>9}  {mt['FP']:>9}")
print(f"Actual spam {mt['FN']:>9}  {mt['TP']:>9}")
for k in ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]:
    print(f"{k:<10}: {mt[k]:.4f}")

# 5b. effect of the classification threshold (for Key Question 5)
print("-" * 60); print("5b. THRESHOLD vs PRECISION / RECALL"); print("-" * 60)
with_p = test_pred.withColumn("p_spam", vector_to_array("probability")[1])
th_rows = []
for t in [0.5, 0.4, 0.3, 0.2, 0.1]:
    pt = with_p.withColumn("pred_t", F.when(F.col("p_spam") >= t, 1.0).otherwise(0.0))
    m_t = metrics(pt, "pred_t")
    th_rows.append((t, m_t["TP"], m_t["FP"], m_t["FN"], round(m_t["Precision"], 4),
                    round(m_t["Recall"], 4), round(m_t["F1"], 4)))
th = pd.DataFrame(th_rows, columns=["threshold", "TP", "FP", "FN", "precision", "recall", "F1"])
display(spark.createDataFrame(th))

# ---------- 6. DATASET SIZE vs TRAINING TIME AND ACCURACY ----------
print("=" * 60); print("6. DATASET SIZE vs TRAINING TIME vs ACCURACY"); print("=" * 60)
size_rows = []
for frac in [0.1, 0.25, 0.5, 1.0]:
    part = train if frac == 1.0 else train.sample(False, frac, seed=1)
    n_rows = part.count()
    t0 = time.time()
    m = LogisticRegression(featuresCol="tfidf", labelCol="label_idx",
                           maxIter=int(best["maxIter"]), regParam=float(best["regParam"]),
                           elasticNetParam=float(best["elasticNetParam"])).fit(part)
    t_fit = time.time() - t0
    p = m.transform(test)
    size_rows.append((n_rows, round(t_fit, 2), round(metrics(p)["Accuracy"], 4),
                      round(auc_eval.evaluate(p), 4)))
size_df = pd.DataFrame(size_rows, columns=["train_rows", "train_sec", "test_accuracy", "test_AUC"])
display(spark.createDataFrame(size_df))
fig, ax1 = plt.subplots(figsize=(7, 4))
ax1.plot(size_df["train_rows"], size_df["train_sec"], marker="o", color="tab:orange")
ax1.set_xlabel("Training rows"); ax1.set_ylabel("Training time (sec)", color="tab:orange")
ax2 = ax1.twinx()
ax2.plot(size_df["train_rows"], size_df["test_accuracy"], marker="s", color="tab:blue")
ax2.set_ylabel("Test accuracy", color="tab:blue")
plt.title("Training time and accuracy vs dataset size"); plt.tight_layout(); plt.show()

# ---------- 7. FULL-BATCH vs MINI-BATCH GRADIENT DESCENT ----------
# Spark MLlib's LogisticRegression uses L-BFGS, which has no mini-batch option. So here the
# gradient descent loop is written by hand: every iteration the workers compute partial
# gradients on their partitions (mapInPandas), and the driver adds them up and updates the weights.
print("=" * 60); print("7. FULL-BATCH vs MINI-BATCH GRADIENT DESCENT"); print("=" * 60)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def train_gd(data, fraction, iters=30, lr_step=0.5, reg=0.001, seed=42):
    d = data.first()["tfidf"].size
    w, b = np.zeros(d), 0.0
    out_schema = "grad_w array<double>, grad_b double, loss double, n long"
    history = []
    for it in range(iters):
        w_now, b_now = w.copy(), float(b)

        def partial_gradient(batches):                      # runs on the workers
            for pdf in batches:
                X = np.vstack(pdf["x"].values)
                y = pdf["label_idx"].values
                p = sigmoid(X @ w_now + b_now)
                err = p - y
                loss = -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).sum()
                yield pd.DataFrame({"grad_w": [(X.T @ err).tolist()], "grad_b": [float(err.sum())],
                                    "loss": [float(loss)], "n": [len(y)]})

        batch = data if fraction >= 1.0 else data.sample(False, fraction, seed + it)
        parts = (batch.select(vector_to_array("tfidf").alias("x"), "label_idx")
                      .mapInPandas(partial_gradient, out_schema).collect())
        n = sum(r["n"] for r in parts)
        grad_w = np.sum([r["grad_w"] for r in parts], axis=0) / n + reg * w   # driver aggregates
        grad_b = sum(r["grad_b"] for r in parts) / n
        w, b = w - lr_step * grad_w, b - lr_step * grad_b                     # driver updates
        history.append(sum(r["loss"] for r in parts) / n)
    return w, b, history

def score(data, w, b):
    def f(batches):
        for pdf in batches:
            X = np.vstack(pdf["x"].values)
            yield pd.DataFrame({"label_idx": pdf["label_idx"].values, "prob": sigmoid(X @ w + b)})
    return (data.select(vector_to_array("tfidf").alias("x"), "label_idx")
                .mapInPandas(f, "label_idx double, prob double"))

gd_rows, curves = [], {}
for name, frac in [("Full-batch (100%)", 1.0), ("Mini-batch (10%)", 0.1), ("Mini-batch (1%)", 0.01)]:
    t0 = time.time()
    w, b, hist = train_gd(train, frac)
    t_gd = time.time() - t0
    s = score(test, w, b)
    acc = s.agg(F.avg(((F.col("prob") >= 0.5).cast("double") == F.col("label_idx")).cast("double"))).first()[0]
    auc = BinaryClassificationEvaluator(labelCol="label_idx", rawPredictionCol="prob",
                                        metricName="areaUnderROC").evaluate(s)
    gd_rows.append((name, round(t_gd, 2), round(hist[-1], 4), round(acc, 4), round(auc, 4)))
    curves[name] = hist
gd_df = pd.DataFrame(gd_rows, columns=["method", "train_sec", "final_batch_loss", "test_accuracy", "test_AUC"])
display(spark.createDataFrame(gd_df))
gd_df.set_index("method")["train_sec"].plot(kind="bar", figsize=(6, 4))
plt.ylabel("Time (sec)"); plt.title("30 iterations: full-batch vs mini-batch GD")
plt.xticks(rotation=15); plt.tight_layout(); plt.show()
plt.figure(figsize=(7, 4))
for name, hist in curves.items():
    plt.plot(hist, label=name)
plt.xlabel("Iteration"); plt.ylabel("Loss on the batch used"); plt.legend()
plt.title("Convergence of gradient descent"); plt.show()

print("=" * 60); print("EXPERIMENT COMPLETED"); print("=" * 60)