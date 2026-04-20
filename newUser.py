from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, desc, regexp_extract, split, count, avg, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import CountVectorizer

# 1. Initialize SparkSession (Local mode to reclaim input stream)
spark = SparkSession.builder \
    .appName("MovieLens_NewUser_Scientific") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Loading raw data and building Vector Space Model... (Please wait)")

# 2. Load Raw Data
movies_df = spark.read.csv("/user/data/movies.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("/user/data/ratings.csv", header=True, inferSchema=True)

# 3. Base Feature Engineering
# Extract Year from title
movies_df = movies_df.withColumn("year", regexp_extract(col("title"), r"\((\d{4})\)", 1).cast("int"))
# Split genres into an Array (required for Vectorization)
movies_df = movies_df.withColumn("genre_array", split(col("genres"), "\\|"))

# Calculate Aggregated Ratings
stats_df = ratings_df.groupBy("movieId").agg(
    count("rating").alias("num_ratings"),
    avg("rating").alias("avg_rating")
)
df = movies_df.join(stats_df, "movieId", "left")

# 4. Vectorization (Convert genre text array to Sparse Vectors)
cv = CountVectorizer(inputCol="genre_array", outputCol="features", vocabSize=50, minDF=1.0)
cv_model = cv.fit(df)
df = cv_model.transform(df)

def run_scientific_recommender():
    print("\n" + "="*70)
    print("MOVIELENS RECOMMENDER SYSTEM")
    print("   [New User Preference Profiling - Content Based Model]")
    print("="*70)
 # User Input Collection
    pref_genre = input("\n[1/4] Favorite movie genre? (e.g., Action, Sci-Fi): ").strip()
    pref_movie = input("[2/4] Name a movie you loved (keyword, e.g., Matrix): ").strip()

    print("\n[3/4] What era of movies do you prefer?")
    print("      [1] Modern (After 2000)")
    print("      [2] Classic (Before 2000)")
    pref_era = input(">> Choice (1 or 2): ").strip()

    print("\n[4/4] What vibe do you prefer?")
    print("      [1] Blockbusters (High popularity)")
    print("      [2] Hidden Gems (Niche but highly rated)")
    pref_vibe = input(">> Choice (1 or 2): ").strip()

    print("\nAnalyzing preferences and calculating cosine similarities...\n")

    # Step 1: Identify Anchor Movie
    anchor_row = df.filter(lower(col("title")).contains(pref_movie.lower())).first()

    if anchor_row is None:
        print("[Warning] Anchor movie not found. Falling back to standard filtering.")
        res = df.filter(col("genres").contains(pref_genre))
    else:
        anchor_title = anchor_row['title']
        anchor_vector = anchor_row['features']
        print(f"[System] Anchor movie identified: {anchor_title}")

        # Define Cosine Similarity UDF capturing the anchor_vector
        @udf(returnType=DoubleType())
        def cosine_sim(v):
            if v is None or anchor_vector is None:
                return 0.0
            dot_product = float(v.dot(anchor_vector))
            norm_v = float(v.norm(2))
            norm_anchor = float(anchor_vector.norm(2))
            if norm_v == 0 or norm_anchor == 0:
                return 0.0
            return dot_product / (norm_v * norm_anchor)

        # Step 2: Recall Phase (Hard Filtering)
        # Exclude the anchor movie itself!
        res = df.filter(col("title") != anchor_title)

        # Base quality filter (prevent garbage data)
        res = res.filter(col("num_ratings") > 10)

        # Era Filter
        if pref_era == '1':
            res = res.filter(col("year") >= 2000)
        elif pref_era == '2':
            res = res.filter(col("year") < 2000)

        # Vibe Filter
        if pref_vibe == '1':
            res = res.filter(col("num_ratings") > 1000)
        elif pref_vibe == '2':
            res = res.filter(col("num_ratings") <= 1000)

        # Step 3: Ranking Phase (Soft Scoring)
        res = res.withColumn("similarity_score", cosine_sim(col("features")))

        # Primary sort: Cosine Similarity. Secondary sort: Rating Average (Tie-breaker)
        res = res.orderBy(desc("similarity_score"), desc("avg_rating"))

    # Step 4: Display Output
    if res.count() == 0:
        print("No matches found for your strict criteria.")
    else:
        print("Here are your Top 5 Personalized Recommendations:")
        if anchor_row is not None:
            # Show similarity score if anchor was used
            res.select("title", "genres", "year", "similarity_score", "avg_rating").limit(5).show(truncate=False)
        else:
            res.select("title", "genres", "year", "avg_rating").limit(5).show(truncate=False)


try:
    run_scientific_recommender()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    spark.stop()
                                                            
