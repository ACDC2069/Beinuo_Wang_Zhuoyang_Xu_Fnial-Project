from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql.functions import col, desc, explode
import random

# 1. Initialize SparkSession with Heavy Memory Configuration
spark = SparkSession.builder \
    .appName("MovieLens_ExistingUser_Final") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "10") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("Loading data for Collaborative Filtering... (Please wait)")

# 2. Load Raw Data
movies_df = spark.read.csv("/user/data/movies.csv", header=True, inferSchema=True)
ratings_df = spark.read.csv("/user/data/ratings.csv", header=True, inferSchema=True)

# 3. ALS Model Training
print("Training ALS model (Matrix Factorization)...")
als = ALS(
    maxIter=10, 
    regParam=0.1, 
    userCol="userId", 
    itemCol="movieId", 
    ratingCol="rating",
    coldStartStrategy="drop"
)
model = als.fit(ratings_df)

def run_existing_user_analysis():
    print("\n" + "="*70)
    print("MOVIELENS RECOMMENDER SYSTEM")
    print("   [Existing User Recommendation - ALS Model]")
    print("="*70)

    # 4. Randomly select an existing user
    all_users = ratings_df.select("userId").distinct().collect()
    random_user_id = random.choice(all_users)["userId"]
    print(f"[System] Randomly selected User ID: {random_user_id}")

    # 5. Get movies already rated by the user
    user_history = ratings_df.filter(col("userId") == random_user_id)
    print(f"[System] Found {user_history.count()} movies already rated by this user.")

    # 6. Generate raw recommendations
    target_user_df = spark.createDataFrame([(random_user_id,)], ["userId"])
    
    raw_recommendations = model.recommendForUserSubset(target_user_df, 20)

    recs_exploded = raw_recommendations.select(
        col("userId"), 
        explode("recommendations").alias("rec")
    ).select(
        "userId", 
        col("rec.movieId").alias("movieId"), 
        col("rec.rating").alias("predicted_rating")
    )

    # 7. Exclude rated movies via Left Anti Join
    final_recommendations = recs_exploded.join(
        user_history, 
        on="movieId", 
        how="left_anti"
    )

    # 8. Join with titles and display final list
    final_result = final_recommendations.join(movies_df, "movieId") \
        .orderBy(desc("predicted_rating")) \
        .select("movieId", "title", "genres")

    print(f"\nTop 5 Personalized Recommendations for User {random_user_id}")
    print("(Excluding movies previously rated by the user)")
    final_result.limit(5).show(truncate=False)

try:
    run_existing_user_analysis()
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    spark.stop()