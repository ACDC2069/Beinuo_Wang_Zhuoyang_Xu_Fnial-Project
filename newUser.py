from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, desc

# 1. 初始化：强制使用 local[*] 本地模式，抢回标准输入流（键盘）控制权！
spark = SparkSession.builder \
    .appName("MovieLens_NewUser_Interactive") \
    .master("local[*]") \
    .getOrCreate()

# 静默日志
spark.sparkContext.setLogLevel("ERROR")

print("Loading enriched movie data... (This might take a few seconds)")
movies_enriched_df = spark.read.csv("/user/hive/enriched_csv", header=True, inferSchema=True)

def start_new_user_session(df):
    print("\n" + "="*70)
    print("MOVIELENS RECOMMENDER SYSTEM / 电影推荐系统")
    print("   [New User Preference Profiling / 新用户偏好采集]")
    print("="*70)

    # 真实的交互输入
    print("\n[1/4] What is your favorite movie genre? (e.g., Action, Sci-Fi, Romance)")
    pref_genre = input(">> 您最喜欢的电影流派是？: ").strip()

    print("\n[2/4] Name a movie you really loved (enter keyword, e.g., Matrix, Toy Story)")
    pref_movie = input(">> 您喜欢哪部具体的电影？(输入关键字): ").strip()
    
    print("\n[3/4] What era of movies do you prefer? / 您偏好的电影年代？")
    print("      [1] Modern / 现代电影 (2000年后/After 2000)")
    print("      [2] Classic / 经典老片 (2000年前/Before 2000)")
    pref_era = input(">> Your choice / 您的选择 (1 or 2): ").strip()
    
    print("\n[4/4] What vibe do you prefer? / 您偏好的受众类型？")
    print("      [1] Blockbusters / 热门大片 (评价人数极多/High popularity)")
    print("      [2] Hidden Gems / 冷门佳作 (高分但小众/Niche but highly rated)")
    pref_vibe = input(">> Your choice / 您的选择 (1 or 2): ").strip()

    print("\n Analyzing preferences and generating recommendations... / 正在分析偏好并生成推荐...\n")

    # 核心过滤逻辑
    query = df.filter(col("Number_of_ratings").isNotNull())
    query = query.filter(
        (col("Genre").contains(pref_genre)) | 
        (lower(col("Movie_name")).contains(pref_movie.lower()))
    )

    if pref_era == '1':
        query = query.filter(col("Year_of_release") >= 2000)
    elif pref_era == '2':
        query = query.filter(col("Year_of_release") < 2000)

    if pref_vibe == '1':
        final = query.orderBy(desc("Number_of_ratings"), desc("Rating_average"))
    else:
        final = query.filter(col("Number_of_ratings") < 1000).orderBy(desc("Rating_average"))

    # 输出结果
    if final.count() == 0:
        print("No exact match found. / 未找到完全匹配的电影。")
        print(" Recommending global all-time favorites instead: / 为您推荐全局高分榜单：")
        df.filter(col("Number_of_ratings") > 1000).orderBy(desc("Rating_average")).limit(5).show(truncate=False)
    else:
        print("Here are your Top 5 Personalized Recommendations / 为您精心挑选的 5 部电影 ✨")
        final.select("Movie_name", "Year_of_release", "Genre", "Rating_average").limit(5).show(truncate=False)

try:
    start_new_user_session(movies_enriched_df)
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    spark.stop()