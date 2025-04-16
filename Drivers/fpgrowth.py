import pandas as pd
from collections import defaultdict
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.fpm import FPGrowth
import os


def process_data(data):
    # Map feature names to feature IDs (unique integer)
    feature_name_id = defaultdict(int)
    next_node_id = 0
    # Map data point ID to list of connected feature IDs and value
    data_points = defaultdict(list)

    for feature in data.columns[:-1]:
        if feature not in feature_name_id:
            feature_name_id[feature] = next_node_id
            next_node_id += 1

    for row_id, row in data.iterrows():
        features = [feature_name_id[f]
                    for f in data.columns[:-1] if row[f] == 1]
        # Assuming class result is in the last column
        class_label = int(row.iloc[-1])

        data_points[row_id] = (features, class_label)

    feature_id_name = {v: k for k, v in feature_name_id.items()}
    print("Data processing complete. Information saved to JSON files.")
    return data_points, feature_id_name


def generate_rules(frequent_patterns, feature_names, total_count):
    rules_rdd = frequent_patterns.rdd.map(lambda row: {
        'rule': " & ".join([feature_names[f] for f in row.items]),
        'support': row.freq / total_count,
        'absolute_support': row.freq
    })
    return rules_rdd.collect()


def fp_growth(data_df, min_support, class_label):
    count_data_points = data_df.filter(
        col("class_label") == class_label).count()
    data_df = data_df.where(col("class_label") == class_label)

    fp_growth = FPGrowth(minSupport=min_support, itemsCol="features")
    model = fp_growth.fit(data_df)
    frequent_itemsets = model.freqItemsets.sort(
        "items").select("items", "freq")

    return frequent_itemsets, count_data_points


def main(data, pos_min_support,class_label, path):

    start_time = time.time()

    # Ensure output directories exist
    # os.makedirs(f"Dataset/{}", exist_ok=True)

    spark = SparkSession.builder.appName("FPM_Processing").getOrCreate()
    try:
        data_points, feature_id_name = process_data(data)

        # Making Spark DataFrame of Data
        data = [(value[0], value[1]) for value in data_points.values()]
        data_df = spark.createDataFrame(data, ["features", "class_label"])

        # Generating Patterns
        frequent_patterns, count = fp_growth(data_df, pos_min_support, class_label)
        print("Frequent Itemsets:")
        frequent_patterns.show(truncate=False)

        rules = generate_rules(frequent_patterns, feature_id_name, count)

        rules_df = pd.DataFrame({
            "rule": [r['rule'] for r in rules],
            "normalized_support": [r['support'] for r in rules]
        })
        p = "pos" if class_label==1 else "neg"
        rules_df.to_csv(path+p+"_rules.csv", index=False)
        print("Rules generated and saved to csv files.")
    except Exception as e:
        print(f"Error: {e}")

    finally:
        spark.stop()

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")
