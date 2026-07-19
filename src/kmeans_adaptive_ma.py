import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# === Step 1: Preprocessing ===
def preprocess_web_traffic_data_clean(train_path):
    logging.info(f"Loading data from: {train_path}")
    train = pd.read_csv(train_path)
    logging.info(f"Loaded shape: {train.shape}")
    time_series_cols = train.columns[1:]
    X = train[time_series_cols].astype(float).values

    # Per-series standardization
    row_means = X.mean(axis=1, keepdims=True)
    row_stds = X.std(axis=1, keepdims=True)
    row_stds[row_stds == 0] = 1  # Avoid divide-by-zero
    X_scaled = (X - row_means) / row_stds

    logging.info(f"Standardization complete. Shape: {X_scaled.shape}")
    return train['Page'].values, X_scaled, row_means, row_stds, train[time_series_cols]

# === Step 2: KMeans Clustering on Summary Stats ===
def cluster_series_kmeans_subsampled(X_scaled, original_df, n_clusters=12, seed=42, sample_size=10000):
    logging.info("Computing summary statistics for clustering...")
    mean_vals = original_df.mean(axis=1).values
    std_vals = original_df.std(axis=1).values
    max_vals = original_df.max(axis=1).values
    min_vals = original_df.min(axis=1).values
    recent_avg = original_df.iloc[:, -30:].mean(axis=1).values
    spike_count = (original_df > (mean_vals[:, None] + 2 * std_vals[:, None])).sum(axis=1).values
    features = np.stack([mean_vals, std_vals, max_vals, min_vals, recent_avg, spike_count], axis=1)

    scaler_summary = StandardScaler()
    features_scaled = scaler_summary.fit_transform(features)

    sampled_idx = np.random.RandomState(seed).choice(
        features_scaled.shape[0], size=min(sample_size, features_scaled.shape[0]), replace=False
    )

    logging.info(f"Applying KMeans on {len(sampled_idx)} samples...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
    cluster_labels_sampled = kmeans.fit_predict(features_scaled[sampled_idx])

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(features_scaled[sampled_idx], cluster_labels_sampled)
    cluster_labels = knn.predict(features_scaled)
    logging.info("KMeans clustering completed.")
    return cluster_labels

# === Step 3: Adaptive Stepwise MA Forecast ===
def forecast_stepwise_ma(X_scaled, cluster_labels, original_df, train_ratio=0.9):
    n_samples, n_days = X_scaled.shape
    train_days = int(train_ratio * n_days)
    forecast_horizon = n_days - train_days
    actuals = np.zeros((n_samples, forecast_horizon))
    forecasts = np.zeros((n_samples, forecast_horizon))

    for cid in np.unique(cluster_labels):
        indices = np.where(cluster_labels == cid)[0]

        # Use raw values (not standardized) to compute meaningful volatility
        cluster_std = np.std(original_df.values[indices])
        window = int(np.clip(cluster_std * 0.1, 3, 30))

        logging.info(f"Cluster {cid}: {len(indices)} series, Adaptive MA window = {window} (raw std = {cluster_std:.4f})")

        for i in indices:
            full_series = X_scaled[i]
            train_series = full_series[:train_days]
            test_series = full_series[train_days:]
            forecast_series = []

            for step in range(forecast_horizon):
                start_idx = max(0, train_days - window + step)
                end_idx = train_days + step
                history = full_series[start_idx:end_idx]
                forecast_val = np.mean(history)
                forecast_series.append(forecast_val)

            forecasts[i, :] = forecast_series
            actuals[i, :] = test_series

    logging.info(f"Forecast horizon: {forecast_horizon} days")
    return actuals, forecasts

# === Step 4: Evaluation ===
def evaluate_forecast(actuals, forecasts, cluster_labels):
    mse = mean_squared_error(actuals.flatten(), forecasts.flatten())
    mae = mean_absolute_error(actuals.flatten(), forecasts.flatten())
    rmse = np.sqrt(mse)
    logging.info(f"[Overall] MSE: {mse:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    for cid in np.unique(cluster_labels):
        idx = np.where(cluster_labels == cid)[0]
        mse_c = mean_squared_error(actuals[idx].flatten(), forecasts[idx].flatten())
        mae_c = mean_absolute_error(actuals[idx].flatten(), forecasts[idx].flatten())
        rmse_c = np.sqrt(mse_c)
        logging.info(f"[Cluster {cid}] MSE: {mse_c:.4f} | MAE: {mae_c:.4f} | RMSE: {rmse_c:.4f}")

# === Step 5: Inverse Scaling ===
def unscale_forecasts(forecasts, row_means, row_stds):
    return forecasts * row_stds[:, -1:] + row_means[:, -1:]

# === Step 6: Visualization ===
def visualize_cluster_forecasts(X_scaled, cluster_labels, actuals, forecasts, page_names, num_per_cluster=2):
    cluster_ids = np.unique(cluster_labels)
    for cid in cluster_ids:
        indices = np.where(cluster_labels == cid)[0][:num_per_cluster]
        for idx in indices:
            full_series = X_scaled[idx]
            train_len = len(full_series) - forecasts.shape[1]
            plt.figure(figsize=(12, 6))
            plt.plot(range(train_len), full_series[:train_len], label="Train", color="blue")
            plt.plot(range(train_len, train_len + forecasts.shape[1]), actuals[idx], label="Actual", color="green")
            plt.plot(range(train_len, train_len + forecasts.shape[1]), forecasts[idx], label="Forecast", color="red", linestyle="--")
            plt.title(f"Cluster {cid} | Page {page_names[idx]}")
            plt.xlabel("Time")
            plt.ylabel("Standardized Traffic")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

# === Step 7: Visualize Unscaled Forecasts (Optional) ===
def visualize_multiple_forecasts(original_df, unscaled_forecasts, pages, num_series=3):
    forecast_horizon = unscaled_forecasts.shape[1]
    actual_test_part = original_df.iloc[:, -forecast_horizon:].reset_index(drop=True)
    for i in range(min(num_series, len(pages))):
        plt.figure(figsize=(12, 6))
        plt.plot(actual_test_part.iloc[i], label='Actual', color='blue')
        plt.plot(unscaled_forecasts[i], label='Forecast', color='red', linestyle='--')
        plt.title(f'Forecast vs Actual (Original Scale): {pages[i]}')
        plt.xlabel('Time')
        plt.ylabel('Traffic')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# === Step 8: Full Pipeline ===
def run_pipeline_with_stepwise_ma(train_path, n_clusters=12):
    pages, X_scaled, row_means, row_stds, original_df = preprocess_web_traffic_data_clean(train_path)
    cluster_labels = cluster_series_kmeans_subsampled(X_scaled, original_df, n_clusters=n_clusters)
    actuals, forecasts = forecast_stepwise_ma(X_scaled, cluster_labels, original_df)
    evaluate_forecast(actuals, forecasts, cluster_labels)
    visualize_cluster_forecasts(X_scaled, cluster_labels, actuals, forecasts, pages, num_per_cluster=2)
    forecasts_unscaled = unscale_forecasts(forecasts, row_means, row_stds)
    visualize_multiple_forecasts(original_df, forecasts_unscaled, pages, num_series=3)

# === Main Entry Point ===
if __name__ == "__main__":
    train_csv_path = "Data/preprocessed_train_2.csv"
    run_pipeline_with_stepwise_ma(train_csv_path, n_clusters=12)
