library(geosphere)  # For distance calculations
library(dplyr)
library(readxl)

# Read the data
df <- read_excel("clinics.xls")

# Convert latitude and longitude to numeric
df$locLat <- as.numeric(df$locLat)
df$locLong <- as.numeric(df$locLong)

# Filter valid latitude and longitude values
df <- df %>%
  filter(locLong >= -180 & locLong <= 180, locLat >= -90 & locLat <= 90) %>%
  filter(!is.na(locLong) & !is.na(locLat))

# Haversine function
haversine <- function(lat1, lon1, lat2, lon2) {
  distHaversine(matrix(c(lon1, lat1), ncol = 2), matrix(c(lon2, lat2), ncol = 2)) * 0.000621371
}

# Measure time for For-loop implementation
start_time <- Sys.time()
haversine_looping <- function(df) {
  distances <- numeric(nrow(df))
  for (i in 1:nrow(df)) {
    distances[i] <- haversine(40.671, -73.985, df$locLat[i], df$locLong[i])
  }
  return(distances)
}
df$distance_loop <- haversine_looping(df)
loop_time <- Sys.time() - start_time

# Measure time for Apply function
start_time <- Sys.time()
df$distance_apply <- apply(df, 1, function(row) {
  haversine(40.671, -73.985, as.numeric(row["locLat"]), as.numeric(row["locLong"]))
})
apply_time <- Sys.time() - start_time

# Measure time for Vectorized implementation
start_time <- Sys.time()
df <- df %>%
  mutate(distance_vectorized = mapply(haversine, 40.671, -73.985, locLat, locLong))
vectorized_time <- Sys.time() - start_time

# Print execution times
execution_times <- data.frame(
  Method = c("For Loop", "Apply", "Vectorized"),
  Time_Seconds = c(loop_time, apply_time, vectorized_time)
)

print(execution_times)
