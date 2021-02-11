##### Description ##############################################################
# Code for Kaggle's Predict Future Sales Competition
# Started on 2021-01-09
# by Ryan Zidan Is'ad
# Work directory is C:/Penting/R/kaggle1
# 
# Special thanks to Christian Albers for his notebook on Kaggle:
# https://www.kaggle.com/chralbers/feature-eng-and-xgb-from-d-larionov-done-in-r/code
getwd()
setwd("C:/Penting/R/kaggle1")





##### Notes ####################################################################
# kalau 3 bulan terakhir no sales, prediksi 0
# kalau total sales cuma <= 3, prediksi 0
# harga tetap kecuali ada perubahan. jadi pakai nilai sebelumnya
# bikin function untuk lag terus mutate ke data
# perhatiin contoh. bisa pakai dplyr biar lebih sederhana
# test_data perlu mutate cons_month jadi 35
# xgboost gabisa ada NA, perlu prediksi harga barang yang belum rilis
# harga item mungkin mirip sama harga item yang punya item_id deket. contoh harga item_id 3579 mungkin mirip sama 3578 atau 2580
# bikin kombinasi shop_id sama categ_id, biar gampang harga median


##### 0 - Packages #############################################################
# You may need to install them first before loading.
library(xgboost)
library(tidyverse)
library(broom)
library(lubridate)







##### 1 - Functions ############################################################
# These are some useful functions that will help to create lag which will be


##### 2 - Importing data #######################################################
training_data = read.csv("sales_train.csv")      #Daily sales data from 2013-01 to 2015-10
test_data     = read.csv("test.csv")             #Predict sales for 2015-11
items.info    = read.csv("items.csv")            #Information about the items
items.categ   = read.csv("item_categories.csv")  #Information about the items categories
shops.info    = read.csv("shops.csv")            #Information about the shops
submit.sample = read.csv("sample_submission.csv")#Format for the prediction result





##### 3 - Data Preparation #####################################################
# Change column "date" in training_data into "Date" format
training_data$date = as.Date(training_data$date, format = "%d.%m.%Y")
str(training_data)

# Price can't be negative. We'll omit those.
training_data = training_data[!(training_data$item_price == -1),]

# Inserting ID and category ID then turning it into monthly sales
data      = training_data %>%
                # Create two time columns with year.month and cummulative month
                mutate(year = year(date),
                       month = month(date)) %>%
                mutate(cons_month = month + 12*(year - 2013)) %>%
                # Summarizing monthly sales
                group_by(cons_month,
                         shop_id,
                         item_id,
                         item_price) %>%
                summarize(month_total = sum(item_cnt_day)) %>%      
                distinct() %>%
                full_join(test_data)


# Could be[?] possible with mutate function but it takes a long time.
data$cons_month = replace_na(data$cons_month, replace = 1)

# All months from 2013.01 to 2015.10
unique_time = unique(na.omit(data$cons_month))

# data will be transformed to list every month from 2013.1 to 2015.10 to allow 
# regression model with lag. It is necessary since data only counts month where
# sales happened.
data      = data %>%
                group_by(item_id, shop_id) %>%
                complete(cons_month = 1:35) %>%
                left_join(y = test_data, by = c("shop_id" = "shop_id", "item_id" = "item_id")) %>%
                # left_join creates two ID columns of which ID.y is from test_data
                subset(select = -ID.x) %>%
                rename(ID = ID.y)

# NA in column "month_total" indicates no sales. Again, mutate function could probably do this[?]
summary(data$month_total)
data$month_total = ifelse(is.na(data$month_total),
                          0,
                          data$month_total)

# Column "item_price" contains NAs. This code fills them with the appropriate value.
# It is assumed that there is a difference in price between each store.
data      = data %>%
                group_by(item_id, shop_id) %>%
                # Assumes no change in price until change is shown in training_data
                fill(item_price, .direction = "down") %>%
                # Assumes starting price as the release price if the product hasn't been released
                fill(item_price, .direction = "up")
                
# Gets column "item_category_id" from items.info
data      = data %>%
                full_join(items.info) %>%
                subset(select = -item_name)

# training_data contains some item that are not going to be predicted, shown by NA 
# in the "ID" column. Although the items are not what this code tries to predict, 
#it its beneficial in regression/forecast attempt. Therefore. they will
# not be omitted.
summary(data$ID)

# There are still NAs in the column "item_price" from products that have not been
# released until 2015.11, which can't be predicted. While, NAs in column "ID"
# represents item and shop combination that are not part of the prediction goal.
summary(data)

# Tidying up the data.
data = data[, c(3,6, 2, 1, 7, 4, 5)] %>%
          arrange(ID)

# Final result of the data should be ordered by time, ID, shop_id, item_id, 
# item_category_id, item_price, and month_total. NAs should exist only in column
# "ID" and "item_price".
glimpse(data)
summary(data)


# Again, this lines of code look kind of different compared to the last even though its goal is to change NA into a defined value. You might realize some similarities though.
data = data %>%
  mutate(ID = ifelse(is.na(ID),
                     # I transformed each shop and item combination to an ID number that has the format of xxyyyyy, where the first two numbers are from shop_id.
                     -1*shop_id*100000 - item_id,
                     ID))




# Median price of item_id
data = data %>%
  group_by(item_id) %>%
  mutate(item_price = ifelse(is.na(item_price),
                             median(item_price, na.rm = T),
                             item_price)) %>%
  ungroup()

# Median price of item_category_id for each shop
data = data %>%
  group_by(shop_id, item_category_id) %>%
  mutate(item_price = ifelse(is.na(item_price),
                             median(item_price, na.rm = T),
                             item_price)) %>%
  ungroup()

# Median price of item_category_id foverall
data = data %>%
  group_by(item_category_id) %>%
  mutate(item_price = ifelse(is.na(item_price),
                             median(item_price, na.rm = T),
                             item_price)) %>%
  ungroup()

summary(data)



?top_n
a = mtcars %>% slice_max(mpg, n = 5)

best_item = data %>%
                group_by(item_id) %>%
                summarize(items_sold = sum(month_total)) %>%
                slice_max(items_sold, n = 30)

worst_item = data %>%
                group_by(item_id, item_category_id) %>%
                summarize(items_sold = sum(month_total)) %>%
                filter(items_sold < 0)

ggplot(data = best_item, aes(x = reorder(item_id, items_sold), y = items_sold)) +
    geom_bar(stat = "identity") +
    coord_flip() +
    theme_minimal()


##### 4 - Forecast preparation #################################################
# This is a column of the price of the most sold item.
# orang bakal ikut pikir-pikir buat beli barang yang paling laku untuk kategori yang sama
# contoh: mending beli hp harga 2 jt atau beli hp yang 2,3 yang banyak orang beli

# ini masih harga median category, proxy barang substitusi
median_cat    =  data %>%
                      group_by(cons_month, 
                               item_category_id) %>%
                      summarize(med_category_price = median(item_price, na.rm = T))

# This is a column that contains the lowest price for the same item given specific month.
# We assume that anyone prefer paying the lowest price since each shop offers the exact same item.
min_item_price = data %>%
                      group_by(cons_month,
                               item_id) %>%
                      summarize(min_item_price = min(item_price, na.rm = T))

# total penjualan masing-masing toko untuk setiap bulan
shop_revenue   = data %>%
                      group_by(cons_month,
                               shop_id,
                               item_category_id) %>%
                      summarize(revenue = sum(item_price*month_total, na.rm = T)) %>%
                      group_by(cons_month,
                               shop_id) %>%
                      summarize(total_revenue = sum(revenue))

# total penjualan seluruh toko tiap bulan, proxy kondisi ekonomi
month_all_sales = shop_revenue %>%
                      group_by(cons_month) %>%
                      summarize(econ = sum(total_revenue))

# total penjualan masing-masing item untuk bulan tertentu
item_tot_sales = data %>%
                      group_by(cons_month,
                               item_id) %>%
                      summarize(item_revenue = sum(item_price*month_total, na.rm = T))











train_df = a %>%
              filter(cons_month <= 33) %>%
              subset(select = -c(ID, month_total))

label_df = a %>%
              filter(cons_month <= 33) %>%
              subset(select = month_total)

valid_df = a %>%
              filter(cons_month == 34) %>%
              subset(select = -c(ID, month_total))

label_valid_df = a %>%
                    filter(cons_month == 34) %>%
                    subset(select = month_total)

test_df  = a %>%
              filter(cons_month == 35) %>%
              subset(select = -c(ID, month_total))

label_test_df  = a %>%
                    filter(cons_month == 35) %>%
                    subset(select = month_total)



#masih sering error karena NA
xgb_params = list(max_depth = 6, min_child_weight = 20,
                  colsample_bytree = 1,
                  subsample = 1, eta = 0.3)

trainmatrix = xgb.DMatrix(data = as.matrix(train_df), label = as.matrix(label_df))
valmatrix = xgb.DMatrix(data = as.matrix(valid_df), label = as.matrix(label_valid_df)) 
testmatrix = xgb.DMatrix(data = as.matrix(test_df), label = as.matrix(label_test_df))

xgb_model = xgb.train(params = xgb_params, data = trainmatrix, nrounds = 100,
                      watchlist = list(oct15 = valmatrix),
                      early_stopping_rounds = 3)

dfout = a %>%
            filter(cons_month == 35, ID >= 0) %>%
            mutate(ID = as.integer(ID),
                   item_cnt_month = item_cnt_month)





  






##### 5 - Test Code ############################################################
# WARNING! 
# This is Unfinished weekly data. Executing this code will result in a
# data frame of around 80 million rows! Takes ~30 minute to execute with my laptop.
# IT IS 3.3 Gb OF DATA! no wonder it takes a long time to compute
df    = training_data %>%
            mutate(week = week(date),
                   year = year(date)) %>%
            mutate(cons_week = week + 53*(year-2013)) %>%
            group_by(cons_week,
                     item_id,
                     shop_id,
                     item_price) %>%
            summarize(week_total = sum(item_cnt_day)) %>%
            distinct() %>%
            full_join(test_data)

df$cons_week = replace_na(df$cons_week, replace = 1)

unique_week = unique(na.omit(df$cons_week))

df    = df %>%
            group_by(item_id,
                     shop_id) %>%
            complete(cons_week = unique_week) %>%
            left_join(y = test_data, by = c("shop_id" = "shop_id", "item_id" = "item_id")) %>%
  # left_join creates two ID columns of which ID.y is from test_data
            subset(select = -ID.x) %>%
            rename(ID = ID.y)

df$week_total = ifelse(is.na(df$week_total),
                       0,
                       df$week_total)

df    = df %>%
  group_by(shop_id, item_id) %>%
  # Assumes no change in price until changes is shown in training_data
  fill(item_price, .direction = "down") %>%
  # Assumes starting price as the release price if the product hasn't been released
  fill(item_price, .direction = "up")

df    = df %>%
            full_join(items.info) %>%
            subset(select = -item_name)
  







