import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv("railway.csv")
print(df)

df.shape
df.describe()
df.info()

#count by payment method and show on pie chart
var1 = df["Payment Method"].value_counts()

plt.figure(figsize=(4, 4))
plt.pie(var1,shadow=True, labels=var1.index, autopct="%1.1f%%")
plt.title("Payment Method Distribution") 
plt.show()

#find how many person purchase ticket from station and in advance with pie chart
df["Purchase Type"].value_counts()
df_station = df[df["Purchase Type"] == "Station"]
var2=df_station["Ticket Type"].value_counts()
plt.figure(figsize=(4, 4))
plt.pie(var2,shadow=True, labels=var2.index, autopct="%1.1f%%", colors=["#c85d46","#467fc8","#c8466b"],
       explode=(0.1,0,0))
plt.title("Ticket In Advanced Booked From Station") 
plt.show()




#find out how many peoples purchase from online and having Adult RailCard
df_online = df[df["Purchase Type"] == "Online"]
var4 = df_online["Railcard"].value_counts()
plt.figure(figsize=(4, 4))
plt.pie(var4, shadow=True, labels=var4.index, autopct="%1.1f%%", 
        colors=["g", "r", "teal"], explode=(0,0.1,0))
plt.title("Purchase Ticket from Online by Railcard Type") 
plt.show()

plt.figure(figsize=(4, 3))
plt.bar(var4.index, var4.values, color=["#c85d46", "#467fc8", "#c8466b"], width=0.4)
plt.title("Railcard Usage in Online Purchases")
plt.ylabel("Count")
plt.show()

#histogram using price  Frequency column
df.Price
counts, bin_edges, _= plt.hist(df.Price,color="#467fc8", edgecolor="black", alpha=0.7, label="Hist")
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
plt.plot(bin_centers, counts,marker='o', linestyle='-', color="red", markersize=6, label="Price Trend")
plt.xticks(np.arange(0,250,20).tolist())
plt.title("Price Distribution")
plt.xlabel("Price")
plt.ylabel("Frequency")
plt.legend()
plt.show()


#Visualize the count of each "Journey Status" using a simple bar graph.

journey_counts = df["Journey Status"].value_counts()

plt.figure(figsize=(8, 5))
plt.bar(journey_counts.index, journey_counts.values, edgecolor="black", alpha=0.8, width=0.4)

plt.title("Journey Status Count", fontsize=14, color="darkblue")
plt.xlabel("Journey Status", fontsize=12, color="purple")
plt.ylabel("Count", fontsize=12, color="purple")
plt.xticks(fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()


 #Fill missing values in the "Reason for Delay" column using the most frequent reason (mode) and visualize the distribution using a line or scatter plot.

df = df.copy()  
df["Reason for Delay"] = df["Reason for Delay"].fillna(df["Reason for Delay"].mode()[0])
delay_counts = df["Reason for Delay"].value_counts()

plt.figure(figsize=(10, 5))
plt.plot(delay_counts.index, delay_counts.values, marker="o", linestyle="-", color="b", markerfacecolor="red")

plt.title("Counts of Reasons for Delay (After Filling Missing Values)", fontsize=14, color="darkblue")
plt.xlabel("Reason for Delay", fontsize=12, color="purple")
plt.ylabel("Count", fontsize=12, color="purple")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.show()



plt.figure(figsize=(10, 5))
plt.scatter(delay_counts.index, delay_counts.values, color="red", edgecolor="black")

plt.title("Counts of Reasons for Delay (After Filling Missing Values)", fontsize=14, color="darkblue")
plt.xlabel("Reason for Delay", fontsize=12, color="purple")
plt.ylabel("Count", fontsize=12, color="purple")
plt.xticks(rotation=45, ha="right", fontsize=10)
plt.grid(axis="y", linestyle="--", alpha=0.6)

plt.show()




#Clean the "Railcard" column by filling missing values with the most frequent Railcard type (mode) and visualize its distribution using appropriate graphs.


df["Railcard"] = df["Railcard"].fillna(df["Railcard"].mode()[0])
railcard_counts = df["Railcard"].value_counts()

#pie chart
plt.figure(figsize=(4, 4))
plt.pie(railcard_counts, labels=railcard_counts.index, autopct="%1.1f%%", colors=["#c85d46","#467fc8","#46c87a","#c8466b"], startangle=140)
plt.title("Railcard Type Distribution (After Cleaning)")
plt.show()

#horizontal bar graph
plt.figure(figsize=(6, 3))
plt.barh(railcard_counts.index, railcard_counts.values, color="#467fc8", edgecolor="black", alpha=0.8)
plt.xlabel("Count")
plt.ylabel("Railcard Type")
plt.title("Railcard Count After Filling Missing Values")
plt.grid(axis="x", linestyle="--", alpha=0.6)
plt.show()



#Standardized "Arrival Time" format and filled missing values with the most frequent time.
df = df.copy()
df["Arrival Time"] = df["Arrival Time"].astype(str).str.strip()

df["Arrival Time"] = pd.to_datetime(df["Arrival Time"], format="%H:%M:%S", errors="coerce")

mode_values = df["Arrival Time"].mode()
if not mode_values.empty:  
    most_frequent_time = mode_values[0]
else:
    most_frequent_time = pd.to_datetime("00:00:00")  
    
df["Arrival Time"] = df["Arrival Time"].fillna(most_frequent_time)



df["Actual Arrival Time"] = df["Actual Arrival Time"].astype(str).str.strip()
df["Actual Arrival Time"] = pd.to_datetime(df["Actual Arrival Time"], format="%H:%M:%S", errors="coerce")

mode_actual_values = df["Actual Arrival Time"].mode()
if not mode_actual_values.empty:  
    most_frequent_actual_time = mode_actual_values[0]
else:
    most_frequent_actual_time = pd.to_datetime("00:00:00")  

df["Actual Arrival Time"] = df["Actual Arrival Time"].fillna(most_frequent_actual_time)



# Final Cleaned Data
df.to_csv("final_data.csv", index=False)

print("Cleaned data saved successfully as 'final_data.csv'.")
df.info()




#analyze the correlation between Price, Refund Request, and Ticket Class to understand how ticket pricing impacts refund requests and different ticket classes.
final_df = pd.read_csv("final_data.csv")

final_df["Refund Request"] = final_df["Refund Request"].map({"Yes": 1, "No": 0})
final_df["Ticket Class"] = final_df["Ticket Class"].astype("category").cat.codes

numeric_cols = final_df[["Price", "Refund Request", "Ticket Class"]]

plt.figure(figsize=(4, 3))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Heatmap of Price, Refund Request & Ticket Class")
plt.show()



#Boxplot on the basis of price to check their is outlet or not

plt.figure(figsize=(6, 4))
sns.boxplot(y=final_df["Price"], color="teal")
plt.title("Box Plot of Price")
plt.ylabel("Price")
plt.show()



# Donut Chart for 'Railcard' Distribution
railcard_counts = final_df["Railcard"].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(railcard_counts, labels=railcard_counts.index, autopct='%1.1f%%',
        wedgeprops={"edgecolor": "black"}, startangle=140, pctdistance=0.85)
# Draw a circle at the center to make it a donut chart
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
plt.gca().add_artist(centre_circle)

plt.title("Donut Chart of Railcard Distribution")
plt.show()



#KDE
final_df = pd.read_csv("final_data.csv")

# KDE Plot for Price
plt.figure(figsize=(8, 5))
sns.kdeplot(final_df["Price"], fill=True, color="blue")
plt.title("KDE Plot of Price")
plt.xlabel("Price")
plt.ylabel("Density")
plt.show()



#pair plot
final_df = pd.read_csv("final_data.csv")

# Ensure "Departure Time" is properly formatted
final_df["Departure Time"] = pd.to_datetime(final_df["Departure Time"], format="%H:%M:%S", errors="coerce")

# Drop rows where conversion failed
final_df = final_df.dropna(subset=["Departure Time"])

# Convert Departure Time to total seconds
final_df["Departure Time (Seconds)"] = final_df["Departure Time"].dt.hour * 3600 + \
                                       final_df["Departure Time"].dt.minute * 60 + \
                                       final_df["Departure Time"].dt.second

# Select numerical columns for pairplot
selected_cols = ["Price", "Departure Time (Seconds)"]

sns.pairplot(final_df[selected_cols], diag_kind="kde", plot_kws={'color': 'purple'}, diag_kws={'color': 'green'})
plt.show()




# Stacked Bar Chart for Ticket Type and Payment Method Distribution
plt.figure(figsize=(8, 5))
ticket_payment_counts = pd.crosstab(final_df["Ticket Type"], final_df["Payment Method"])
ticket_payment_counts.plot(kind="bar", stacked=True, color=["#467fc8", "#46c87a", "#c85d46", "#c8466b"])
plt.title("Stacked Bar Chart of Ticket Type and Payment Method Distribution")
plt.xlabel("Ticket Type")
plt.ylabel("Count")
plt.legend(title="Payment Method")
plt.show()




# Count Plot for Ticket Type and Ticket Class
plt.figure(figsize=(8, 5))
sns.countplot(x="Ticket Type", hue="Ticket Class", data=final_df, palette=["#8E44AD", "#3498DB"])
plt.title("Count Plot of Ticket Type and Ticket Class")
plt.xlabel("Ticket Type")
plt.ylabel("Count")
plt.legend(title="Ticket Class")
plt.show()


df.isnull().sum()