#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


superstore_df = pd.read_csv("./data/Sample - Superstore.csv", encoding='windows-1252')


# In[3]:


superstore_df.head(2)


# In[4]:


superstore_df['Order Date'] = pd.to_datetime(superstore_df['Order Date'])
superstore_df['Ship Date'] = pd.to_datetime(superstore_df['Ship Date'])


# In[5]:


superstore_df.isnull().sum()


# In[6]:


superstore_df.describe()


# In[7]:


superstore_df.skew(numeric_only=True)


# ## RFM

# Since the data is from '2014-01-03' to '2017-12-30', when we calcualte the recency column we can't use today's date as it is too far so we will use TODAY as 1st Jan 2018

# In[8]:


#setting today's date as a variable as it will keep changing

TODAY = pd.to_datetime('2018-01-01')


# In[9]:


rfm = superstore_df.groupby(['Customer ID']).agg({'Order ID':'count',
                                            'Sales':'sum',
                                            'Order Date': lambda x: (TODAY-x.max()).days}).reset_index().rename(
                                                columns={'Order ID':'frequency',
                                                         'Sales':'monetary',
                                                         'Order Date':'recency'})

rfm.head()


# Note on pd.qcut: https://pbpython.com/pandas-qcut-cut.html
# 
# - The simplest use of qcut is to define the number of quantiles and let pandas figure out how to divide up the data. So if we declare q as 4, we tell pandas to create 4 equal sized groupings of the data. You'll notice that because we asked for quantiles with q=4 the bins match the percentiles from the describe function.
# 
# - One of the challenges with this approach is that the bin labels are not very easy to explain to an end user. For instance, if we wanted to divide our customers into 5 groups (aka quintiles) like an airline frequent flier approach, we can explicitly label the bins to make them easier to interpret. That's where the label parameter is useful.

# In[10]:


rfm['recency_quantile'] = pd.qcut(rfm["recency"],q=4,labels=[1,2,3,4])
rfm['frequency_quantile'] = pd.qcut(rfm["frequency"],q=4,labels=[4,3,2,1])
rfm['monetary_quantile'] = pd.qcut(rfm["monetary"],q=4,labels=[4,3,2,1])

rfm.head()


# Note: Difference between str(df['col]) vs df['col].astype(str) (https://stackoverflow.com/questions/30095172/difference-between-str-and-astypestr)
# 
# - str(df['col']) will convert the contents of the entire column in to single string. It will end up with a very big string. 
# - You should use `.astype(str)` if you want to convert your entire column to type string

# Note: Bitwise vs Logical operators 
# 
# - Logical operators (`and` and `or`): are used to combine Boolean expressions, and they operate on whole expressions rather than element-wise. They work by evaluating the truth value of entire expressions, not the individual elements of an array or pandas Series.
# 
# - Bitwise operators (`&` and `|`): When used with pandas Series (or NumPy arrays), these operators perform element-wise operations. This means they compare each element of the series individually. Even though these operators are called "bitwise," when applied to Boolean arrays (like those in pandas), they act as element-wise logical operators.

# In[11]:


rfm['rfm_rank'] = rfm['recency_quantile'].astype(str) + rfm['frequency_quantile'].astype(str) + rfm['monetary_quantile'].astype(str)

rfm.head()


# In[12]:


def customer_label_function(rfm):
    if rfm['rfm_rank']=='111':
        return "champions"
    elif (rfm['frequency_quantile']==1) & (rfm['recency_quantile']!=4):
        return "loyal customers"
    elif (rfm['monetary_quantile']==1) & (rfm['recency_quantile']!=4):
        return "big spenders"
    elif (rfm['recency_quantile']==1) & (rfm['frequency_quantile']!=4):
        return "recent users"
    elif rfm['rfm_rank']=='444':
       return  "lost"
    else:
       return  "other"


# In[13]:


rfm['customer_label'] = rfm.apply(customer_label_function,axis=1)


# In[14]:


rfm['customer_label'].value_counts()


# In[15]:


rfm.head()


# In[16]:


len(rfm)


# ## Customer Segmentation of Loyal Customers

# ### Feature Engineering 

# In[17]:


loyal_customers = rfm[rfm['customer_label']=='loyal customers']


# In[18]:


loyal_customers.head(2)


# In[19]:


loyal_customer_sales = loyal_customers.merge(superstore_df,how='inner',on='Customer ID')


# In[20]:


loyal_customer_sales.columns


# In[21]:


loyal_customer_sales[loyal_customer_sales['Row ID'].duplicated()]


# In[22]:


loyal_customer_sales.head()


# In[23]:


loyal_customer_sales['Segment'].value_counts()


# In[24]:


loyal_customer_sales['Category'].value_counts()


# In[25]:


loyal_customer_sales['Sub-Category'].value_counts()


# In[26]:


loyal_customer_sales['Region'].value_counts()


# In[27]:


loyal_customer_sales['Discount'].value_counts()


# In[28]:


loyal_customer_sales['Ship Mode'].value_counts()


# In[29]:


loyal_customer_sales.sort_values(by=['Customer ID','Order Date'],inplace=True)
loyal_customer_sales['Previous Order Date'] = loyal_customer_sales.groupby(['Customer ID']).shift()['Order Date']
loyal_customer_sales['Time b/w previous purchase'] = (loyal_customer_sales['Order Date']-loyal_customer_sales['Previous Order Date']).dt.days


# In[30]:


loyal_customer_sales.head()


# In[31]:


loyal_customer_sales[loyal_customer_sales['Time b/w previous purchase']<0]


# In[32]:


loyal_customer_sales['Country'].value_counts()


# In[33]:


# get most recent city, state and region

grouped_location_data = loyal_customer_sales.sort_values(by=['Customer ID','Order Date'],ascending=[True,
                                                                            False]).groupby(['Customer ID']).first()[['City',
                                                                                                                      'State',
                                                                                                                      'Region']].reset_index()

grouped_location_data.rename(columns={"City":'recent_city',"State":'recent_state',"Region":'recent_region'},inplace=True)
grouped_location_data.head(2)


# In[34]:


grouped_category_data = loyal_customer_sales.groupby(['Customer ID','Category']).count()['Order ID'].reset_index()

grouped_category_data.head(2)


# In[35]:


pivoted_category_data = grouped_category_data.pivot(index='Customer ID',
                                                    columns='Category',values='Order ID').reset_index().rename(columns = {'Furniture':'furniture_count',
                                                                                                                'Office Supplies':'office_supplies_count',
                                                                                                                'Technology':'tech_count'})
pivoted_category_data.fillna(0,inplace=True)
pivoted_category_data.head(2)


# In[36]:


grouped_shipping_data = loyal_customer_sales.groupby(['Customer ID','Ship Mode']).count()['Order ID'].reset_index()

pivoted_shipping_data = grouped_shipping_data.pivot(index='Customer ID',
                                                    columns='Ship Mode',values='Order ID').reset_index().rename(columns = {'Standard Class':'standard_class_count',
                                                                                                                'Second Class':'second_class_count',
                                                                                                                'First Class':'first_class_count',
                                                                                                                'Same Day':'same_class_count'})

pivoted_shipping_data.fillna(0,inplace=True)
pivoted_shipping_data.head(2)


# In[37]:


grouped_segment_data = loyal_customer_sales.groupby(['Customer ID','Segment']).count()['Order ID'].reset_index()

pivoted_segment_data = grouped_segment_data.pivot(index='Customer ID',
                                                    columns='Segment',values='Order ID').reset_index().rename(columns = {'Consumer':'consumer_segment_count',
                                                                                                                'Home Office':'home_office_segment_count',
                                                                                                                'Corporate':'corporate_segment_count'})

pivoted_segment_data.fillna(0,inplace=True)
pivoted_segment_data.head(2)


# In[38]:


grouped_sales_data = loyal_customer_sales.groupby(['Customer ID']).agg({'Sales':'mean','Quantity':'mean',
                                                #    'Segment': lambda x: x.value_counts().index[0],
                                                #    'Ship Mode':lambda x: x.value_counts().index[0],
                                                #    'Category':lambda x: x.value_counts().index[0],
                                                   'Time b/w previous purchase':'mean',
                                                   'Order Date':lambda x: (TODAY-x.min()).days}).reset_index().rename(columns={
                                                       "Sales":'avg_order_value','Quantity':'avg_quantity',
                                                       "Segment":'freq_segment','Ship Mode':'freq_ship_mode',
                                                       'Time b/w previous purchase':'avg_time_b/w_orders',
                                                       'Order Date':'customer_tenure'
                                                   })

grouped_sales_data.head(2)


# In[39]:


# grouped_data = grouped_sales_data.merge(grouped_location_data,on='Customer ID',how='inner')
# grouped_data = grouped_data.merge(pivoted_category_data,on='Customer ID',how='inner')
# grouped_data = grouped_data.merge(pivoted_shipping_data,on='Customer ID',how='inner')
# grouped_data = grouped_data.merge(pivoted_segment_data,on='Customer ID',how='inner')

grouped_data = grouped_sales_data.merge(loyal_customer_sales[['Customer ID','frequency', 'monetary', 'recency', 'recency_quantile',
       'frequency_quantile', 'monetary_quantile', 'rfm_rank']],on='Customer ID',how='inner').drop_duplicates(subset=['Customer ID'])

grouped_data.head()


# In[40]:


grouped_data.columns


# ### Clustering

# In[41]:


#TODO: 
# 1. convert categorical columns to numeric - done
# 2. Check assumptions of K Means and scale data
# 3. Apply PCA
# 4. Get optimal K using silhoutte method and apply K Means


# In[42]:


from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[67]:


# clustering_data = grouped_data.drop(columns=['Customer ID','recent_city','recent_state','recent_region'])
clustering_data = grouped_data.drop(columns=['Customer ID','recency_quantile', 'frequency_quantile', 'monetary_quantile',
       'rfm_rank','monetary'])


# In[68]:


# vif_data = clustering_data.copy()

# for index in range(len(vif_data.columns)):
#     print(vif_data.columns[index] + ":")
#     print(variance_inflation_factor(vif_data.values, index))


# In[69]:


clustering_data.columns


# In[70]:


clustering_data.dtypes


# In[71]:


for col in clustering_data.columns:
    clustering_data[col] = clustering_data[col].astype('float64')

clustering_data.dtypes


# In[72]:


clustering_data.isnull().sum()


# In[73]:


clustering_data.shape


# K Means assumptions:
# 
# 1. Variables have same mean
# 2. Variables have same variance
# 3. Symmetric distribution of variables
# 

# In[74]:


clustering_data.skew(numeric_only=True)


# In[75]:


import matplotlib.pyplot as plt

plt.hist(clustering_data['avg_order_value'])
plt.show()


# In[76]:


clustering_data.describe()


# Need to transform data as the assumptions do not hold

# In[77]:


std = StandardScaler()
scaled_data = std.fit_transform(clustering_data)


# In[78]:


scaled_data.shape


# Apply PCA to reduce the dimensions

# In[79]:


from sklearn.decomposition import PCA

pca= PCA()
pca_clustering_data = pca.fit(scaled_data)


# In[80]:


pca.explained_variance_ratio_


# In[81]:


plt.plot(range(1,len(pca.explained_variance_ratio_)+1),pca.explained_variance_ratio_.cumsum(),marker='o',linestyle='--')
plt.title('Explained Variance by Components')
plt.xlabel('# of Components')
plt.title('Cumulative Explained Variance')


# The graph shows the amount of variance captured (on the y-axis) depending on the number of components we include (the x-axis). A rule of thumb is to preserve around 80 % of the variance. So, in this instance, we decide to keep 4 components.

# In[82]:


pca= PCA(n_components=4)
pca_clustering_data = pca.fit_transform(scaled_data)


# In[83]:


pca_clustering_data.shape


# K Means

# In[84]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
silhouette_avg = []
for num_clusters in range_n_clusters:
    # initialise kmeans
    kmeans = KMeans(n_clusters=num_clusters,init='k-means++',random_state=123)
    kmeans.fit(pca_clustering_data)
    cluster_labels = kmeans.labels_
    # silhouette score
    silhouette_avg.append(silhouette_score(pca_clustering_data, cluster_labels))

plt.plot(range_n_clusters,silhouette_avg)
plt.xlabel('Values of K') 
plt.ylabel('Silhouette score') 
plt.title('Silhouette analysis For Optimal k')
plt.show()


# We see that the k means silhouette score is maximized at k = 3. So, we will take 3 clusters.

# In[85]:


kmeans = KMeans(n_clusters=3,init='k-means++',random_state=123)
kmeans.fit(pca_clustering_data)
cluster_labels = kmeans.labels_


# In[86]:


len(cluster_labels)


# In[87]:


grouped_data


# In[92]:


kmeans_pca_df = pd.concat([grouped_data['Customer ID'],clustering_data.copy()],axis=1)
kmeans_pca_df['kmeans_labels'] = cluster_labels
kmeans_pca_df


# In[94]:


kmeans_pca_df['kmeans_labels'].value_counts()


# ## Cluster Analysis

# ### Cluster 0
# 
# - Lowest average order value ($154)
# - Mean avg_quantity valye (3.69)
# - Higher days between order i.e. customers take time between orders
# - Highest customer tenure: long time customers
# - lowest frequency: haven't placed that many orders
# - Lowest recency: though the number is that far off from the other cluster, they haven't placed orders recently
# 
# ### Cluster 0: Long-Time Customers with Low Engagement
# 
# - Characteristics: These customers have the lowest average order value, a moderate purchase quantity, take the longest time between orders, have the highest tenure, but the lowest order frequency and recency.
# 
# - Recommendations:
# 1. Re-engagement Campaigns: Use targeted marketing campaigns to encourage repeat purchases. Offer time-sensitive discounts or promotions to prompt these customers to make a purchase sooner.
# 
# 2. Loyalty Programs: Since they are long-time customers, consider implementing or enhancing loyalty rewards to re-engage them. Offer incentives for more frequent purchases or larger order values.
# 
# 3. Personalized Offers: Use personalized marketing to remind them of their past purchases or suggest products related to their previous buying habits.

# In[96]:


kmeans_pca_df[kmeans_pca_df['kmeans_labels']==0].describe()


# ### Cluster 1
# 
# - Highest average order value ($320)
# - Highest avg_quantity value (4.15)
# - Average(mean) days between order 
# - lowest customer tenure: relatively new customers
# - low(average) frequency: haven't placed that many orders (better than cluster 0 but behind cluster 2)
# - highest recency: they have placed many orders recently
# 
# ### Cluster 1: High-Value Recent Customers
# 
# - Characteristics: Highest average order value and quantity, average days between orders, the lowest customer tenure, low frequency, but highest recent activity.
# 
# - Recommendations:
# 
# 1. Upsell and Cross-Sell Opportunities: Capitalize on their high engagement and recent activity by offering complementary products or premium versions of what they have purchased.
# 
# 2. Retention Strategies: Focus on retaining these high-value customers by providing excellent customer service, exclusive offers, or early access to new products.
# 
# 3. Targeted Communications: Since these customers have high recency and value, ensure regular, personalized communication to maintain their interest and engagement.

# In[97]:


kmeans_pca_df[kmeans_pca_df['kmeans_labels']==1].describe()


# ### Cluster 2
# 
# - Lower average order value ($181): not as low as cluster 0 but not that high also
# - Highest avg_quantity value (3.48)
# - Lowest days between order: they don't have that many days between orders
# - Average customer tenure
# - highest frequency: they have placed that most orders 
# - Average recency: the number is barely better than cluster 0 but behind cluster 1
# 
# ### Cluster 2: Frequent Buyers with Moderate Order Value
# 
# - Characteristics: Moderate order value, highest purchase quantity, lowest days between orders, average tenure, highest frequency, and average recency.
# 
# - Recommendations:
# 
# 1. Encourage Higher Spending: These customers purchase frequently but have a moderate order value. Use techniques like bundle offers or volume discounts to encourage them to increase their order size.
# 
# 2. Sustain Frequent Purchases: Maintain their frequent purchasing behavior by offering a subscription service or regular delivery options to lock in their spending habits.
# 
# 3. Engage with Product Recommendations: Provide them with personalized product recommendations based on their frequent purchases to boost both order value and engagement.

# In[98]:


kmeans_pca_df[kmeans_pca_df['kmeans_labels']==2].describe()


# In[ ]:




