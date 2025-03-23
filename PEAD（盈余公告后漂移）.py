#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import tushare as ts
from tqdm import tqdm  # 用于显示进度条（可选：pip install tqdm）

# 初始化Tushare接口
ts.set_token('2d522f171c5a992e9a84aceb6e6aa90489cb9ef60a453d1ed122e1d7')  # 替换为您的Tushare Token
pro = ts.pro_api()

#获取目标行业股票
stock_list = pro.stock_basic(exchange='', list_status='L', fields='ts_code,name,industry')
target_industries = ['银行', '白酒', '全国地产']
stock_list= stock_list[stock_list['industry'].isin(target_industries)]

all_results = []  # 存储所有股票的结果

for ts_code in tqdm(stock_list['ts_code'], desc="处理进度"):
    try:
        # 获取当前股票近财务数据（2019-2023）
        income = pro.fina_indicator(
            ts_code=ts_code,
            start_date='20191231',
            end_date='20231231',
            fields='ts_code,ann_date,end_date,eps'
        )
        if income.empty:
            continue  # 跳过无数据的股票
        income = income.sort_values('ann_date', ascending=False).drop_duplicates('end_date', keep='first')
        income['end_date'] = pd.to_datetime(income['end_date'], format='%Y%m%d')
        income = income.set_index('end_date').sort_index()

        # 获取上一年同期数据（预期EPS）
        income_past = pro.fina_indicator(
            ts_code=ts_code,
            start_date='20181231',
            end_date='20221231',
            fields='ts_code,ann_date,end_date,eps'
        )
        if not income_past.empty:
            income_past = income_past.sort_values('ann_date', ascending=False).drop_duplicates('end_date', keep='first')
            income_past['end_date'] = pd.to_datetime(income_past['end_date'], format='%Y%m%d')
            income_past = income_past.set_index('end_date').sort_index().rename(columns={'eps': 'expected_eps'})
            
            # 合并并计算UE
            income['last_year_date'] = income.index - pd.DateOffset(years=1)
            income = income.join(income_past[['expected_eps']], on='last_year_date', how='left')
            income = income.drop(columns='last_year_date')
            income['UE'] = income['eps'] - income['expected_eps']
        else:
            income['UE'] = None  # 无历史数据时标记缺失

        # 添加股票名称和行业信息
        stock_info = stock_list[stock_list['ts_code'] == ts_code].iloc[0]
        income['name'] = stock_info['name']
        income['industry'] = stock_info['industry']
        
        all_results.append(income.reset_index())
    
    except Exception as e:
        print(f"股票 {ts_code} 处理失败: {str(e)}")

# 合并所有结果并保存
if all_results:
    final_df = pd.concat(all_results, ignore_index=True)
    final_df['end_date'] = pd.to_datetime(final_df['end_date'])
    final_df['quarter'] = final_df['end_date'].dt.to_period('Q')
    
    # 按行业和季度计算UE标准差
    sue_std = final_df.groupby(['industry', 'quarter'])['UE'].std().reset_index()
    sue_std.rename(columns={'UE': 'ue_std'}, inplace=True)
    
    # 合并标准差到原始数据
    final_df = final_df.merge(sue_std, on=['industry', 'quarter'], how='left')
    
    # 计算SUE（处理标准差为0或缺失值）
    final_df['SUE'] = final_df['UE'] / final_df['ue_std'].replace(0, pd.NA)
    
    # 保存结果
    final_df.to_csv('stock_sue_results.csv', index=False, encoding='utf-8-sig')
    print("数据已保存，SUE计算完成！")
else:
    print("无有效数据生成")


# In[31]:


pead=pd.read_csv('stock_sue_results.csv')


# In[32]:


pead


# In[33]:


# 定义列名（与图片中的列名对齐）
columns = [
    'trade_date', 'close', 'daily_return','ts_code'
]

# 创建空DataFrame
combined_df = pd.DataFrame(columns=columns)

for i in stock_list['ts_code']:
    price_data = pro.daily(ts_code=i, start_date='20190101', end_date='20240719', 
                       fields='trade_date,close,pre_close')
    price_data['trade_date'] = pd.to_datetime(price_data['trade_date'])
    price_data['daily_return'] = (price_data['close'] / price_data['pre_close']) - 1
    price_data = price_data[['trade_date', 'close', 'daily_return']]
    
    price_data['ts_code']=i
    combined_df = pd.concat([combined_df, price_data], ignore_index=True)


# In[34]:


# 获取沪深300指数日收益率
hs300 = pro.index_daily(ts_code='000300.SH', start_date='20190101', end_date='20240719',
                        fields='trade_date,close,pre_close')
hs300['trade_date'] = pd.to_datetime(hs300['trade_date'])
hs300['mk_return'] = (hs300['close'] / hs300['pre_close']) - 1
hs300 = hs300[['trade_date', 'mk_return']]


# In[35]:


# 合并市场收益率到股价数据
combined_df = combined_df.merge(hs300, on='trade_date', how='left')


# 计算异常收益率（个股日收益 - 市场日收益）
combined_df['AR'] = combined_df['daily_return'] - combined_df['mk_return']

# 按股票代码分组，并按日期升序排列
combined_df = combined_df.sort_values(['ts_code', 'trade_date'])


# In[36]:


# 去除不必要的前缀
pead['ann_date'] = pead['ann_date'].astype(str).str.extract(r'(\d{8})')[0]

# 将 ann_date 列转换为日期格式
pead['ann_date'] = pd.to_datetime(pead['ann_date'], format='%Y%m%d')


# In[37]:


import pandas as pd
from pandas.tseries.offsets import BusinessDay

# 转换日期格式
combined_df["trade_date"] = pd.to_datetime(combined_df["trade_date"])

# 生成事件窗口起始和结束日期
bd = BusinessDay()
pead["window_start"] = pead["ann_date"] - 20 * bd  # 前20个交易日
pead["window_end"] = pead["ann_date"] + 90 * bd    # 后90个交易日


# In[38]:


# 初始化空DataFrame存储结果
merged_data = pd.DataFrame()

# 逐行处理SUE表
for idx, row in pead.iterrows():
    ts_code = row["ts_code"]
    start = row["window_start"]
    end = row["window_end"]
    
    # 筛选该事件窗口内的AR数据
    mask = (
        (combined_df["ts_code"] == ts_code) &
        (combined_df["trade_date"] >= start) &
        (combined_df["trade_date"] <= end)
    )
    window_ar = combined_df[mask].copy()
    
    # 添加事件信息
    window_ar["ann_date"] = row["ann_date"]
    window_ar["SUE"] = row["SUE"]
    
    # 累积到结果
    merged_data = pd.concat([merged_data, window_ar], ignore_index=True)


# In[39]:


merged_data["event_day"] = (merged_data["trade_date"] - merged_data["ann_date"]).dt.days


# In[55]:


# 分10组（Decile 1最低，Decile 10最高）
merged_data["decile"] = pd.qcut(merged_data["SUE"], q=10, labels=range(1, 11))


# In[56]:


# 按分位数组和事件日分组计算平均AR
daily_ar = merged_data.groupby(["decile", "event_day"])["AR"].mean().reset_index()

# 按分位数组累加AR得到CAR
daily_ar["CAR"] = daily_ar.groupby("decile")["AR"].cumsum()


# In[57]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

plt.figure(figsize=(12, 8))
colors = plt.cm.Blues(np.linspace(0.3, 5, 20))  # Decile越高颜色越深

for decile in range(1, 11):
    subset = daily_ar[daily_ar["decile"] == decile]
    plt.plot(subset["event_day"], subset["CAR"] * 100,
            color=colors[decile-1],
            linewidth=1.5 if decile != 10 else 2.5,
            linestyle="--" if decile == 10 else "-",
            label=f'Decile {decile}')

# 标注公告日（0点）和零线
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.8)
plt.axhline(y=0, color='black', linewidth=1, alpha=0.8)

# 标签与标题
plt.title("Cumulative Abnormal Returns by Earnings Surprise Decile", fontsize=14)
plt.xlabel("Days from Earnings Announcement", fontsize=12)
plt.ylabel("Cumulative Average Excess Return (%)", fontsize=12)
plt.legend(title="Earnings Surprise Decile", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()


# In[ ]:




