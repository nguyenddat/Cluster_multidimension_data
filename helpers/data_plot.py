import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.cm as cm
import plotly.express as px
import plotly.graph_objects as go


def multi_scatterplot(df):
    df = df.copy()
    df = df.sort_values('epochtime')

    motion_data = df[df['datatype'] == 'motion']  # Binary data
    continuous_data = df[df['datatype'] != 'motion']  # Dữ liệu liên tục
    
    datatype_counts = continuous_data['datatype'].value_counts()
    datatypes = datatype_counts.index.tolist()
    datatypes = continuous_data['datatype'].unique()
    colors = sns.color_palette("terrain_r", len(datatypes))
    color_dict = dict(zip(datatypes, colors))

    fig, ax1 = plt.subplots(figsize=(10, 4))

    for datatype in datatypes:
        data_subset = continuous_data[continuous_data['datatype'] == datatype]
        ax1.scatter(
            data_subset['epochtime'], 
            data_subset['value'], 
            marker='.', 
            label=datatype, 
            color=color_dict[datatype]
        )

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Continuous Value', fontsize=12)
    ax1.legend(title='Continuous Data', fontsize=10, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.5)

    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.xticks(rotation=45)
    # plt.xlabel("Tháng")
    # plt.ylabel(f"{datatype} ({unit})")
    # plt.title(f"Biểu đồ {datatype} theo thời gian")
    plt.show()


def multi_lineplot(df, title):
    df = df.copy() 
    df["epochtime"] = pd.to_datetime(df["epochtime"])
    df = df.sort_values('epochtime')
  
    motion_data = df[df['datatype'] == 'motion']  # Binary data
    continuous_data = df[df['datatype'] != 'motion']  # Dữ liệu liên tục

    datatypes = continuous_data['datatype'].unique()
    colors = sns.color_palette("husl", len(datatypes))
    color_dict = dict(zip(datatypes, colors))
    
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # Plot each datatype as a separate line
    for datatype in datatypes:
        data_subset = continuous_data[continuous_data['datatype'] == datatype]
        ax1.plot(data_subset['epochtime'], data_subset['value'], 
                 marker='o', linestyle='-', label=datatype, 
                 color=color_dict[datatype], markersize=4, alpha=0.8)

    ax1.set_xlabel('Time', fontsize=12)
    ax1.set_ylabel('Continuous Value', fontsize=12)
    ax1.legend(title='Continuous Data', fontsize=10, loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)

    if not motion_data.empty:
        ax2 = ax1.twinx()  # Trục y thứ hai
        ax2.scatter(motion_data['epochtime'], motion_data['value'], 
                    color='black', marker='o', s=10, alpha=0.6, label='Motion (binary)')
        ax2.set_ylabel('Motion (0 or 1)', fontsize=12)
        ax2.legend(fontsize=10, loc='lower right')

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=45)

    if title:
        ax1.set_title(title, fontsize=14)
    else:
        start_time = df['epochtime'].min().strftime('%Y-%m-%d %H:%M')
        end_time = df['epochtime'].max().strftime('%Y-%m-%d %H:%M')
        ax1.set_title(f'Sensor Data: {start_time} to {end_time}', fontsize=14)

    plt.tight_layout()
    plt.show()


def barplot(time, value):
    df = pd.DataFrame({"time": time, "value": value})

    df["month"] = df["time"].dt.to_period("M")
    monthly_counts = df.groupby(["month", "value"]).size().unstack(fill_value=0)

    months = monthly_counts.index.astype(str)

    x = np.arange(len(months))  # Mảng chỉ số cho trục x
    width = 0.4  # Độ rộng mỗi cột

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, monthly_counts[0], width=width, label="0", color="blue")
    plt.bar(x + width/2, monthly_counts[1], width=width, label="1", color="orange")

    # Định dạng trục x
    plt.xticks(x, months, rotation=45)
    plt.xlabel("Tháng")
    plt.ylabel("Số lần xuất hiện")
    plt.title("Phân bố giá trị theo tháng")
    plt.legend(title="Giá trị")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Hiển thị biểu đồ
    plt.show()


def lineplot(time, value, data):
    datatype = data["datatype"]
    unit = data["unit"]

    plt.figure(figsize = (10, 4))
    # plt.plot(time, value, marker="x", color = "red")
    plt.scatter(time, value, marker=".", color="blue")
    
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.xticks(rotation=45)
    plt.xlabel("Tháng")
    plt.ylabel(f"{datatype} ({unit})")
    plt.title(f"Biểu đồ {datatype} theo thời gian")
    plt.show()


def boxplot(data):
    plt.figure(figsize=(10, 4))

    sns.boxplot(y = "Giữa 2 lần đo (phút)", data = data, color = "skyblue")
    # sns.stripplot(y = "Giữa 2 lần đo (phút)", data = data, color = "black", alpha = 0.5, jitter = True)
    plt.title('Box plot của các khoảng thời gian giữa 2 lần đo', fontsize=14)
    plt.ylabel('Giữa 2 lần đo', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def plot_3d(data, labels):
    colors = px.colors.qualitative.Set1
    fig = go.Figure()

    for i in range(len(labels)):
        cluster_points = data[labels == i]
        
        fig.add_trace(go.Scatter3d(
            x=cluster_points[:, 0],
            y=cluster_points[:, 1],
            z=cluster_points[:, 2],
            mode='markers',
            marker=dict(size=6, color=colors[i % len(colors)], opacity=0.8),
            name=f'Cluster {i}'
        ))

    fig.update_layout(
        title=f'Biểu diễn các cụm dữ liệu lấy theo giá trị trung bình của time-series',
        scene=dict(
            xaxis_title='Feature 1 (mean)',
            yaxis_title='Feature 2 (mean)',
            zaxis_title='Feature 3 (mean)',
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        width=1000,  # Tăng chiều rộng
        height=450   # Tăng chiều cao
    )
    fig.show()