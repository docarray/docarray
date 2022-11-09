import pandas as pd
import plotly.express as px

storages = ['memory', 'annlite', 'elasticsearch', 'qdrant', 'weaviate', 'redis']

df = pd.DataFrame()

for storage in storages:
    df_tmp = pd.read_csv(f'benchmark-qps-{storage}.csv')
    df_tmp.rename(
        columns={
            'Recall at k=10 for vector search': 'Recall@10',
            'Find by vector': 'QPS',
        },
        inplace=True,
    )
    df_tmp.sort_values(by=['Recall@10'], inplace=True)
    df = pd.concat([df, df_tmp])

fig = px.scatter(
    df,
    x="Recall@10",
    y="QPS",
    color='Storage Backend',
    hover_data=['Max_Connections', 'EF_Construct', 'EF'],
    trendline="lowess",
)

fig.write_html('benchmark.html', include_plotlyjs='cdn', full_html=False)
