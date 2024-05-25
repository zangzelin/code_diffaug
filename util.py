import torch
import plotly.graph_objects as go
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px

out_circle_color = ["#ffffd9","#feffd8","#feffd6","#fdfed5","#fdfed4","#fcfed3","#fcfed2","#fbfdd0","#fafdcf","#fafdce","#f9fdcd","#f9fdcb","#f8fcca","#f7fcc9","#f7fcc8","#f6fcc7","#f6fbc6","#f5fbc5","#f4fbc4","#f4fbc3","#f3fac2","#f2fac1","#f1fac0","#f1f9bf","#f0f9be","#eff9bd","#eff9bc","#eef8bb","#edf8bb","#ecf8ba","#ebf7b9","#eaf7b9","#eaf7b8","#e9f6b8","#e8f6b7","#e7f6b7","#e6f5b6","#e5f5b6","#e4f4b5","#e3f4b5","#e2f4b5","#e1f3b4","#e0f3b4","#dff2b4","#ddf2b4","#dcf1b4","#dbf1b4","#daf0b4","#d9f0b3","#d7efb3","#d6efb3","#d5eeb3","#d3eeb3","#d2edb3","#d1edb4","#cfecb4","#ceecb4","#ccebb4","#cbebb4","#c9eab4","#c8e9b4","#c6e9b4","#c4e8b4","#c3e7b5","#c1e7b5","#bfe6b5","#bde5b5","#bce5b5","#bae4b5","#b8e3b6","#b6e2b6","#b4e2b6","#b2e1b6","#b0e0b6","#aedfb6","#acdfb7","#aadeb7","#a8ddb7","#a6dcb7","#a4dbb7","#a2dbb8","#a0dab8","#9ed9b8","#9cd8b8","#99d7b9","#97d7b9","#95d6b9","#93d5b9","#91d4b9","#8fd3ba","#8dd2ba","#8ad2ba","#88d1ba","#86d0bb","#84cfbb","#82cebb","#80cebb","#7ecdbc","#7cccbc","#7acbbc","#78cabc","#76cabd","#73c9bd","#71c8bd","#6fc7bd","#6dc6be","#6bc6be","#6ac5be","#68c4be","#66c3bf","#64c3bf","#62c2bf","#60c1bf","#5ec0c0","#5cbfc0","#5abfc0","#59bec0","#57bdc0","#55bcc1","#53bbc1","#52bac1","#50bac1","#4eb9c1","#4db8c1","#4bb7c1","#49b6c2","#48b5c2","#46b4c2","#45b3c2","#43b2c2","#42b1c2","#40b0c2","#3fafc2","#3daec2","#3cadc2","#3bacc2","#39abc2","#38aac2","#37a9c2","#35a8c2","#34a7c2","#33a6c2","#32a5c2","#31a3c1","#30a2c1","#2fa1c1","#2ea0c1","#2d9fc1","#2c9dc0","#2b9cc0","#2a9bc0","#299ac0","#2898bf","#2897bf","#2796bf","#2695be","#2693be","#2592be","#2591bd","#248fbd","#248ebc","#238cbc","#238bbb","#228abb","#2288ba","#2287ba","#2185b9","#2184b9","#2182b8","#2181b8","#217fb7","#217eb6","#207cb6","#207bb5","#2079b5","#2078b4","#2076b3","#2075b3","#2073b2","#2072b1","#2070b1","#216fb0","#216daf","#216cae","#216aae","#2169ad","#2167ac","#2166ac","#2164ab","#2163aa","#2261aa","#2260a9","#225ea8","#225da7","#225ca7","#225aa6","#2259a5","#2257a5","#2256a4","#2354a3","#2353a3","#2352a2","#2350a1","#234fa0","#234ea0","#234c9f","#234b9e","#234a9d","#23499d","#23479c","#23469b","#23459a","#224499","#224298","#224197","#224096","#223f95","#223e94","#213d93","#213c92","#213a91","#203990","#20388f","#20378d","#1f368c","#1f358b","#1e348a","#1e3388","#1d3287","#1d3185","#1c3184","#1c3082","#1b2f81","#1a2e7f","#1a2d7e","#192c7c","#182b7a","#172b79","#172a77","#162975","#152874","#142772","#132770","#13266e","#12256c","#11246b","#102469","#0f2367","#0e2265","#0d2163","#0d2161","#0c2060","#0b1f5e","#0a1e5c","#091e5a","#081d58"]

def pad_table_data(data, pad_dim, d1, d2):
    
    d = torch.cat(
        [data.detach().cpu(), torch.zeros(pad_dim)]
        ).numpy().reshape(d1, d2)
    return d

def point(cen=np.array([0, 0])):
    
    o = np.array([0, 0]).reshape(1, 2) + cen
    a = np.array([1, 0]).reshape(1, 2) + cen
    b = np.array([0.5, 0.866]).reshape(1, 2) + cen
    c = np.array([-0.5, 0.866]).reshape(1, 2) + cen
    d = np.array([-1, 0]).reshape(1, 2) + cen
    e = np.array([-0.5, -0.866]).reshape(1, 2) + cen
    f = np.array([0.5, -0.866]).reshape(1, 2) + cen
    no = [np.array([None, None]).reshape(1, 2)] 

    a_list = [a, b, c, d, e, f]

    # 0到a差值
    oa = [t*o+(1-t)*a for t in np.linspace(0, 1, 10)]
    ob = [t*o+(1-t)*b for t in np.linspace(0, 1, 10)]
    oc = [t*o+(1-t)*c for t in np.linspace(0, 1, 10)]
    od = [t*o+(1-t)*d for t in np.linspace(0, 1, 10)]
    oe = [t*o+(1-t)*e for t in np.linspace(0, 1, 10)]
    of = [t*o+(1-t)*f for t in np.linspace(0, 1, 10)]
    # og = [t*o+(1-t)*g for t in np.linspace(0, 1, 10)]

    ab = [t*a+(1-t)*b for t in np.linspace(0, 1, 10)]
    bc = [t*b+(1-t)*c for t in np.linspace(0, 1, 10)]
    cd = [t*c+(1-t)*d for t in np.linspace(0, 1, 10)]
    de = [t*d+(1-t)*e for t in np.linspace(0, 1, 10)]
    ef = [t*e+(1-t)*f for t in np.linspace(0, 1, 10)]
    # fg = [t*f+(1-t)*g for t in np.linspace(0, 1, 10)]
    fa = [t*f+(1-t)*a for t in np.linspace(0, 1, 10)]


    list_point = oa+no+ob+no+oc+no+od+no+oe+no+of+no
    list_point += ab + no+ bc + no+ cd + no+ de + no+ ef + no+ fa + no

    # np.concatenate(oa+no) 

    # print(oa)

    return np.concatenate(list_point)

def p_point(center_x = 0, center_b = 0, center_r = 0.866):

    a = point(cen=np.array([center_x+0, center_b + 0]))
    b = point(cen=np.array([center_x+2, center_b + 0]))
    c = point(cen=np.array([center_x+-2, center_b + 0]))
    d = point(cen=np.array([center_x+0, center_b + 2*center_r]))
    e = point(cen=np.array([center_x+0, center_b + -2*center_r]))
    f = point(cen=np.array([center_x+2, center_b + -2*center_r]))
    g = point(cen=np.array([center_x+2, center_b + 2*center_r]))
    h = point(cen=np.array([center_x+-2, center_b + 2*center_r]))
    i = point(cen=np.array([center_x+-2, center_b + -2*center_r]))
    
    a1 = point(cen=np.array([center_x+4, center_b + 0]))
    a2 = point(cen=np.array([center_x+-4, center_b + 0]))
    a3 = point(cen=np.array([center_x+4, center_b + 2*center_r]))
    a4 = point(cen=np.array([center_x+4, center_b + -2*center_r]))
    a5 = point(cen=np.array([center_x+-4, center_b + 2*center_r]))
    a6 = point(cen=np.array([center_x+-4, center_b + -2*center_r]))
    a7 = point(cen=np.array([center_x+0, center_b + 4*center_r]))
    a8 = point(cen=np.array([center_x+0, center_b + -4*center_r]))

    b1 = point(cen=np.array([center_x+2, center_b + -4*center_r]))
    b2 = point(cen=np.array([center_x+2, center_b + 4*center_r]))
    b3 = point(cen=np.array([center_x+-2, center_b + -4*center_r]))
    b4 = point(cen=np.array([center_x+-2, center_b + 4*center_r]))

    return np.concatenate([a, b , c, d, e, f, g, h, i, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4])

def LineBackground(fig, model, center_r = 0.866):

    line_data = np.concatenate([
            p_point(center_x = 0, center_b = 0, center_r = center_r),
            p_point(center_x = 4, center_b = 0, center_r = center_r),
            p_point(center_x = -4, center_b = 0, center_r = center_r),
            p_point(center_x = 0, center_b = 4*center_r, center_r = center_r),
            p_point(center_x = 0, center_b = -4*center_r, center_r = center_r),
         ])
    
    # import pdb; pdb.set_trace()
    mask = (line_data == None)
    line_data[mask] = 0
    line_data = (line_data/3.0).astype(np.float32)
    line_data  = model.ToPoincare(
        torch.tensor(line_data)
        ).detach().cpu().numpy()
    line_data[mask] = None

    fig.add_trace(go.Scatter(
        x=line_data[:,0], 
        y=line_data[:,1], 
        line_shape='spline',
        marker=dict(color='rgb(245, 245, 245)'),
        line=dict(
            color=out_circle_color[80],
            width=0.5,
        )
        ))
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1.05, y0=-1.05, x1=1.05, y1=1.05,
        line=dict(
            color=out_circle_color[100],
            width=0.5,
        )
    )
    return fig

def ScatterVis(fig, emb_vis, labels, size=2):
    
    # c_list = list(px.colors.qualitative.Light24) * 10
    c_list = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"] *10

    label_list = list(set(labels))
    
    label_dis = []
    for label in label_list:
        dis = np.linalg.norm(emb_vis[labels==label])
        label_dis.append(dis)
    
    # sort the label by the distance
    label_list = [x for _, x in sorted(zip(label_dis, label_list))]
    
    for label in label_list:
        mask = (labels == label)

        scatter_emb_vis = go.Scatter(
            x=emb_vis[mask][:,0],
            y=emb_vis[mask][:,1],
            mode='markers',
            marker=dict(
                color=c_list[label],  # 使用labels来设置颜色
                colorscale='Viridis',  # 设置颜色映射
                cmin=0,  # 设置颜色映射范围的最小值
                cmax=len(set(labels)) - 1,  # 设置颜色映射范围的最大值
                size=size,
            ),
            line=dict(
            color='black',  # 设置边框颜色为黑色
            width=1  # 设置边框宽度
            ),
        )
        fig.add_trace(scatter_emb_vis)
    return fig


def ScatterCenter(fig, cluster_center, labels):
    scatter_cluster_center = go.Scatter(
        x=cluster_center[:,0],
        y=cluster_center[:,1],
        mode='markers',
        marker=dict(
            color='black',              # 使用labels来设置颜色
            colorscale='Viridis',       # 设置颜色映射
            cmin=0,                     # 设置颜色映射范围的最小值
            cmax=len(set(labels)) - 1,   # 设置颜色映射范围的最大值
        ),
        line=dict(
            color='black',              # 设置边框颜色为黑色
            width=1                     # 设置边框宽度
        ),
    )
    fig.add_trace(scatter_cluster_center)
        
    return fig

def OuterRing(fig, cluster_center, labels, l3_r=1.35, l2_r=1.2, l1_r=1.05):
    # cluster with kmeans
    kmeans = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(cluster_center)
    l2_labels = kmeans.labels_
    l2_center = kmeans.cluster_centers_
    
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1*l1_r, y0=-1*l1_r, x1=l1_r, y1=l1_r,
        line=dict(
            color=out_circle_color[120],
            width=0.5,
            # opacity=0.5,
        )
    )
    # normalize the cluster_center
    cluster_center_normed = l1_r * (cluster_center / np.linalg.norm(cluster_center, axis=1, keepdims=True))
    # import pdb; pdb.set_trace()
    
    scatter_cluster_center_normed = go.Scatter(
        x=cluster_center_normed[:,0],
        y=cluster_center_normed[:,1],
        mode='markers',
        marker=dict(
            color='black',                # 使用labels来设置颜色
            colorscale='Viridis',       # 设置颜色映射
            cmin=0,                     # 设置颜色映射范围的最小值
            cmax=len(set(labels)) - 1,   # 设置颜色映射范围的最大值
            size=5,
        ),
        line=dict(
            color='#fbf4f9',              # 设置边框颜色为黑色
            width=1,                    # 设置边框宽度
        ),
    )
    fig.add_trace(scatter_cluster_center_normed)
    
    l2_center_normed = l2_r * (l2_center / np.linalg.norm(l2_center, axis=1, keepdims=True))
    scatter_l2_center_normed = go.Scatter(
        x=l2_center_normed[:,0],
        y=l2_center_normed[:,1],
        mode='markers',
        marker=dict(
            color='black',                # 使用labels来设置颜色
            colorscale='Viridis',       # 设置颜色映射
            cmin=0,                     # 设置颜色映射范围的最小值
            cmax=len(set(labels)) - 1,   # 设置颜色映射范围的最大值
            size=5,
        ),
        line=dict(
            color='black',              # 设置边框颜色为黑色
            width=1,                    # 设置边框宽度
        ),
    )
    fig.add_trace(scatter_l2_center_normed)
    
    kmeans = KMeans(n_clusters=5, random_state=0, n_init="auto").fit(l2_center)
    l3_labels = kmeans.labels_
    l3_center = kmeans.cluster_centers_

    l3_center_normed = l3_r * (l3_center / np.linalg.norm(l3_center, axis=1, keepdims=True))

    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1*l3_r, y0=-1*l3_r, x1=l3_r, y1=l3_r,
        line=dict(
            color=out_circle_color[180],
            width=0.5,
            # opacity=0.5,
        )
    )

    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1*l2_r, y0=-1*l2_r, x1=l2_r, y1=l2_r,
        line=dict(
            color=out_circle_color[180],
            width=0.5,
            # opacity=0.5,
        )
    )

    scatter_l3_center_normed = go.Scatter(
        x=l3_center_normed[:,0],
        y=l3_center_normed[:,1],
        mode='markers',
        marker=dict(
            color='black',                # 使用labels来设置颜色
            colorscale='Viridis',       # 设置颜色映射
            cmin=0,                     # 设置颜色映射范围的最小值
            cmax=len(set(labels)) - 1,  # 设置颜色映射范围的最大值
            size=7
        ),
        line=dict(
            color='black',              # 设置边框颜色为黑色
            width=1                     # 设置边框宽度
        ),
    )
    fig.add_trace(scatter_l3_center_normed)

    l1_l2_link = []
    for l1_center_index, l2_center_index in enumerate(l2_labels):
        l1_l2_link.append(np.array([[0, 0]]))
        l1_l2_link.append(cluster_center[l1_center_index][None, :])
        l1_l2_link.append(cluster_center_normed[l1_center_index][None, :])
        l1_l2_link.append(l2_center_normed[l2_center_index][None, :])
        l1_l2_link.append(l3_center_normed[l3_labels[l2_center_index]][None, :])
        l1_l2_link.append(np.array([[None, None]]))
    
    l1_l2_link = np.concatenate(l1_l2_link, axis=0)
    # import pdb; pdb.set_trace()

    fig.add_trace(go.Scatter(
        x=l1_l2_link[:,0], 
        y=l1_l2_link[:,1], 
        line_shape='spline',
        marker=dict(color='rgba(128, 128, 128, {})'.format(0.2)),
        line=dict(
            width=0.5,
        ),
        ))

    return fig


def OuterRing_S(fig, cluster_center, labels, l3_r=1.35, l2_r=1.2, l1_r=1.005):
    # cluster with kmeans
    fig.add_shape(
        type="circle",
        xref="x", yref="y",
        x0=-1*l1_r, y0=-1*l1_r, x1=l1_r, y1=l1_r,
        line=dict(
            color=out_circle_color[120],
            width=0.5,
        )
    )
    return fig


def Layout(fig,):

    fig.update_layout(
        plot_bgcolor='white',  # 设置背景色为白色
        autosize=False,  # 关闭自动调整大小
        width=1000,  # 设置图表宽度
        height=800,  # 设置图表高度
        margin=dict(
            l=50,  # 左边距
            r=50,  # 右边距
            b=50,  # 底边距
            t=80,  # 顶边距
            pad=2  # 图表与边界的间距
        ), # 设置背景色为白色
    )
    return fig