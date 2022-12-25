from sklearn.metrics import precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import json

def auc_score(y_predict, y_true):
    total_score=0
    assert len(y_predict)==len(y_true)
    for i in range(len(y_true)):
        for j in range(i+1,len(y_true)):
            if(y_true[i]==y_true[j]):
                continue
            if y_true[i]>y_true[j] and y_predict[i]<y_predict[j]:
                total_score+=1
                continue
            if y_true[i]<y_true[j] and y_predict[i]>y_predict[j]:
                total_score+=1
                continue
            if y_predict[i]==y_predict[j]:
                total_score+=0.5
    all_sample=sum(y_true)*(len(y_true)-sum(y_true))
    return total_score/all_sample

def best_F_score(y_pred,y_true):
    y_pred=np.asarray(y_pred, dtype=np.float64)
    y_true=np.asarray(y_true)
    max_y_pred=np.max(y_pred)
    min_y_pred=np.min(y_pred)
    y_pred=(y_pred-min_y_pred)/(max_y_pred-min_y_pred)
    precisions, recalls, thresholds=precision_recall_curve(y_true, y_pred, pos_label=0)
    f1_scores=(2*precisions*recalls)/(precisions+recalls)
    best_f_score=np.max(f1_scores[np.isfinite(f1_scores)])
    threshold=thresholds[np.argmax(f1_scores[np.isfinite(f1_scores)])]
    threshold=(threshold*(max_y_pred-min_y_pred)+min_y_pred).astype(np.float32)
    return best_f_score,threshold

def alert_delay(y_pred_window, threshold):
    for i in range(y_pred_window):
        if y_pred_window[i]<threshold:
            return i

def plot_anomaly_detect(result, kpi, label, threshold=None):
    x=np.linspace(-12,12, len(result))
    if threshold==None:
        _, threshold=best_F_score(result, label)
    plt.plot(x,kpi,color="black")
    _x=[]
    _y=[]
    print(threshold)
    for i in range(len(result)):
        if(result[i]<threshold):
            _x.append(x[i])
            _y.append(kpi[i])
    plt.scatter(_x,_y,c='r',marker='x')
    _x=[]
    _y=[]
    for i in range(len(result)):
        if label[i]==1:
            _x.append(x[i])
            _y.append(kpi[i])
    plt.scatter(_x,_y,s=15,c='b')
    plt.show()
'''
draw all point
'''
f=open("result\cpu4.txt")
data=json.load(f)
print(best_F_score(data['result'], data['label']))
print(auc_score(data['result'], data['label']))
plot_anomaly_detect(data['result'], data['kpi'], data['label'])

'''
draw z sample point 2dim
'''
color=["red","orangered","darkorange","orange","gold","yellow","lawngreen","green","cyan","deepskyblue","dodgerblue","royalblue","blue","blueviolet","violet","purple","magenta","deeppink","crimson"]
x=[]
y=[]
# for i in data:
#     x.append(i[0][0])
#     y.append(i[0][1])
# for c in range(len(color)):
#     length=int(len(x)/len(color))
#     _x=x[c*length:(c+1)*length]
#     _y=y[c*length:(c+1)*length]
#     plt.scatter(_x,_y,s=10,c=color[c])
# plt.show()

'''
draw z sample point 3dim
'''
# x=[]
# y=[]
# z=[]
# for i in data:
#     x.append(i[0][0])
#     y.append(i[0][1])
#     z.append(i[0][2])
# ax = plt.axes(projection ="3d")
# for c in range(len(color)):
#     length=int(len(x)/len(color))
#     _x=x[c*length:(c+1)*length]
#     _y=y[c*length:(c+1)*length]
#     _z=z[c*length:(c+1)*length]
#     ax.scatter3D(_x,_y,_z,s=10,c=color[c])
# plt.show()

'''
draw bar with F_score and AUC
'''
# y=[0.8798092227012037,0.9190723176216946,0.8113385314405656,0.48235403586316267]
# x=np.arange(4)+1
# plt.bar(x,y,0.5)
# plt.xticks(ticks=range(1,5),
#     labels=["cpu4","g","server_res_eth1out_curve_6","server_res_eth1out_curve_61"])
# plt.ylabel("AUC")
# plt.title("Anomaly point rate:1%")
# plt.show()

'''
draw best f-score or AUC with z-dims
'''
# x=[1,2,3,4,5,8,13,21]
# y1=[0.6450727401158779,0.7782664567685073,0.878024162180664,0.8877646382836695,0.8833962588690604,0.8855824357766251,0.872391698064391,0.8919888995917619]
# y2=[0.742098516987851,0.8215057333525667,0.851903042384372,0.8847995448677984,0.8780837851603774,0.8921071351251044,0.8780116880716208,0.8838122684916034]
# y3=[0.6660615275172092,0.7749682330883593,0.8009312834271795,0.8637323238007912,0.8540734164374129,0.853804498900726,0.847041821942054,0.846730591983848]
# plt.plot(x,y1,c="r")
# plt.scatter(x, y1,c="r",marker="^")
# plt.plot(x,y2,c="b")
# plt.scatter(x, y2,c="b",marker="v")
# plt.plot(x,y3,c="green")
# plt.scatter(x, y3,c="green",marker=".")
# plt.ylim(0,1)
# plt.xlim(1,21)
# plt.xlabel("Z_dims")
# plt.ylabel("AUC")
# plt.xticks(x)
# plt.grid(linewidth=1)
# plt.show()


