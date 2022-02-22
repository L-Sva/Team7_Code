from core import RAWFILES
from ml_main import combine_signal_background, load_model_file, load_train_validate_test, fit_new_model
from ml_main import make_selector, optimize_threshold, predict_prob, plot_features, plot_sb
from ml_tools import ml_strip_columns, test_false_true_negative_positive
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import xgboost

def genOutlineSegments(mapimg, x0, x1, y0, y1):
    # Curtosy of:
    # https://stackoverflow.com/questions/24539296/outline-a-region-in-a-graph
    # a vertical line segment is needed, when the pixels next to each other horizontally
    # belong to diffferent groups (one is part of the mask, the other isn't)
    # after this ver_seg has two arrays, one for row coordinates, the other for column coordinates 
    ver_seg = np.where(mapimg[:,1:] != mapimg[:,:-1])

    # the same is repeated for horizontal segments
    hor_seg = np.where(mapimg[1:,:] != mapimg[:-1,:])

    # if we have a horizontal segment at 7,2, it means that it must be drawn between pixels
    #   (2,7) and (2,8), i.e. from (2,8)..(3,8)
    # in order to draw a discountinuous line, we add Nones in between segments
    l = []
    for p in zip(*hor_seg):
        l.append((p[1], p[0]+1))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan,np.nan))

    # and the same for vertical segments
    for p in zip(*ver_seg):
        l.append((p[1]+1, p[0]))
        l.append((p[1]+1, p[0]+1))
        l.append((np.nan, np.nan))

    # now we transform the list into a numpy array of Nx2 shape
    segments = np.array(l)

    # now we need to know something about the image which is shown
    #   at this point let's assume it has extents (x0, y0)..(x1,y1) on the axis
    #   drawn with origin='lower'
    # with this information we can rescale our points

    segments[:,0] = x0 + (x1-x0) * segments[:,0] / mapimg.shape[1]
    segments[:,1] = y0 + (y1-y0) * segments[:,1] / mapimg.shape[0]

    return segments[:,0], segments[:,1]

signal = load_train_validate_test(RAWFILES.SIGNAL, validate_size=0)
bk = load_train_validate_test(RAWFILES.PSI2S, validate_size=0)
train, test = combine_signal_background(signal, bk)

VAR_A = 'q2'
VAR_B = 'B0_MM'

names = [name for name in signal[0] if not name in ('category', VAR_A, VAR_B)]

fig, axs = plt.subplots(1,2,sharey=True)
plt.sca(axs[0])
plt.hist2d(signal[1][VAR_A].to_numpy(),signal[1][VAR_B].to_numpy(),bins=(50,100))
plt.sca(axs[1])
_, edgesX, edgesY, _ = plt.hist2d(bk[1][VAR_A].to_numpy(),bk[1][VAR_B].to_numpy(),bins=(50,100))
plt.close()

ML_SAVE_DIR = 'ml_models'
MODEL_PATH = os.path.join(ML_SAVE_DIR,'0000_bivariate_quick.model')

if not os.path.exists(MODEL_PATH):
    print('Must train new model')
    model = fit_new_model(train[:30000], reject_column_names=names, params = {
        'n_estimators': 100,
        'max_depth': 3,
        'reg_alpha': 0,
        'reg_lambda': 1,
    })
    model.save_model(MODEL_PATH)
else:
    model = load_model_file(MODEL_PATH)

print('Model ready')

sig_prob = predict_prob(test, model, reject_column_names=names)

thresh = optimize_threshold(test, sig_prob, bk_penalty=1)
print('ML:',test_false_true_negative_positive(test, sig_prob, thresh))
alt_sig = (~((test[VAR_A] > 13.25) & (test[VAR_A] < 14) & (test[VAR_B] > 5245) & (test[VAR_B] < 5345))).astype('int')
print('Select:',test_false_true_negative_positive(test, alt_sig, 0.5))
selector = make_selector(model, thresh, reject_column_names=names)

print('Selector ready')

centers_X = (edgesX[:-1] + edgesX[1:]) / 2
centers_Y = (edgesY[:-1] + edgesY[1:]) / 2
xx, yy = np.meshgrid(centers_X, centers_Y)

data = np.stack((yy.flat,xx.flat), axis=-1)
data = pd.DataFrame(data, columns=(VAR_B,VAR_A))
data['category'] = (predict_prob(data, model, names) > thresh).astype('int')
data['category_2'] = (
    (data[VAR_A] > 13.25) & (data[VAR_A] < 14) & (data[VAR_B] > 5245) & (data[VAR_B] < 5335)
).astype('int')

data_plt = data['category'].to_numpy().reshape(xx.shape)
data_plt_2 = data['category_2'].to_numpy().reshape(xx.shape)
tuple_edges = edgesX[0], edgesX[-1], edgesY[0], edgesY[-1]
tuple_centers = centers_X[0], centers_X[-1], centers_Y[0], centers_Y[-1]
ml_boundary = genOutlineSegments(data_plt, *tuple_edges)
selector_boundary = genOutlineSegments(data_plt_2, *tuple_edges)

fig, axs = plt.subplots(1,2,sharey=True,sharex=True)

plt.sca(axs[0])
plt.hist2d(signal[1][VAR_A].to_numpy(),signal[1][VAR_B].to_numpy(),bins=(50,100))
plt.plot(*ml_boundary,color=(0.8,0,0,.7), linewidth=2,label='ML boundary')
plt.plot(*selector_boundary,color=(0.8,0.8,0,.7), linewidth=2,label='Selector boundary')
plt.title('Signal')
plt.ylabel('B0_MM')
plt.xlabel('q2')
plt.legend(loc='lower right')

plt.sca(axs[1])
plt.hist2d(bk[1][VAR_A].to_numpy(),bk[1][VAR_B].to_numpy(),bins=(50,100))
plt.plot(*ml_boundary,color=(0.8,0,0,.7), linewidth=2,label='ML boundary')
plt.plot(*selector_boundary,color=(0.8,0.8,0,.7), linewidth=2,label='Selector boundary')
plt.title('PSI2S background')
plt.xlabel('q2')

plt.ylim(5220,5340)
plt.xlim(12.4,14.5)
plt.show()

model.get_booster().feature_names = [VAR_B,VAR_A]

# xgboost.plot_tree(model, rankdir='LR')
# plt.show()



