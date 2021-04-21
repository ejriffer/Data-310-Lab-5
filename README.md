# Data-310-Lab-5

## Question 4: For the breast cancer data (from sklearn library), if you choose a test size of 0.25 (25% of your data), with a random_state of 1693, how many observations are in your training set?

dat = load_breast_cancer()

X = pd.DataFrame(data=dat.data, columns=dat.feature_names)

y = dat.target

x_train,x_test,y_train,y_test = tts(X,y, test_size = 0.25, random_state = 1693)

len(x_train) = 426

## Question 6: Using your Kernel SVM model with a radial basis function kernel, predict the classification of a tumor if it has a radius mean of 16.78 and a texture mean of 17.89.

x1 = pd.DataFrame(X['mean radius'])

x2 = pd.DataFrame(X['mean texture'])

x = pd.concat([x1,x2], axis = 1)

x = np.array(x)

model = SVC(kernel='rbf', C=1 ,gamma='auto')

scale = StandardScaler()

pipe = Pipeline([('scale', scale),('model',model)])

pipe.fit(x, y)

predicted_classes = model.predict(x)

rm = 16.78

tm = 17.89

model.predict([[rm,tm]]) = array([0])

## Question 7: Using your logistic model, predict the probability a tumor is malignant if it has a radius mean of 15.78 and a texture mean of 17.89.

x1 = pd.DataFrame(X['mean radius'])

x2 = pd.DataFrame(X['mean texture'])

x = pd.concat([x1,x2], axis = 1)

x = np.array(x)

model = LogisticRegression()

scale = StandardScaler()

pipe = Pipeline([('scale', scale),('model',model)])

pipe.fit(x, y)

predicted_classes = pipe.predict(x)

rm = 15.78

tm = 17.89

pipe.predict_proba([[rm,tm]]) = array([[0.64134136, 0.35865864]])

## Question 8: Using your nearest neighbor classifier with k=5 and weights='uniform', predict if a tumor is benign or malignant if the Radius Mean  is 17.18, and the Texture Mean is 8.65

model = neighbors.KNeighborsClassifier(n_neighbors = 5, weights='uniform')

model.fit(x, y)

predicted_classes = model.predict(x)

rm = 17.18

tm = 8.65

model.predict_proba([[rm,tm]]) = array([[0.4, 0.6]])

## Question 9: Consider a RandomForest classifier with 100 trees, max depth of 5 and random state 1234. From the data consider only the "mean radius" and the "mean texture" as the input features. If you apply a 10-fold stratified cross-validation and estimate the mean AUC (based on the receiver operator characteristics curve) the answer is

kf = KFold(n_splits=10, random_state=1234,shuffle=True)

model = rfc(n_estimators=100,max_depth=5,random_state=1234)

scale = StandardScaler()

#ctscan = datasets.load_breast_cancer()

#X = ctscan.data

cv = StratifiedKFold(n_splits=10)

classifier = rfc(n_estimators=100,max_depth=5,random_state=1234)

tprs = []

aucs = []

mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(10,8))

for i, (train, test) in enumerate(cv.split(x, y)):

    classifier.fit(x[train], y[train])
    
    viz = plot_roc_curve(classifier, x[test], y[test],
    
                         name='ROC fold {}'.format(i),
                         
                         alpha=0.3, lw=1, ax=ax)
                         
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    
    interp_tpr[0] = 0.0
    
    tprs.append(interp_tpr)
    
    aucs.append(viz.roc_auc)

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',

        label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)

mean_tpr[-1] = 1.0

mean_auc = auc(mean_fpr, mean_tpr)

std_auc = np.std(aucs)

ax.plot(mean_fpr, mean_tpr, color='b',

        label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
        
        lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)

tprs_upper = np.minimum(mean_tpr + std_tpr, 1)

tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,

                label=r'$\pm$ 1 std. dev.')

ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],

       title="Receiver operating characteristic example")
       
ax.legend(loc="lower right")

plt.show()

<img width="631" alt="Screen Shot 2021-04-21 at 10 30 34 AM" src="https://user-images.githubusercontent.com/74326062/115571045-9db5f180-a28c-11eb-8460-d7e6ab454320.png">
