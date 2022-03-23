# Classification in Scikit Learn

Types of classification:

- sklearn.linear_model.RidgeClassifier<br>
  class sklearn.linear_model.RidgeClassifier(alpha=1.0, *, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, class_weight=None, solver='auto', positive=False, random_state=None)
  
- sklearn.linear_model.RidgeClassifierCV <br>
  class sklearn.linear_model.RidgeClassifierCV(alphas=(0.1, 1.0, 10.0), *, fit_intercept=True, normalize='deprecated', scoring=None, cv=None, class_weight=None, store_cv_values=False)
  
- sklearn.discriminant_analysis.LinearDiscriminantAnalysis<br>
  class sklearn.discriminant_analysis.LinearDiscriminantAnalysis(solver='svd', shrinkage=None, priors=None, n_components=None, store_covariance=False, tol=0.0001, covariance_estimator=None)
  
- sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis<br>
  class sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(*, priors=None, reg_param=0.0, store_covariance=False, tol=0.0001)
  
- sklearn.svm.SVC<br>
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)

- sklearn.svm.NuSVC<br>
class sklearn.svm.NuSVC(*, nu=0.5, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=- 1, decision_function_shape='ovr', break_ties=False, random_state=None)

- sklearn.svm.LinearSVC<br>
class sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', *, dual=True, tol=0.0001, C=1.0, multi_class='ovr', fit_intercept=True, intercept_scaling=1, class_weight=None, verbose=0, random_state=None, max_iter=1000)

- sklearn.linear_model.SGDClassifier<br>
class sklearn.linear_model.SGDClassifier(loss='hinge', *, penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000, tol=0.001, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight=None, warm_start=False, average=False)

- sklearn.neighbors.KNeighborsClassifier<br>
class sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)

- sklearn.neighbors.RadiusNeighborsClassifier<br>
class sklearn.neighbors.RadiusNeighborsClassifier(radius=1.0, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None, n_jobs=None, **kwargs)

- sklearn.gaussian_process.GaussianProcessClassifier<br>
class sklearn.gaussian_process.GaussianProcessClassifier(kernel=None, *, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None)

- sklearn.naive_bayes.CategoricalNB<br>
class sklearn.naive_bayes.CategoricalNB(*, alpha=1.0, fit_prior=True, class_prior=None, min_categories=None)

- sklearn.naive_bayes.MultinomialNB<br>
class sklearn.naive_bayes.MultinomialNB(*, alpha=1.0, fit_prior=True, class_prior=None)

- sklearn.naive_bayes.BernoulliNB<br>
class sklearn.naive_bayes.BernoulliNB(*, alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)

- sklearn.naive_bayes.GaussianNB<br>
class sklearn.naive_bayes.GaussianNB(*, priors=None, var_smoothing=1e-09)

- sklearn.tree.DecisionTreeClassifier<br>
class sklearn.tree.DecisionTreeClassifier(*, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)

- sklearn.ensemble.RandomForestClassifier<br>
class sklearn.ensemble.RandomForestClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

- sklearn.ensemble.ExtraTreesClassifier<br>
class sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)

- sklearn.ensemble.VotingClassifier<br>
class sklearn.ensemble.VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)

- sklearn.multiclass.OneVsRestClassifier<br>
class sklearn.multiclass.OneVsRestClassifier(estimator, *, n_jobs=None)

- sklearn.multiclass.OutputCodeClassifier<br>
class sklearn.multiclass.OutputCodeClassifier(estimator, *, code_size=1.5, random_state=None, n_jobs=None)

- sklearn.multioutput.MultiOutputClassifier<br>
class sklearn.multioutput.MultiOutputClassifier(estimator, *, n_jobs=None)

- sklearn.multioutput.ClassifierChain<br>
class sklearn.multioutput.ClassifierChain(base_estimator, *, order=None, cv=None, random_state=None)

- sklearn.calibration.CalibratedClassifierCV<br>
class sklearn.calibration.CalibratedClassifierCV(base_estimator=None, *, method='sigmoid', cv=None, n_jobs=None, ensemble=True)

- sklearn.neural_network.MLPClassifier<br>
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,), activation='relu', *, solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)

